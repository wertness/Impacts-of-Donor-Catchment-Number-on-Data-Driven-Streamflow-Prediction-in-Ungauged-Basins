#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from typing import Optional
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# --- Unified plotting style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "axes.unicode_minus": False,
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
})

# ---------- Basic utilities ----------
def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in (None, "utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)  # Raise the error if all attempts fail

def project_lonlat(df: pd.DataFrame, lon: str, lat: str, crs_out: str) -> np.ndarray:
    # Note: this uses projected planar distance in meters, not exact geodesic distance
    t = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    x, y = t.transform(df[lon].to_numpy(), df[lat].to_numpy())
    return np.vstack([x, y]).T

def auto_pick_target(df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    """从表中自动选择一个“相似样本数/邻居数”列作为横轴数据列。"""
    if prefer and prefer in df.columns:
        return prefer
    cands = [
        "max_geo_n","max_geo_N",
        "max_geo_n_x","max_geo_n_y",
        "max_geo_n","max_geo_n",
    ]
    for c in cands:
        if c in df.columns:
            return c
    lowers = {c.lower(): c for c in df.columns}
    for key in ["max_similarity_n","max_similarity","similarity","max_geo_n","geo_n","neighbors","k"]:
        for k, orig in lowers.items():
            if key in k:
                return orig
    sample_cols = ", ".join(list(df.columns)[:12]) + ("..." if df.shape[1] > 12 else "")
    raise ValueError(f"未找到可用的横轴数据列（例如 max_similarity_n / max_geo_n / similarity 等）。可见列示例：{sample_cols}")

def make_weights(df: pd.DataFrame,
                 weight_col: Optional[str],
                 weight_expr: Optional[str]) -> Optional[np.ndarray]:
    s = None
    if weight_expr:
        s = df.eval(weight_expr, engine="python",
                    local_dict={"log1p": np.log1p, "log": np.log, "log10": np.log10,
                                "abs": np.abs, "clip": np.clip,
                                "minimum": np.minimum, "maximum": np.maximum})
    elif weight_col:
        if weight_col not in df.columns:
            raise ValueError(f"找不到权重列：{weight_col}")
        s = df[weight_col]
    if s is None:
        return None
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    mn, mx = float(np.min(s)), float(np.max(s))
    return (s - mn) / max(mx - mn, 1e-12) if mx > mn else np.zeros_like(s)

# ---------- Point-wise station density (excluding self) ----------
def station_density_pointwise(
    xy: np.ndarray,
    radius_m: float,
    kernel: str = "uniform",          # uniform/gaussian/exponential/linear/inverse
    bandwidth: Optional[float] = None,# sigma or lambda; use R/3 when None
    inv_power: float = 1.0,
    weights: Optional[np.ndarray] = None,  # Weights applied to neighboring points; use equal weights when None
    normalize: bool = False,          # True = local average; False = density (weighted sum)
    eps: float = 1.0,
) -> np.ndarray:
    n = xy.shape[0]
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    bw = max((radius_m / 3.0 if bandwidth is None else float(bandwidth)), eps)
    tree = cKDTree(xy, leafsize=40)
    dens = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float) if normalize else None

    for i in range(n):
        nbrs = tree.query_ball_point(xy[i], r=radius_m)
        nbrs = [j for j in nbrs if j != i]  # exclude the focal point itself
        if not nbrs:
            continue
        nbrs = np.asarray(nbrs, dtype=int)
        d = np.linalg.norm(xy[nbrs] - xy[i], axis=1)

        k = kernel.lower()
        if k == "uniform":
            K = (d <= radius_m).astype(float)
        elif k == "gaussian":
            K = np.exp(-0.5 * (d / bw) ** 2); K[d > radius_m] = 0.0
        elif k == "exponential":
            K = np.exp(-d / bw);             K[d > radius_m] = 0.0
        elif k == "linear":
            K = np.maximum(0.0, 1.0 - d / radius_m)
        elif k == "inverse":
            K = 1.0 / np.power(d + eps, inv_power); K[d > radius_m] = 0.0
        else:
            raise ValueError("kernel 必须为 uniform/gaussian/exponential/linear/inverse")

        dens[i] = np.sum(w[nbrs] * K)
        if normalize:
            denom[i] = np.sum(K)

    if normalize:
        dens = np.divide(dens, np.maximum(denom, 1e-12))

    # Rescale to [0, 1]
    mn, mx = float(np.nanmin(dens)), float(np.nanmax(dens))
    return (dens - mn) / max(mx - mn, 1e-12) if mx > mn else np.zeros_like(dens)

# ---------- Plotting (0?50 and 50?600) ----------
def plot_bins(df: pd.DataFrame, xcol: str, ycol: str,
              out_dir: str, base: str, ymax: Optional[float] = 1.0):
    """按 xcol 分组绘制箱线图，两段：0–50 与 50–600，并标注相关性 + 图例"""
    def _plot(xmin, xmax, left_open, fname, title):
        # Filter the data
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = (x > xmin) & (x <= xmax) if left_open else (x >= xmin) & (x <= xmax)
        x, y = x[m], y[m]
        if not len(x):
            print(f"[Plot] {title} 无数据"); return

        # Bin the values by integer
        xi = x.round().astype("Int64")
        vmin, vmax = int(xi.min()), int(xi.max())
        groups, labels = [], []
        for v in range(vmin, vmax+1):
            vals = y[xi == v].to_numpy()
            if vals.size >= 2:
                groups.append(vals); labels.append(str(v))
        if not groups:
            print(f"[Plot] {title} 每箱样本不足"); return

        # Spearman correlation for the current range
        from scipy.stats import spearmanr
        xy = pd.DataFrame({"x": x, "y": y}).dropna()
        rho = pval = np.nan; n = len(xy)
        if n >= 3 and xy.nunique().min() > 1:
            rho, pval = spearmanr(xy["x"], xy["y"], nan_policy="omit")

        # Draw the figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bp = ax.boxplot(groups, vert=True, patch_artist=True, widths=0.35,
                        showmeans=True, meanline=True)

        # Color settings
        maroon = (240/255, 65/255, 85/255)   # dark red for boxes and the mean
        green  = "tab:green"                 # mean line
        black  = (0, 0, 0)                   # black for the median line
        gray   = (0.4, 0.4, 0.4)

        # boxes
        for box in bp["boxes"]:
            box.set_facecolor(maroon)
            box.set_alpha(0.75)
            box.set_edgecolor(gray)
            box.set_linewidth(1.2)

        # whiskers and caps
        for line in bp["whiskers"] + bp["caps"]:
            line.set_linewidth(1.2)
            line.set_color(gray)

        # median line
        for line in bp["medians"]:
            line.set_linewidth(2.0)
            line.set_color(black)

        # mean line
        for line in bp["means"]:
            line.set_linewidth(2.0)
            line.set_color(green)

        # legend
        median_proxy = plt.Line2D([], [], color=black, linewidth=2, label="Median")
        mean_proxy   = plt.Line2D([], [], color=green, linewidth=2, label="Mean")
        ax.legend(handles=[median_proxy, mean_proxy],
                  loc="upper right", frameon=True, fontsize=11)

        # Axis labels and title (x-axis text fixed as max_geo_n; title passed as an argument)
        ax.set_xlabel(" geo OPN")   # only change the display label, not the underlying data column
        ax.set_ylabel(ycol)
        ax.set_title(title)          # use the passed-in title instead of a hard-coded title

        ax.set_xticks(np.arange(1, len(labels)+1)); ax.set_xticklabels(labels)
        if ymax is not None: ax.set_ylim(0, ymax)
        ax.grid(True, ls="--", alpha=0.35)

        # Annotate the Spearman result on the figure
        if not np.isnan(rho):
            ax.text(0.02, 0.98,
                    f"Spearman ρ={rho:.3f}\np={pval:.3g}\nn={n}",
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize=11, color=black,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=gray, alpha=0.8))
        else:
            ax.text(0.02, 0.98, f"Spearman ρ=N/A\nn={n}",
                    ha="left", va="top", transform=ax.transAxes,
                    fontsize=11, color=black,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=gray, alpha=0.8))

        # Save the figure
        path = os.path.join(out_dir, fname)
        plt.tight_layout(); plt.savefig(path); plt.show()
        print(f"[OK] 保存图像: {path}")

    # Two ranges: 0?50 and (50?600] ? use the requested title format
    _plot(0, 50, False, f"{base}_density_box_0_50.png", "")
    _plot(50, 600, True,  f"{base}_density_box_50_600.png", "")

# ---------- Main workflow ----------
def run(csv_path: str, lon: str, lat: str,
        radius_m: float, crs_out: str,
        kernel: str, scale_m: Optional[float], inv_power: float,
        normalize: bool, filter_query: Optional[str],
        weight_col: Optional[str], weight_expr: Optional[str],
        ymax: Optional[float]):
    df_all = load_table(csv_path)
    for c in (lon, lat):
        if c not in df_all.columns:
            raise ValueError(f"缺列：{c}")
    df = df_all if not filter_query else df_all.query(filter_query, engine="python")
    df = df.dropna(subset=[lon, lat]).copy()
    if df.empty:
        raise RuntimeError("经纬度为空，筛选后无记录。")

    # Optional weights
    w = make_weights(df, weight_col, weight_expr)

    # Projection and density calculation (planar distance in meters, not geodesic)
    xy = project_lonlat(df, lon, lat, crs_out)
    density = station_density_pointwise(
        xy=xy, radius_m=radius_m, kernel=kernel,
        bandwidth=scale_m, inv_power=inv_power,
        weights=w, normalize=normalize, eps=1.0
    )
    df["station_density"] = density

    # Choose the x-axis data column: use max_similarity_n if available, otherwise select one automatically
    xcol = "max_similarity_n" if "max_similarity_n" in df.columns else auto_pick_target(df, None)
    print(f"[XCOL] 用数据列 '{xcol}' 作为横轴；图上标签显示为 'max_geo_n'。")

    # Spearman correlation using xcol and station_density
    pair = df[[xcol, "station_density"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) >= 3 and pair.nunique().min() > 1:
        rho, p = spearmanr(pair["station_density"], pair[xcol], nan_policy="omit")
        print(f"[Spearman] rho={rho:.6g}, p={p:.6g}, n={len(pair)}")
    else:
        print("[Spearman] 数据不足或无变化，跳过。")

    # Save the CSV output
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"{base}_points_with_station_density.csv")
    keep = [c for c in [
        ("gauge_id" if "gauge_id" in df.columns else None),
        ("site_id"  if "site_id"  in df.columns else None),
        ("id"       if "id"       in df.columns else None),
        lon, lat, "station_density", xcol
    ] if c]
    df.loc[:, keep].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已导出: {out_csv}")

    # Generate the plots using xcol for binning
    plot_bins(df, xcol=xcol, ycol="station_density",
              out_dir=out_dir, base=base, ymax=ymax)

# ---------- Command-line interface ----------
def parse_args():
    p = argparse.ArgumentParser(description="站点密度（点上核密度，排除自身）→ 提取到点并分组出图（图题已改为 density_boxplot(...)）")
    p.add_argument("--csv", default=r"../data/csv/attr_density_with_target.csv")
    p.add_argument("--lon", default="gauge_lon")
    p.add_argument("--lat", default="gauge_lat")
    p.add_argument("--radius-m", type=float, default=300000, help="邻域半径（米）")
    p.add_argument("--crs", default="EPSG:3857", help="距离度量投影（平面距离，非测地线）")
    p.add_argument("--kernel", default="uniform",
                   choices=["uniform","gaussian","exponential","linear","inverse"],
                   help="距离核类型")
    p.add_argument("--scale-m", type=float, default=None, help="gaussian σ / exponential λ（默认 R/3）")
    p.add_argument("--inv-power", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true", help="返回局部平均；默认关闭=密度")
    p.add_argument("--filter", default=None, help="pandas.query 语法")
    p.add_argument("--weight-col", default=None)
    p.add_argument("--weight-expr", default=None)
    p.add_argument("--ymax", type=float, default=1.0, help="<0 表示不限制")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(csv_path=args.csv, lon=args.lon, lat=args.lat,
        radius_m=args.radius_m, crs_out=args.crs,
        kernel=args.kernel, scale_m=args.scale_m, inv_power=args.inv_power,
        normalize=args.normalize, filter_query=args.filter,
        weight_col=args.weight_col, weight_expr=args.weight_expr,
        ymax=(None if (args.ymax is not None and args.ymax < 0) else args.ymax))
