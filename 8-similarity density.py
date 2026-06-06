#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 点 → 6维属性 KDE 权重 × 点上空间核密度（无栅格热力图）
→ 把 kde_value 写回每个点 → 与 max_similarity_n 做 Spearman 相关
→ 画两张箱线图（0–50 与 50–600），并在图左上角标注 Spearman ρ、p、n

固定 6 个属性列：
    ["area_x","aridity_fao_pm","kar_pc_sse_x","slp_dg_sav_x","for_pc_sse","p_mean"]
"""

from __future__ import annotations
import argparse
import os
from typing import Optional, List
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde, spearmanr
import matplotlib.pyplot as plt

# ------- Unified plotting style (18 pt font and narrower boxes) -------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "axes.unicode_minus": False,
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# Fixed set of six attribute columns
ATTR_COLS = ["area_x","aridity_fao_pm","kar_pc_sse_x","slp_dg_sav_x","for_pc_sse","p_mean"]

# ============== I/O and general utilities ==============
def load_points_from_csv(csv_path: str) -> pd.DataFrame:
    """读取 CSV/Excel，自动尝试常见编码（CSV）。"""
    lower = csv_path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(csv_path)
    last_err = None
    for enc in (None, "utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def project_lonlat_to_crs(df: pd.DataFrame, lon_col: str, lat_col: str, dst_crs: str) -> np.ndarray:
    """WGS84 经纬度投影到目标 CRS（用于米制邻域半径）"""
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform(df[lon_col].to_numpy(), df[lat_col].to_numpy())
    return np.vstack([xs, ys]).T

def auto_pick_target_col(df: pd.DataFrame, prefer: str | None = None) -> str:
    """自动识别 max_similarity_n 列名（兼容大小写/后缀/拼写差异）。"""
    if prefer and prefer in df.columns:
        return prefer
    candidates = [
        "max_similarity_n", "max_similarity_N", "max_similarity_n_x", "max_similarity_N_x",
        "max_similarity_n_y", "max_similarity_N_y", "max_geo_n", "max_similairy_n",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lowers = {c.lower(): c for c in df.columns}
    for key in ["max_similarity_n", "max_similarity", "similarity", "max_geo_n"]:
        for k, orig in lowers.items():
            if key in k:
                return orig
    raise ValueError("未找到目标列（例如 max_similarity_n / max_geo_n）。请检查输入表头。")

# ============== 6D attribute KDE weights ==============
def kde_attr_weights(df: pd.DataFrame, cols: List[str],
                     zscore: bool=True, bw: str|float="scott",
                     out_norm: str="minmax") -> np.ndarray:
    """
    对给定属性列做 gaussian_kde，返回每个点的属性空间密度（作为权重）。
    - zscore：对每个属性做标准化
    - bw：'scott'/'silverman' 或数值（传给 scipy 的 bw_method）
    - out_norm：'minmax' / 'zscore' / 'none'
    """
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if zscore:
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std==0, 1.0, std)
        X = (X - mu)/std
    data = X.T  # (d, n)
    try:
        if isinstance(bw, (int,float)):
            kde = gaussian_kde(data, bw_method=float(bw))
        else:
            kde = gaussian_kde(data, bw_method=bw)
        dens = kde(data)
    except Exception:
        # Fallback for collinearity or singular covariance: small perturbation plus a wider bandwidth
        rng = np.random.default_rng(42)
        data_perturb = data + 1e-6 * rng.standard_normal(data.shape)
        kde = gaussian_kde(data_perturb, bw_method="silverman")
        dens = kde(data_perturb)

    w = dens.astype(float)
    if out_norm == "minmax":
        mn, mx = float(w.min()), float(w.max())
        w = (w - mn) / max(mx - mn, 1e-12)
    elif out_norm == "zscore":
        w = (w - w.mean()) / max(w.std(ddof=1), 1e-12)
        w = (w - w.min()) / max(w.max() - w.min(), 1e-12)
    elif out_norm == "none":
        pass
    else:
        raise ValueError("out_norm 只能为 minmax / zscore / none")
    return w

# ============== Point-wise spatial kernel density (no raster) ==============
def pointwise_spatial_kde(
    xy: np.ndarray,
    weights: Optional[np.ndarray] = None,
    radius_m: float = 100000.0,
    kernel: str = "gaussian",     # supported kernels: 'gaussian' / 'circle' / 'combo'
    normalize: bool = True,
    bandwidth: Optional[float] = None,  # None means h = radius_m / 2 (used only for gaussian/combo)
    leafsize: int = 40,
    batch_divisor: int = 20,
) -> np.ndarray:
    """
    在每个点上计算空间核密度：
    kde_i = sum_j w_j * K( dist(x_i,x_j) / h )
    - 'circle'：K(u) = 1(u<=1)
    - 'gaussian'：K(u) = exp(-0.5*u^2)  (u = d / h)
    - 'combo'：K = 0.5*gaussian + 0.5*circle
    normalize=True 时再除以 sum_j K(...)（得到局部加权均值，数值更稳）
    最后做一次 [0,1] 的 minmax 归一化。
    """
    n = xy.shape[0]
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    tree = cKDTree(xy, leafsize=leafsize)

    if bandwidth is None:
        h = max(radius_m / 2.0, 1e-6)
    else:
        h = max(float(bandwidth), 1e-6)

    num = np.zeros(n, dtype=float)   # weighted sum
    den = np.zeros(n, dtype=float)   # sum of kernel weights

    batch = max(1, n // max(batch_divisor, 1))
    for start in range(0, n, batch):
        end = min(n, start + batch)
        idxs_list = tree.query_ball_point(xy[start:end], r=radius_m)

        for offset, nbrs in enumerate(idxs_list):
            i = start + offset
            if not nbrs:
                continue
            nbrs = np.asarray(nbrs, dtype=int)
            d = np.linalg.norm(xy[nbrs] - xy[i], axis=1)

            if kernel == "circle":
                K = (d <= radius_m).astype(float)
            elif kernel == "gaussian":
                u = d / h
                K = np.exp(-0.5 * u * u)
            elif kernel == "combo":
                u = d / h
                Kg = np.exp(-0.5 * u * u)
                Ku = (d <= radius_m).astype(float)
                K = 0.5 * Kg + 0.5 * Ku
            else:
                raise ValueError("kernel 必须是 gaussian / circle / combo")

            num[i] = np.sum(w[nbrs] * K)
            den[i] = np.sum(K)

    kde = np.divide(num, np.maximum(den, 1e-12)) if normalize else num
    mn, mx = float(np.nanmin(kde)), float(np.nanmax(kde))
    kde = (kde - mn) / max(mx - mn, 1e-12)
    return kde

# ============== Plotting: two boxplots (0?50 and 50?600) ==============
def plot_kde_boxplots(df: pd.DataFrame, target_col: str, out_dir: str, base_name: str,
                      ymax: float | None = 1.0, min_count_per_box: int = 2,
                      widths: float = 0.35):
    """
    画两张图：
      - 图1：x ∈ [0,50] 的箱线图（逐整数分箱），标题 density_boxplot(0-50)
      - 图2：x ∈ (50,600] 的箱线图（逐整数分箱），标题 density_boxplot(50-600)
    保存到 out_dir。并在**左上角**标注该区间 Spearman ρ、p、n。
    """
    xraw = pd.to_numeric(df[target_col], errors="coerce")
    y = pd.to_numeric(df["kde_value"], errors="coerce")
    ok = xraw.notna() & y.notna()
    xraw = xraw[ok]; y = y[ok]

    def _boxplot_for_range(xmin, xmax, left_open, fname, title):
        # Range mask
        if xmax is None:
            mask = (xraw > xmin) if left_open else (xraw >= xmin)
        else:
            mask = ((xraw > xmin) if left_open else (xraw >= xmin)) & (xraw <= xmax)

        if mask.sum() == 0:
            print(f"[Plot] {title}（范围内无数据）跳过。")
            return

        xs = xraw[mask].round().astype(int)
        ys = y[mask]

        # Bin the data by integer values
        if xmax is None:
            x_levels = np.arange(xs.min(), xs.max()+1, 1, dtype=int)
            if left_open:
                x_levels = x_levels[x_levels > xmin]
        else:
            start = int(np.floor(xmin)) + (1 if left_open else 0)
            x_levels = np.arange(start, int(np.floor(xmax))+1, 1, dtype=int)

        groups, labels = [], []
        for xv in x_levels:
            vals = ys[xs == xv].to_numpy()
            if vals.size >= min_count_per_box:
                groups.append(vals)
                labels.append(str(xv))
        if not groups:
            print(f"[Plot] {title}（每箱样本不足，min_count={min_count_per_box}）跳过。")
            return

        # Spearman correlation within this range using the original xraw and y values
        xy_pair = pd.DataFrame({"x": xraw[mask], "y": y[mask]}).dropna()
        rho = pval = np.nan; n = len(xy_pair)
        if n >= 3 and xy_pair.nunique().min() > 1:
            rho, pval = spearmanr(xy_pair["x"], xy_pair["y"], nan_policy="omit")

        # Draw the figure
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax = fig.add_subplot(111)
        bp = ax.boxplot(groups, vert=True, patch_artist=True, widths=widths,
                        showmeans=True, meanline=True)

        # Light styling: mean in green and median in black
        for box in bp['boxes']:
            box.set_alpha(0.85)
            box.set_linewidth(1.2)
        for line in bp['whiskers'] + bp['caps']:
            line.set_linewidth(1.2)
        for line in bp['means']:
            line.set_linewidth(1.8)
            line.set_color("tab:green")
        for line in bp['medians']:
            line.set_linewidth(2.2)
            line.set_color("black")

        # Legend
        mean_proxy   = plt.Line2D([], [], color="tab:green", linewidth=1.8, label="Mean")
        median_proxy = plt.Line2D([], [], color="black",     linewidth=2.2, label="Median")
        ax.legend(handles=[median_proxy, mean_proxy], loc="upper right", frameon=True)

        # Axis labels and title
        ax.set_xlabel(" simsilaity OPN")   # fixed display label
        ax.set_ylabel("kde_value")
        ax.set_title(title)

        ax.set_xticks(np.arange(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=0)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Place the Spearman result in the upper-left corner at axes coordinates (0.02, 0.98)
        if not np.isnan(rho):
            ax.text(
                0.02, 0.98,
                f"Spearman ρ={rho:.3f}\np={pval:.3g}\nn={n}",
                ha="left", va="top", transform=ax.transAxes, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85)
            )
        else:
            ax.text(
                0.02, 0.98,
                f"Spearman ρ=N/A\nn={n}",
                ha="left", va="top", transform=ax.transAxes, color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85)
            )

        plt.tight_layout()
        png_path = os.path.join(out_dir, fname)
        fig.savefig(png_path)
        plt.show()
        print(f"[OK] 保存图像: {png_path}")

    # Figure 1: 0?50 (closed interval)
    _boxplot_for_range(0, 50, False,
                       f"{base_name}_kde_boxplot_0_50.png",
                       "")
    # Figure 2: 50?600 (left-open, right-closed interval)
    _boxplot_for_range(50, 600, True,
                       f"{base_name}_kde_boxplot_50_600.png",
                       "")

# ============== Main workflow (point-wise density only) ==============
def run(csv_path: str, lon_col: str, lat_col: str,
        radius_m: float, crs_out: str,
        kernel: str, normalize: bool,
        filter_query: Optional[str] = None,
        weight_expr: Optional[str] = None,
        weight_col: Optional[str] = None,
        ymax: Optional[float] = 1.0):
    # 1) Read the data and validate the inputs
    full_df = load_points_from_csv(csv_path)
    for c in (lon_col, lat_col, *ATTR_COLS):
        if c not in full_df.columns:
            raise ValueError(f"找不到列：{c}")

    # 2) Filter the data and drop missing longitude/latitude and attribute values
    df = full_df.copy()
    if filter_query:
        try:
            df = df.query(filter_query, engine="python")
        except Exception as e:
            raise ValueError(f"--filter 解析失败：{e}")
    df = df.dropna(subset=[lon_col, lat_col] + ATTR_COLS)
    if df.empty:
        raise RuntimeError("经纬度或 6 维属性存在空值，筛选后无记录。")

    # 3) Compute 6D attribute KDE weights
    w_attr = kde_attr_weights(df, ATTR_COLS, zscore=True, bw="scott", out_norm="minmax")

    # 4) Optional expression-based weights
    w_expr = None
    if weight_expr:
        try:
            s = df.eval(weight_expr, engine="python",
                        local_dict={"log1p": np.log1p, "log": np.log, "log10": np.log10,
                                    "abs": np.abs, "clip": np.clip,
                                    "minimum": np.minimum, "maximum": np.maximum})
            s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)
            w_expr = s.to_numpy(dtype=float)
        except Exception as e:
            raise ValueError(f"--weight-expr 解析失败：{e}")

    # 5) Optional explicit weights from a column
    w_col = None
    if weight_col is not None and weight_col in df.columns:
        w_col = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.nanmax(w_col) > np.nanmin(w_col):
            w_col = (w_col - np.nanmin(w_col)) / (np.nanmax(w_col) - np.nanmin(w_col))
        else:
            w_col = np.zeros_like(w_col)

    # Combine the final point weights
    w_final = w_attr.copy()
    if w_expr is not None:
        w_final *= w_expr
    if w_col is not None:
        w_final *= w_col
    mn, mx = float(np.nanmin(w_final)), float(np.nanmax(w_final))
    w_final = (w_final - mn) / max(mx - mn, 1e-12)

    # 6) Project the coordinates for meter-based radii
    xy = project_lonlat_to_crs(df, lon_col, lat_col, crs_out)

    # 7) Compute point-wise spatial kernel density (without raster output)
    kde_point = pointwise_spatial_kde(
        xy=xy,
        weights=w_final,
        radius_m=radius_m,
        kernel=kernel,          # supported kernels: 'gaussian' / 'circle' / 'combo'
        normalize=normalize,
        bandwidth=None,         # None => h = radius_m/2 for gaussian/combo
    )
    df["kde_value"] = kde_point

    # 8) Identify the target column and compute the overall Spearman correlation
    target_col = auto_pick_target_col(df, prefer=None)
    pair = df[[target_col, "kde_value"]].apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(pair) >= 3 and pair.nunique().min() > 1:
        rho, p = spearmanr(pair["kde_value"], pair[target_col], nan_policy="omit")
        print(f"[Spearman] 全局：rho = {rho:.6g}, p = {p:.6g}, n = {len(pair)}")
    else:
        print(f"[Spearman] 全局：有效配对不足（n={len(pair)}），无法计算相关。")

    # 9) Save the point-level CSV in the same directory as the input CSV
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"{base}_points_with_kde.csv")

    cols_to_save = [lon_col, lat_col, "kde_value"]
    if target_col not in cols_to_save:
        cols_to_save.append(target_col)
    for c in ["gauge_id", "site_id", "id"]:
        if c in df.columns and c not in cols_to_save:
            cols_to_save.insert(0, c)

    df_out = df.loc[:, [c for c in cols_to_save if c in df.columns]].copy()
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 已导出点采样 CSV: {out_csv}")

    # 10) Generate two boxplots (0?50 and 50?600) and annotate the range-specific Spearman result in the upper-left corner
    try:
        plot_kde_boxplots(df, target_col=target_col, out_dir=out_dir,
                          base_name=os.path.splitext(os.path.basename(out_csv))[0],
                          ymax=ymax, widths=0.35)
    except Exception as e:
        print(f"[Plot] 出图失败：{e}")

# ============== Command-line arguments ==============
def parse_args():
    p = argparse.ArgumentParser(description="CSV 点 → 点上空间核密度（无栅格），并输出 Spearman 相关与箱线图（左上角标注）")
    p.add_argument("--filter", dest="filter_query", default=None,
                   help="筛选条件（pandas query 语法），例：region=='US' & amount>0")
    p.add_argument("--weight-expr", dest="weight_expr", default=None,
                   help="表达式作为附加权重（会与 6D 属性权重相乘），例：log1p(loss)")
    p.add_argument("--weight-col", dest="weight_col", default=None,
                   help="列名作为附加权重（会与 6D 属性权重相乘），会自动缩放到[0,1]")
    p.add_argument("--radius", dest="radius_m", type=float, default=100000.0,
                   help="点邻域半径，单位米（默认 100000）")
    p.add_argument("--crs", dest="crs_out", default="EPSG:3857",
                   help="用于距离度量的投影（默认 EPSG:3857）")
    p.add_argument("--kernel", dest="kernel", default="combo", choices=["gaussian","circle","combo"],
                   help="空间核类型（默认 combo）")
    p.add_argument("--no-norm", dest="normalize", action="store_false",
                   help="关闭局部核权归一化（默认开启）")
    p.add_argument("--ymax", dest="ymax", type=float, default=1.0,
                   help="箱线图 y 轴上限（默认 1.0；设为 -1 表示不限制）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        csv_path="../data/csv/attr_density_with_target.csv",
        lon_col="gauge_lon",
        lat_col="gauge_lat",
        radius_m=args.radius_m,
        crs_out=args.crs_out,
        kernel=args.kernel,
        normalize=args.normalize,
        filter_query=args.filter_query,
        weight_expr=args.weight_expr,
        weight_col=args.weight_col,
        ymax=(None if (args.ymax is not None and args.ymax < 0) else args.ymax),
    )
