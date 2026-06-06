#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from pyproj import Transformer
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==== 全局风格 ====
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "axes.unicode_minus": False,
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "legend.fontsize": 18, "legend.title_fontsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18,
})

# ---------- I/O ----------
def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in (None, "utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def apply_filter(df: pd.DataFrame, query: Optional[str]) -> pd.DataFrame:
    if not query: return df
    try:
        return df.query(query, engine="python")
    except Exception as e:
        raise ValueError(f"--filter 解析失败：{e}")

def make_weights(df: pd.DataFrame, weight_expr: Optional[str]) -> Optional[np.ndarray]:
    if not weight_expr: return None
    try:
        s = df.eval(weight_expr, engine="python",
                    local_dict={"log1p": np.log1p, "log": np.log, "log10": np.log10,
                                "abs": np.abs, "clip": np.clip,
                                "minimum": np.minimum, "maximum": np.maximum})
    except Exception as e:
        raise ValueError(f"--weight-expr 解析失败：{e}")
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    mn, mx = float(np.min(s)), float(np.max(s))
    return (s - mn) / max(mx - mn, 1e-12) if mx > mn else np.zeros_like(s)

def project_lonlat(df: pd.DataFrame, lon: str, lat: str, crs_out: str) -> np.ndarray:
    t = Transformer.from_crs("EPSG:4326", crs_out, always_xy=True)
    x, y = t.transform(df[lon].to_numpy(), df[lat].to_numpy())
    return np.vstack([x, y]).T

# ---------- 点上距离核密度（排除自身） ----------
def pointwise_distance_kde(
    xy: np.ndarray,
    radius_m: float,
    kernel: str = "exponential",        # exponential/gaussian/linear/inverse/uniform
    scale_m: Optional[float] = None,    # 高斯σ / 指数λ；None→R/3
    inv_power: float = 1.0,
    weights: Optional[np.ndarray] = None,
    normalize: bool = False,            # True=局部均值；False=密度（权重和）
    eps: float = 1.0
) -> np.ndarray:
    n = xy.shape[0]
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    bw = max((radius_m/3.0 if scale_m is None else float(scale_m)), eps)
    tree = cKDTree(xy, leafsize=40)
    dens = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float) if normalize else None
    for i in range(n):
        nbrs = tree.query_ball_point(xy[i], r=radius_m)
        nbrs = [j for j in nbrs if j != i]  # 排除自身
        if not nbrs: continue
        nbrs = np.asarray(nbrs, dtype=int)
        d = np.linalg.norm(xy[nbrs] - xy[i], axis=1)
        k = kernel.lower()
        if k == "uniform":
            K = (d <= radius_m).astype(float)
        elif k == "gaussian":
            K = np.exp(-0.5*(d/bw)**2); K[d > radius_m] = 0.0
        elif k == "exponential":
            K = np.exp(-d/bw);         K[d > radius_m] = 0.0
        elif k == "linear":
            K = np.maximum(0.0, 1.0 - d/radius_m)
        elif k == "inverse":
            K = 1.0/np.power(d + eps, inv_power); K[d > radius_m] = 0.0
        else:
            raise ValueError("kernel 必须为 exponential/gaussian/linear/inverse/uniform")
        dens[i] = np.sum(w[nbrs] * K)
        if normalize: denom[i] = np.sum(K)
    if normalize:
        dens = np.divide(dens, np.maximum(denom, 1e-12))
    # 缩放到 [0,1]，便于可视化
    mn, mx = float(np.nanmin(dens)), float(np.nanmax(dens))
    return (dens - mn)/max(mx - mn, 1e-12) if mx > mn else np.zeros_like(dens)

# ---------- 经纬度坐标轴格式 ----------
def format_lon_label(lon: float) -> str:
    hemi = "E" if lon >= 0 else "W"
    return f"{abs(lon):.0f}°{hemi}"
def format_lat_label(lat: float) -> str:
    hemi = "N" if lat >= 0 else "S"
    return f"{abs(lat):.0f}°{hemi}"
def set_lonlat_axis_labels(ax, crs_proj: str):
    to_geo = Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)
    def x_fmt(x, _=None):
        ymid = np.mean(ax.get_ylim())
        lon, _ = to_geo.transform(x, ymid)
        return format_lon_label(lon)
    def y_fmt(y, _=None):
        xmid = np.mean(ax.get_xlim())
        _, lat = to_geo.transform(xmid, y)
        return format_lat_label(lat)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_fmt))
    ax.set_xlabel("Longitude (°)"); ax.set_ylabel("Latitude (°)")

# ---------- 点热力图（散点着色） ----------
def plot_point_heatmap(
    df: pd.DataFrame,
    lon: str, lat: str, crs_out: str, value_col: str,
    usa_shp: Optional[str] = None,
    title: str = "Pointwise Density (0–1)",
    out_file: Optional[str] = None, # <--- 变量名修改为 out_file
    s: float = 1000, cmap: str = "inferno"
):
    # 投影到绘图坐标
    xy = project_lonlat(df, lon, lat, crs_out)
    x, y = xy[:,0], xy[:,1]

    # 叠加边界
    # ... (此处省略边界处理代码，保持不变) ...
    gdf_usa = None
    if usa_shp and os.path.exists(usa_shp):
        gdf_usa = gpd.read_file(usa_shp)
        if gdf_usa.crs is None:
            raise ValueError("USA 边界缺少 CRS")
        gdf_usa = gdf_usa.to_crs(crs_out)
        # 去掉 AK/HI/PR 等
        drop_codes = {"AK","HI","PR","GU","VI","AS","MP"}
        if "STUSPS" in gdf_usa.columns:
            gdf_usa = gdf_usa[~gdf_usa["STUSPS"].isin(drop_codes)].reset_index(drop=True)
        elif "NAME" in gdf_usa.columns:
            drop_names = {"Alaska","Hawaii","Puerto Rico","Guam","American Samoa",
                          "Commonwealth of the Northern Mariana Islands","Northern Mariana Islands",
                          "Virgin Islands of the United States","Virgin Islands"}
            gdf_usa = gdf_usa[~gdf_usa["NAME"].isin(drop_names)].reset_index(drop=True)


    # 画图
    # **修改：将 figsize 更改为 (10, 10) **
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    sc = ax.scatter(x, y, c=df[value_col].to_numpy(),
                    s=s, cmap="coolwarm",
                    edgecolors="black", linewidths=0, alpha=0.9)

    if gdf_usa is not None and not gdf_usa.empty:
        gdf_usa.boundary.plot(ax=ax, linewidth=0.2, color="k")
        # 适度裁剪视图到边界范围
        minx, miny, maxx, maxy = gdf_usa.total_bounds
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    set_lonlat_axis_labels(ax, crs_out)
    cbar = plt.colorbar(sc, ax=ax,shrink=0.6,pad=0.02)
    cbar.set_label("Pointwise Density (normalized 0–1)")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.25)
    plt.tight_layout()
    # **修改：保存文件到 out_file**
    if out_file:
        plt.savefig(out_file)
        print(f"[OK] 已导出热力图：{out_file}")
    if out_file or not plt.isinteractive():
        plt.show()

# ---------- 主流程 ----------
def run(csv_path: str, lon_col: str, lat_col: str,
        radius_m: float, crs_out: str,
        kernel: str, scale_m: Optional[float], inv_power: float,
        normalize: bool, filter_query: Optional[str],
        weight_expr: Optional[str],
        no_show: bool, out_pdf: Optional[str]): # <--- 变量名修改为 out_pdf

    df_all = load_table(csv_path)
    for c in (lon_col, lat_col):
        if c not in df_all.columns:
            raise ValueError(f"找不到列：{c}")
    df = apply_filter(df_all, filter_query).dropna(subset=[lon_col, lat_col]).copy()
    if df.empty:
        raise RuntimeError("经纬度为空，无法计算。")

    # 权重（可选）
    w = make_weights(df, weight_expr)

    # 点上密度（不做栅格/不画箱线图）
    xy = project_lonlat(df, lon_col, lat_col, crs_out)
    dens = pointwise_distance_kde(
        xy=xy, radius_m=radius_m, kernel=kernel,
        scale_m=scale_m, inv_power=inv_power,
        weights=w, normalize=normalize, eps=1.0
    )
    df["kde_value"] = dens

    # 保存点 CSV
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_csv = os.path.join(out_dir, f"{base}_geo_points_with_kde.csv")
    keep = [c for c in [
        ("gauge_id" if "gauge_id" in df.columns else None),
        ("site_id"  if "site_id"  in df.columns else None),
        ("id"       if "id"       in df.columns else None),
        lon_col, lat_col, "kde_value"
    ] if c]
    df.loc[:, keep].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 点密度已保存到：{out_csv}")

    # 画“点热力图”
    if not no_show or out_pdf: # 使用 out_pdf
        usa_shp = r"../data/shp/cb_2018_us_state_20m.shp"
        title = f""
        plot_point_heatmap(
            df, lon=lon_col, lat=lat_col, crs_out=crs_out, value_col="kde_value",
            usa_shp=usa_shp, title=title, out_file=out_pdf, s=24, cmap="coolwarm" # <--- 传递 out_pdf
        )

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="点上距离核密度 → 直接绘制“点热力图”（不栅格、不箱线图）")
    p.add_argument("--csv", default=r"../data/csv/attr_density_with_target.csv")
    p.add_argument("--lon-col", default="gauge_lon")
    p.add_argument("--lat-col", default="gauge_lat")
    p.add_argument("--crs-out", default="EPSG:3857", help="距离度量投影（默认 EPSG:3857）")

    p.add_argument("--kernel", default="exponential",
                   choices=["exponential","gaussian","linear","inverse","uniform"],
                   help="距离核类型")
    p.add_argument("--radius-m", type=float, default=100000, help="半径（米）")
    p.add_argument("--scale-m", type=float, default=None, help="gaussian σ / exponential λ（默认 R/3）")
    p.add_argument("--inv-power", type=float, default=1.0, help="inverse 核的幂")
    p.add_argument("--normalize", action="store_true", help="返回局部平均(除以核权和)；默认关闭=密度")

    p.add_argument("--filter", default=None, help="pandas.query 条件")
    p.add_argument("--weight-expr", default=None, help="表达式作为邻点权重（默认等权）")

    p.add_argument("--no-show", action="store_true", help="不弹出预览窗口")
    # **修改：更改参数名、默认值和帮助信息**
    p.add_argument("--out-pdf", default=r"point_density_heatmap.pdf", help="保存 PDF 路径（可为空不保存）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # **修改：调用 run 时使用 out_pdf 变量**
    run(csv_path=args.csv,
        lon_col=args.lon_col, lat_col=args.lat_col,
        radius_m=args.radius_m, crs_out=args.crs_out,
        kernel=args.kernel, scale_m=args.scale_m, inv_power=args.inv_power,
        normalize=args.normalize, filter_query=args.filter,
        weight_expr=args.weight_expr,
        no_show=args.no_show, out_pdf=(None if not args.out_pdf else args.out_pdf))
