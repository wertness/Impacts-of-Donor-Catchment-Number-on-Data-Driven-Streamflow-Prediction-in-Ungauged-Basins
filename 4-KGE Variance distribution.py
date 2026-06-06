import ast
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import TwoSlopeNorm
import os

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 18,
    "axes.unicode_minus": False,
})


# =============================
# 数据读取与预处理
# =============================
def read_kge_data(files):
    frames = []
    for f in files:
        fpath = Path(f)
        print(f"Checking file: {fpath}, Exists: {os.path.exists(fpath)}")
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} does not exist. Skipping.")
            continue
        df = pd.read_csv(fpath)
        print(f"Read {len(df)} rows from {fpath}")
        df["model"] = fpath.stem
        frames.append(df)

    if not frames:
        raise ValueError("未读取到任何 KGE 数据文件。")

    return pd.concat(frames, ignore_index=True)


def preprocess_kge_data(df, kge_col="KGE"):
    def _unbox(x):
        if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
            try:
                v = ast.literal_eval(x)
                return v[0] if isinstance(v, (list, tuple)) and v else np.nan
            except Exception:
                return np.nan
        return x

    out = df.copy()
    out[kge_col] = out[kge_col].apply(_unbox)
    out[kge_col] = pd.to_numeric(out[kge_col], errors="coerce").fillna(0.0)
    out[kge_col] = out[kge_col].clip(lower=0.0)
    return out


def merge_kge_and_station_data(kge_data, station_info, on="gauge_id"):
    return pd.merge(kge_data, station_info, on=on, how="inner")


# =============================
# 标准差（仅正值）计算
# =============================
def std_nonzero(values):
    """仅使用 >0 的值计算无偏样本标准差；数量<=1 返回0"""
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    n = arr.size
    return float(arr.std(ddof=1)) if n > 1 else 0.0


# =============================
# 绘图函数
# =============================
def _ensure_wgs84(gdf):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _add_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, xlocs=range(-125, -66, 10), ylocs=range(24, 50, 5),
                      linestyle="--", linewidth=0.5, alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 18}
    gl.ylabel_style = {"size": 18}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def plot_kge_distribution_on_map(merged_data, boundary_file, title, save_path):
    us_boundary = _ensure_wgs84(gpd.read_file(boundary_file))
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj})
    ax.set_title(title)
    us_boundary.plot(ax=ax, edgecolor="black", facecolor="lightgray", linewidth=0.5)
    ax.set_extent([-125.0, -66.9, 24.3, 49.3], crs=proj)
    _add_gridlines(ax)

    vals = np.asarray(merged_data["KGE_std"].values, dtype=float)
    logvals = np.log1p(np.log1p(np.clip(vals, 0, None)))
    vmax = np.nanmax(logvals) if np.isfinite(logvals).any() else 1.0

    sizes = np.clip(logvals * 60, 20, 300)
    sc = ax.scatter(merged_data["gauge_lon"], merged_data["gauge_lat"],
                    c=logvals, cmap="viridis", s=60,
                    edgecolors="black", linewidths=0.2,
                    vmin=0.0, vmax=vmax, transform=proj, zorder=3)

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("KGE STD(log)")

    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    # 保存绘制的图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_std_diff_map(diff_df, boundary_file, title, save_path):
    us_boundary = _ensure_wgs84(gpd.read_file(boundary_file))
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": proj})
    ax.set_title(title)
    us_boundary.plot(ax=ax, edgecolor="black", facecolor="lightgray", linewidth=0.5)
    ax.set_extent([-125.0, -66.9, 24.3, 49.3], crs=proj)
    _add_gridlines(ax)

    vals = np.asarray(diff_df["KGE_std_diff"].values, dtype=float)
    abs95 = np.nanpercentile(np.abs(vals), 95)
    vmax = float(abs95 if np.isfinite(abs95) and abs95 > 0 else np.nanmax(np.abs(vals)) or 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    sc = ax.scatter(diff_df["gauge_lon"], diff_df["gauge_lat"], c=vals,
                    norm=norm, cmap="coolwarm", s=60,
                    edgecolors="black", linewidths=0.2,
                    transform=proj, zorder=3)

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cbar.set_label("KGE Standard Deviation Difference (geo - similarity)")

    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    plt.tight_layout()

    # 保存绘制的图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================
# 主流程
# =============================
if __name__ == "__main__":
    geo_files = [
        r"../data/csv/KGE_results_filtered_geo3.csv",
        r"../data/csv/KGE_results_filtered_geo5.csv",
        r"../data/csv/KGE_results_filtered_geo7.csv",
        r"../data/csv/KGE_results_filtered_geo10_converted.csv",
        r"../data/csv/KGE_results_filtered_geo15.csv",
        r"../data/csv/KGE_results_filtered_geo20_converted.csv",
        r"../data/csv/KGE_results_filtered_geo25.csv",
        r"../data/csv/KGE_results_filtered_geo30.csv",
        r"../data/csv/KGE_results_filtered_geo40.csv",
        r"../data/csv/KGE_results_filtered_geo50_converted.csv",
        r"../data/csv/KGE_results_filtered_geo75.csv",
        r"../data/csv/KGE_results_filtered_geo100_converted.csv",
        r"../data/csv/KGE_results_filtered_geo150.csv",
        r"../data/csv/KGE_results_filtered_geo200_converted.csv",
        r"../data/csv/KGE_results_filtered_geo250.csv",
        r"../data/csv/KGE_results_filtered_geo300.csv",
        r"../data/csv/KGE_results_filtered_geo350.csv",
        r"../data/csv/KGE_results_filtered_geo400.csv",
        r"../data/csv/KGE_results_filtered_geo450.csv",
        r"../data/csv/KGE_results_filtered_geo500.csv",
        r"../data/csv/KGE_results_filtered_geo550.csv",
        r"../data/csv/KGE_results_filtered_geo600.csv",
    ]

    similarity_files = [
        r"../data/csv/KGE_results_filtered_similarity3.csv",
        r"../data/csv/KGE_results_filtered_similarity5.csv",
        r"../data/csv/KGE_results_filtered_similarity7.csv",
        r"../data/csv/KGE_results_filtered_similarity10_converted.csv",
        r"../data/csv/KGE_results_filtered_similarity15.csv",
        r"../data/csv/KGE_results_filtered_similarity20_converted.csv",
        r"../data/csv/KGE_results_filtered_similarity25.csv",
        r"../data/csv/KGE_results_filtered_similarity30.csv",
        r"../data/csv/KGE_results_filtered_similarity40.csv",
        r"../data/csv/KGE_results_filtered_similarity50_converted.csv",
        r"../data/csv/KGE_results_filtered_similarity75.csv",
        r"../data/csv/KGE_results_filtered_similarity100_converted.csv",
        r"../data/csv/KGE_results_filtered_similarity150.csv",
        r"../data/csv/KGE_results_filtered_similarity200_converted.csv",
        r"../data/csv/KGE_results_filtered_similarity250.csv",
        r"../data/csv/KGE_results_filtered_similarity300.csv",
        r"../data/csv/KGE_results_filtered_similarity350.csv",
        r"../data/csv/KGE_results_filtered_similarity400.csv",
        r"../data/csv/KGE_results_filtered_similarity450.csv",
        r"../data/csv/KGE_results_filtered_similarity500.csv",
        r"../data/csv/KGE_results_filtered_similarity550.csv",
        r"../data/csv/KGE_results_filtered_similarity600.csv",
    ]

    station_info_file = r"../data/csv/filtered_camels_10day_attributes_cleaned_update.csv"
    boundary_file = r"../data/shp/cb_2018_us_state_20m.shp"

    # 确保目标目录存在
    save_directory = '.'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 读取和预处理数据
    all_geo_data = preprocess_kge_data(read_kge_data(geo_files))
    all_similarity_data = preprocess_kge_data(read_kge_data(similarity_files))
    station_info = pd.read_csv(station_info_file)

    # 合并数据
    geo_merged = merge_kge_and_station_data(all_geo_data, station_info)
    sim_merged = merge_kge_and_station_data(all_similarity_data, station_info)

    # 计算标准差
    geo_std = geo_merged.groupby("gauge_id")["KGE"].apply(std_nonzero).rename("KGE_std").reset_index()
    sim_std = sim_merged.groupby("gauge_id")["KGE"].apply(std_nonzero).rename("KGE_std").reset_index()

    geo_with_std = pd.merge(geo_std, station_info[["gauge_id", "gauge_lon", "gauge_lat"]], on="gauge_id", how="inner")
    sim_with_std = pd.merge(sim_std, station_info[["gauge_id", "gauge_lon", "gauge_lat"]], on="gauge_id", how="inner")

    # 绘制图像并保存
    plot_kge_distribution_on_map(geo_with_std, boundary_file, "",
                                 os.path.join(save_directory, "geo_std_distribution.png"))
    plot_kge_distribution_on_map(sim_with_std, boundary_file, "",
                                 os.path.join(save_directory, "similarity_std_distribution.png"))

    # 差值计算
    std_compare = pd.merge(
        geo_std.rename(columns={"KGE_std": "KGE_std_geo"}),
        sim_std.rename(columns={"KGE_std": "KGE_std_sim"}),
        on="gauge_id", how="inner"
    )
    std_compare["KGE_std_diff"] = std_compare["KGE_std_geo"] - std_compare["KGE_std_sim"]
    diff_with_coord = pd.merge(std_compare, station_info[["gauge_id", "gauge_lon", "gauge_lat"]], on="gauge_id",
                               how="inner")

    # 绘制差值图并保存
    plot_std_diff_map(diff_with_coord, boundary_file,
                      "KGE Standard Deviation Difference: Geo − Similarity (centered at 0)",
                      os.path.join(save_directory, "std_diff_map.png"))

    print("✅ Done.")
