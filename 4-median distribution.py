import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import ast

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

# 读取KGE数据文件并合并
def read_kge_data(files):
    """
    读取多个KGE数据文件并将它们合并成一个DataFrame
    :param files: KGE数据文件路径列表
    :return: 合并后的DataFrame
    """
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        model_name = file.split("\\")[-1].split(".")[0]  # 获取模型名称
        df['model'] = model_name  # 添加模型名称列
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# 数据预处理：去掉列表结构的方括号，并转换为数值
def preprocess_kge_data(df):
    """
    处理KGE数据，去掉方括号并转换为数值，负值替换为0
    :param df: 原始数据DataFrame
    :return: 预处理后的DataFrame
    """
    df['KGE'] = df['KGE'].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith('[') else x
    )
    df['KGE'] = pd.to_numeric(df['KGE'], errors='coerce').fillna(0)
    df['KGE'] = df['KGE'].apply(lambda x: max(x, 0))  # 替换负KGE值为0
    return df

# 合并KGE数据与站点经纬度数据
def merge_kge_and_station_data(kge_data, station_info):
    """
    合并KGE数据和站点信息
    :param kge_data: KGE数据
    :param station_info: 站点信息数据
    :return: 合并后的数据
    """
    merged_data = pd.merge(kge_data, station_info, on='gauge_id', how='inner')
    return merged_data

# 读取边界文件并绘制地图（已改为使用 Cartopy Gridliner 的经纬度标签与字号）
def plot_kge_distribution_on_map(merged_data, boundary_file, output_file, title):
    """
    绘制KGE中位数的地理分布图
    :param merged_data: 合并后的数据，包含gauge_id, KGE, gauge_lat, gauge_lon
    :param boundary_file: 美国边界文件的路径（shp）
    :param output_file: 输出文件路径
    :param title: 图表标题
    """
    # 加载边界文件
    us_boundary = gpd.read_file(boundary_file)
    # 若无坐标系或坐标系非 WGS84，则设定/转换为 EPSG:4326（经纬度）
    if us_boundary.crs is None:
        us_boundary = us_boundary.set_crs(epsg=4326, allow_override=True)
    elif us_boundary.crs.to_epsg() != 4326:
        us_boundary = us_boundary.to_crs(epsg=4326)

    # 设置地图投影
    proj = ccrs.PlateCarree()

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})

    # 绘制美国地图边界
    ax.set_title(title, fontsize=18)
    us_boundary.plot(ax=ax, edgecolor='black', facecolor='lightgray', linewidth=0.5)

    # 仅显示美国范围（经纬度范围：-125, -66.93457, 24.396308, 49.384358）
    ax.set_extent([-125.0, -66.93457, 24.396308, 49.384358], crs=proj)

    # 绘制每个gauge_id的位置（此处保持与你原逻辑一致：按 KGE 着色）
    sc = ax.scatter(
        merged_data['gauge_lon'],
        merged_data['gauge_lat'],
        c=merged_data['KGE'],
        cmap='coolwarm',
        s=60,
        edgecolors='black',
        linewidths=0.2,
        transform=proj,
        zorder=3
    )

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('KGE Median Value', fontsize=18)

    # —— 关键改动：使用 Cartopy 的 Gridliner 绘制经纬度标签，并设置字号 ——
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=range(-125, -66, 10),
        ylocs=range(24, 50, 5),
        linestyle='--',
        linewidth=0.5,
        alpha=0.6
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 18}   # ← 经度标签字号
    gl.ylabel_style = {"size": 18}   # ← 纬度标签字号
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 轴标签（若不需要可去掉，不影响 Gridliner 的经纬度刻度文字）
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)

    # 保存与显示
    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300)
    plt.show()

# =============================
# 文件路径：所有KGE数据文件的路径和站点信息文件路径
# =============================
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

# 站点信息文件路径，包含'gauge_id', 'gauge_lat', 'gauge_lon'等列
station_info_file = r"../data/csv/filtered_camels_10day_attributes_cleaned_update.csv"

# 读取KGE数据并预处理
all_geo_data = read_kge_data(geo_files)
all_geo_data = preprocess_kge_data(all_geo_data)

all_similarity_data = read_kge_data(similarity_files)
all_similarity_data = preprocess_kge_data(all_similarity_data)

# 读取站点信息
station_info = pd.read_csv(station_info_file)

# 合并KGE数据和站点信息
geo_merged_data = merge_kge_and_station_data(all_geo_data, station_info)
similarity_merged_data = merge_kge_and_station_data(all_similarity_data, station_info)

# 计算每个gauge_id下的KGE中位数
geo_kge_median_data = geo_merged_data.groupby('gauge_id')['KGE'].median().reset_index()
geo_kge_median_data.columns = ['gauge_id', 'KGE_median']

similarity_kge_median_data = similarity_merged_data.groupby('gauge_id')['KGE'].median().reset_index()
similarity_kge_median_data.columns = ['gauge_id', 'KGE_median']

# 将中位数数据和站点信息合并（此处仍保留你原有逻辑）
geo_merged_data_with_median = pd.merge(geo_merged_data, geo_kge_median_data, on='gauge_id', how='inner')
similarity_merged_data_with_median = pd.merge(similarity_merged_data, similarity_kge_median_data, on='gauge_id', how='inner')

# 边界文件路径
boundary_file = r"../data/shp/cb_2018_us_state_20m.shp"

# 输出文件路径
geo_output_file = r"geo_kge_median_distribution.png"
similarity_output_file = r"similarity_kge_median_distribution.png"

# 绘制KGE中位数的地理分布图（Geo模型）
plot_kge_distribution_on_map(
    geo_merged_data_with_median,
    boundary_file,
    geo_output_file,
    ""
)

# 绘制KGE中位数的地理分布图（Similarity模型）
plot_kge_distribution_on_map(
    similarity_merged_data_with_median,
    boundary_file,
    similarity_output_file,
    ""
)
