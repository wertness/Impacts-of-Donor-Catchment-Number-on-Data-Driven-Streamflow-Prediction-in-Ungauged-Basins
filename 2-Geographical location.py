# coding=utf-8
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredText
import numpy as np

# 设置全局字体大小为12pt
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'  # 可以选择其他字体如 'Arial', 'Times New Roman'

# 1. 加载美国边界数据 (Shapefile)
gdf = gpd.read_file(r"../data/shp/cb_2018_us_state_20m.shp")

# 2. 加载站点数据 (CSV 文件)
df = pd.read_csv(r"../data/csv/filtered_camels_10day_attributes_cleaned_learn.csv")

# 检查是否包含 'gauge_lon' 和 'gauge_lat'
if 'gauge_lon' not in df.columns or 'gauge_lat' not in df.columns:
    raise ValueError("数据缺少 'gauge_lon' 或 'gauge_lat' 列，请检查数据格式。")

# 3. 创建投影：经纬度投影 (PlateCarree)
proj = ccrs.PlateCarree()

# 4. 创建画布
fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=proj)

# 5. 设置背景色为淡灰色
ax.set_facecolor('lightgray')

# 6. 添加地图要素：州界、国家边界
ax.add_feature(cfeature.STATES, edgecolor='white', linewidth=0.7)  # 州界，白色边界
ax.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=1.0)  # 国家边界，白色边界
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.0)  # 海岸线

# 7. 绘制美国边界数据
gdf.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1.5)

# 8. 绘制站点经纬度
scatter = ax.scatter(
    df["gauge_lon"],  # 经度
    df["gauge_lat"],  # 纬度
    s=35,  # 点的大小
    color="red",  # 点的颜色
    alpha=0.8,  # 透明度
    transform=ccrs.PlateCarree(),  # 经纬度坐标系
    edgecolors='darkred',  # 点边界颜色
    linewidth=0.5,  # 点边界宽度
    label="CAMELS Stations"
)

# 9. 设置显示范围
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())  # 美国本土区域

# 10. 绘制比例尺
scale_lon = -123
scale_lat = 27
scale_length_km = 500
km_per_degree = 95
scale_length_deg = scale_length_km / km_per_degree

# 绘制比例尺主线
ax.plot(
    [scale_lon, scale_lon + scale_length_deg],
    [scale_lat, scale_lat],
    transform=ccrs.PlateCarree(),
    color='black',
    linewidth=3
)

# 绘制比例尺两端垂直线
ax.plot(
    [scale_lon, scale_lon],
    [scale_lat - 0.2, scale_lat + 0.2],
    transform=ccrs.PlateCarree(),
    color='black',
    linewidth=2
)
ax.plot(
    [scale_lon + scale_length_deg, scale_lon + scale_length_deg],
    [scale_lat - 0.2, scale_lat + 0.2],
    transform=ccrs.PlateCarree(),
    color='black',
    linewidth=2
)

# 11. 绘制比例尺文本
ax.text(
    scale_lon + scale_length_deg / 2,
    scale_lat - 1,
    f"{scale_length_km} km",
    horizontalalignment='center',
    verticalalignment='center',
    transform=ccrs.PlateCarree(),
    fontsize=18,
    fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
)


# 12. 添加指北针
def add_north_arrow(ax, x, y, size=30):
    """
    在指定位置添加指北针
    """
    ax.text(x, y, 'N', transform=ccrs.PlateCarree(),
            fontsize=16, fontweight='bold',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', edgecolor='black'))

    # 添加箭头
    ax.annotate('', xy=(x, y + 0.5), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                transform=ccrs.PlateCarree())


# 在右上角添加指北针
add_north_arrow(ax, -68, 48, size=30)

# 13. 创建更美观的图例 - 移动到右下角
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor='red', markersize=8,
               markeredgecolor='darkred', markeredgewidth=0.5,
               label='CAMELS Stations'),
    plt.Line2D([0], [0], color='white', lw=1.5,
               label='State Boundaries'),
    plt.Line2D([0], [0], color='black', lw=3,
               label=f'Scale ({scale_length_km} km)')
]

# 添加图例到右下角
legend = ax.legend(handles=legend_elements,
                   loc='lower right',  # 修改为右下角
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   facecolor='white',
                   edgecolor='black',
                   fontsize=18)

# 14. 添加经纬度边框
try:
    # 设置经纬度网格线
    gl = ax.gridlines(draw_labels=True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 设置标签位置和样式
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    print("经纬度边框和刻度添加成功！")
except Exception as e:
    print(f"添加经纬度边框时出错: {e}")
    raise

# 15. 添加标题（如果不需要可以注释掉）
# plt.title("Locations of watersheds in the CAMELS dataset",
#          fontsize=14, fontweight='bold', pad=20)

# 16. 添加数据来源说明（可选）- 移动到左下角
ax.text(0.02, 0.02, 'Data Source: CAMELS Dataset',
        transform=ax.transAxes, fontsize=18,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 17. 调整布局并显示
plt.tight_layout()
plt.show()

