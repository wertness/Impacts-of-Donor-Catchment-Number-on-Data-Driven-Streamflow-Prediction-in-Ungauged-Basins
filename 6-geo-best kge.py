import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
import math
# 读取数据
merged_data = pd.read_csv(r"../data/csv/Merged_with_Attributes.csv")

# 提取所有KGE_geoN列
kge_columns = [col for col in merged_data.columns if col.startswith('KGE_geo')]
print("选中的方法列：", kge_columns)

# 提取N值并排序
method_N_values = [int(col.split('geo')[-1]) for col in kge_columns]
sorted_N_values = sorted(method_N_values)

# 使用 Viridis colormap 创建渐变颜色列表
cmap = matplotlib.colormaps.get_cmap('viridis').resampled(len(sorted_N_values))
viridis_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(sorted_N_values))]


# 为每个方法分配一个 viridis 颜色
method_colors = {
    f'KGE_geo{n}': viridis_colors[i]
    for i, n in enumerate(sorted_N_values)
}

# # 自定义颜色（例如，使用绿色到黄色的渐变色）
# custom_colors = [
#     "#311B92", "#1A237E", "#0D47A1", "#01579B", "#006064", "#00838F", "#0097A7", "#00ACC1",
#     "#00BCD4", "#26C6DA", "#4DD0E1", "#E0F7FA", "#E8F5E9", "#81C784", "#66BB6A", "#4CAF50",
#     "#43A047", "#507800", "#558B2F", "#1B5E20"
# ]




# # 为每个方法分配一个颜色
# method_colors = {
#     f'KGE_geo{n}': custom_colors[i % len(custom_colors)]  # 循环使用颜色
#     for i, n in enumerate(sorted_N_values)
# }

# 打印每个类别绘制前对应的颜色
print("\n方法与颜色对应关系：")
for method, color in method_colors.items():
    print(f"{method}: {color}")

# 找出每个测站最大KGE对应的方法
merged_data['max_KGE_method'] = merged_data[kge_columns].idxmax(axis=1)
merged_data['color'] = merged_data['max_KGE_method'].map(method_colors)

# 读取美国边界
us_shapefile = r"../data/shp/cb_2018_us_state_20m.shp"
gdf_us = gpd.read_file(us_shapefile)

# 创建地图
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
gdf_us.plot(ax=ax, color='lightgray', edgecolor='black')
ax.set_extent([-125, -66.5, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# 测站坐标
x = merged_data['gauge_lon'].values
y = merged_data['gauge_lat'].values

# 绘制测站点
scatter = ax.scatter(x, y, c=merged_data['color'], s=70, edgecolors='black',
                     linewidth=1.5, alpha=0.7, transform=ccrs.PlateCarree())

# 全部图例项：每个方法一个
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color, markersize=12, label=method)
    for method, color in method_colors.items()
]

# ===== 根据方法数量自动确定 legend 布局与底部留白 =====
num_methods = len(method_colors)
rows = 4
ncol = math.ceil(num_methods / rows)

# 随行数自动下移图例，并增加底部边距，避免裁切
y_offset = -0.04      # 行越多，越往下
bottom_margin = 0.10 + 0.06 * (rows - 1)   # 为图例预留的底部空白

# 先设置底部边距，再放图例，再紧凑布局
plt.subplots_adjust(bottom=bottom_margin)

leg = ax.legend(
    handles=legend_elements,
    title="Method",
    loc='upper center',              # 以图例的顶部中心作为锚点
    bbox_to_anchor=(0.5, y_offset),  # 居中并放到图外底部
    ncol=ncol,                       # 自动列数
    columnspacing=5,
    labelspacing=0.6,
    handlelength=1.1,
    handletextpad=0.4,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    prop={'weight': 'bold'}
)


# 经纬度网格线设置
gridlines = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1.0, color='gray')
gridlines.xlocator = plt.MultipleLocator(5)
gridlines.ylocator = plt.MultipleLocator(5)
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER
gridlines.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.top_labels = False
gridlines.right_labels = False

# 移除坐标轴刻度数字
ax.set_xticks([])
ax.set_yticks([])

# 标题与布局优化
plt.title("Map of Maximum KGE Method Distribution by Gauge ID (USA)", fontsize=16)
plt.tight_layout()
plt.show()
