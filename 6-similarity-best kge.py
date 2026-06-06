import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# 读取数据
merged_data = pd.read_csv(r"../data/csv/merged_KGE_results_with_max_and_attributes.csv")

# 提取所有 KGE_similarityN 列
kge_columns = [col for col in merged_data.columns if col.startswith('KGE_similarity')]

# 提取 N 值并排序
similarity_N = sorted([int(col.split('similarity')[-1]) for col in kge_columns])
kge_columns_sorted = [f'KGE_similarity{n}' for n in similarity_N]

# 使用 Viridis colormap 创建颜色映射
cmap = matplotlib.colormaps.get_cmap('viridis').resampled(len(kge_columns_sorted))
viridis_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(kge_columns_sorted))]

# 方法与颜色映射（保持与 kge_columns_sorted 对齐的顺序）
method_colors = {method: viridis_colors[i] for i, method in enumerate(kge_columns_sorted)}

# 计算每个测站最大 KGE 对应的方法
merged_data['max_KGE_method'] = merged_data[kge_columns].idxmax(axis=1)

# 映射颜色
merged_data['color'] = merged_data['max_KGE_method'].map(method_colors)

# 打印检查
print(merged_data[['gauge_id', 'max_KGE_method', 'color']].head())

# 读取美国边界 Shapefile
us_shapefile = r"../data/shp/cb_2018_us_state_20m.shp"
gdf_us = gpd.read_file(us_shapefile)

# 创建地图
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
gdf_us.plot(ax=ax, color='lightgray', edgecolor='black')
ax.set_extent([-125, -66.5, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# 坐标提取
x = merged_data['gauge_lon'].values
y = merged_data['gauge_lat'].values

# 绘制点
scatter = ax.scatter(
    x, y,
    c=merged_data['color'],
    s=70,                      # 点大小
    edgecolors='black',
    linewidth=1.2,
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

# 图例元素（点大小可调 markersize）
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color, markeredgecolor='black',
           markersize=9, linewidth=0, label=method)
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
    columnspacing=1.2,
    labelspacing=0.6,
    handlelength=1.1,
    handletextpad=0.4,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    prop={'weight': 'bold'}
)

# 经纬度网格
gridlines = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1.0, color='gray')
gridlines.xlocator = MultipleLocator(5)
gridlines.ylocator = MultipleLocator(5)
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER
gridlines.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.top_labels = False
gridlines.right_labels = False

# 去除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 标题与布局
plt.title("Map of Maximum KGE Method Distribution by Gauge ID (USA)", fontsize=16)
# 紧凑布局（在调整过 bottom 后再调用）
plt.tight_layout()

plt.show()
