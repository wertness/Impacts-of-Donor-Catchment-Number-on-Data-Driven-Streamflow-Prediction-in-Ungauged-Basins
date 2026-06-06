import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Read the data
merged_data = pd.read_csv(r"../data/csv/merged_KGE_results_with_max_and_attributes.csv")

# Extract all KGE_similarityN columns
kge_columns = [col for col in merged_data.columns if col.startswith('KGE_similarity')]

# Extract the N values and sort them
similarity_N = sorted([int(col.split('similarity')[-1]) for col in kge_columns])
kge_columns_sorted = [f'KGE_similarity{n}' for n in similarity_N]

# Create the color mapping with the Viridis colormap
cmap = matplotlib.colormaps.get_cmap('viridis').resampled(len(kge_columns_sorted))
viridis_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(kge_columns_sorted))]

# Map methods to colors in the same order as kge_columns_sorted
method_colors = {method: viridis_colors[i] for i, method in enumerate(kge_columns_sorted)}

# Determine the method associated with the maximum KGE at each station
merged_data['max_KGE_method'] = merged_data[kge_columns].idxmax(axis=1)

# Map the colors
merged_data['color'] = merged_data['max_KGE_method'].map(method_colors)

# Print a quick check of the results
print(merged_data[['gauge_id', 'max_KGE_method', 'color']].head())

# Read the U.S. boundary shapefile
us_shapefile = r"../data/shp/cb_2018_us_state_20m.shp"
gdf_us = gpd.read_file(us_shapefile)

# Create the map
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
gdf_us.plot(ax=ax, color='lightgray', edgecolor='black')
ax.set_extent([-125, -66.5, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# Extract the coordinates
x = merged_data['gauge_lon'].values
y = merged_data['gauge_lat'].values

# Plot the points
scatter = ax.scatter(
    x, y,
    c=merged_data['color'],
    s=70,                      # marker size
    edgecolors='black',
    linewidth=1.2,
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

# Legend elements (marker size can be adjusted with markersize)
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color, markeredgecolor='black',
           markersize=9, linewidth=0, label=method)
    for method, color in method_colors.items()
]

# Automatically adjust the legend layout and bottom margin based on the number of methods
num_methods = len(method_colors)
rows = 4
ncol = math.ceil(num_methods / rows)

# Shift the legend downward as rows increase and enlarge the bottom margin to avoid clipping
y_offset = -0.04      # Move the legend farther down as the number of rows increases
bottom_margin = 0.10 + 0.06 * (rows - 1)   # Bottom space reserved for the legend

# Set the bottom margin first, then place the legend, and finally tighten the layout
plt.subplots_adjust(bottom=bottom_margin)

leg = ax.legend(
    handles=legend_elements,
    title="Method",
    loc='upper center',              # Use the top center of the legend as the anchor point
    bbox_to_anchor=(0.5, y_offset),  # Center it and place it below the plot area
    ncol=ncol,                       # Automatically computed number of columns
    columnspacing=1.2,
    labelspacing=0.6,
    handlelength=1.1,
    handletextpad=0.4,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    prop={'weight': 'bold'}
)

# Longitude and latitude gridlines
gridlines = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1.0, color='gray')
gridlines.xlocator = MultipleLocator(5)
gridlines.ylocator = MultipleLocator(5)
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER
gridlines.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.top_labels = False
gridlines.right_labels = False

# Remove the axis tick labels
ax.set_xticks([])
ax.set_yticks([])

# Title and layout settings
plt.title("Map of Maximum KGE Method Distribution by Gauge ID (USA)", fontsize=16)
# Apply tight_layout() after adjusting the bottom margin
plt.tight_layout()

plt.show()
