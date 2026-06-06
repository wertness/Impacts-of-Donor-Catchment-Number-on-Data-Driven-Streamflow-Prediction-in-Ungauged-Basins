import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
import math
# Read the data
merged_data = pd.read_csv(r"../data/csv/Merged_with_Attributes.csv")

# Extract all KGE_geoN columns
kge_columns = [col for col in merged_data.columns if col.startswith('KGE_geo')]
print("选中的方法列：", kge_columns)

# Extract the N values and sort them
method_N_values = [int(col.split('geo')[-1]) for col in kge_columns]
sorted_N_values = sorted(method_N_values)

# Create a gradient color list using the Viridis colormap
cmap = matplotlib.colormaps.get_cmap('viridis').resampled(len(sorted_N_values))
viridis_colors = [matplotlib.colors.to_hex(cmap(i)) for i in range(len(sorted_N_values))]


# Assign a Viridis color to each method
method_colors = {
    f'KGE_geo{n}': viridis_colors[i]
    for i, n in enumerate(sorted_N_values)
}

# Optional custom color palette (for example, a green-to-yellow gradient)
# custom_colors = [
#     "#311B92", "#1A237E", "#0D47A1", "#01579B", "#006064", "#00838F", "#0097A7", "#00ACC1",
#     "#00BCD4", "#26C6DA", "#4DD0E1", "#E0F7FA", "#E8F5E9", "#81C784", "#66BB6A", "#4CAF50",
#     "#43A047", "#507800", "#558B2F", "#1B5E20"
# ]




# Assign a custom color to each method
# method_colors = {
# reuse colors cyclically
#     for i, n in enumerate(sorted_N_values)
# }

# Print the color assigned to each category before plotting
print("\n方法与颜色对应关系：")
for method, color in method_colors.items():
    print(f"{method}: {color}")

# Find the method associated with the maximum KGE at each station
merged_data['max_KGE_method'] = merged_data[kge_columns].idxmax(axis=1)
merged_data['color'] = merged_data['max_KGE_method'].map(method_colors)

# Read the U.S. boundary
us_shapefile = r"../data/shp/cb_2018_us_state_20m.shp"
gdf_us = gpd.read_file(us_shapefile)

# Create the map
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
gdf_us.plot(ax=ax, color='lightgray', edgecolor='black')
ax.set_extent([-125, -66.5, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# Station coordinates
x = merged_data['gauge_lon'].values
y = merged_data['gauge_lat'].values

# Plot the station points
scatter = ax.scatter(x, y, c=merged_data['color'], s=70, edgecolors='black',
                     linewidth=1.5, alpha=0.7, transform=ccrs.PlateCarree())

# Build the full legend with one entry per method
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=color, markersize=12, label=method)
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
    columnspacing=5,
    labelspacing=0.6,
    handlelength=1.1,
    handletextpad=0.4,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    prop={'weight': 'bold'}
)


# Configure longitude and latitude gridlines
gridlines = ax.gridlines(draw_labels=True, linestyle='--', linewidth=1.0, color='gray')
gridlines.xlocator = plt.MultipleLocator(5)
gridlines.ylocator = plt.MultipleLocator(5)
gridlines.xformatter = LONGITUDE_FORMATTER
gridlines.yformatter = LATITUDE_FORMATTER
gridlines.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
gridlines.top_labels = False
gridlines.right_labels = False

# Remove the numeric axis tick labels
ax.set_xticks([])
ax.set_yticks([])

# Improve the title and overall layout
plt.title("Map of Maximum KGE Method Distribution by Gauge ID (USA)", fontsize=16)
plt.tight_layout()
plt.show()
