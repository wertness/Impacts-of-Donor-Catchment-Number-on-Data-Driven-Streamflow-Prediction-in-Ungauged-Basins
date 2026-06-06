# coding=utf-8
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredText
import numpy as np

# Set the global font size to 12 pt
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'  # Other fonts such as 'Arial' or 'Times New Roman' can also be used

# 1. Load the U.S. boundary data (shapefile)
gdf = gpd.read_file(r"../data/shp/cb_2018_us_state_20m.shp")

# 2. Load the station data (CSV file)
df = pd.read_csv(r"../data/csv/filtered_camels_10day_attributes_cleaned_learn.csv")

# Check whether 'gauge_lon' and 'gauge_lat' are present
if 'gauge_lon' not in df.columns or 'gauge_lat' not in df.columns:
    raise ValueError("数据缺少 'gauge_lon' 或 'gauge_lat' 列，请检查数据格式。")

# 3. Create the geographic projection (PlateCarree)
proj = ccrs.PlateCarree()

# 4. Create the figure canvas
fig = plt.figure(figsize=(14, 10))
ax = plt.axes(projection=proj)

# 5. Set a light gray background color
ax.set_facecolor('lightgray')

# 6. Add map features: state and national boundaries
ax.add_feature(cfeature.STATES, edgecolor='white', linewidth=0.7)  # state boundaries with white edges
ax.add_feature(cfeature.BORDERS, edgecolor='white', linewidth=1.0)  # national boundaries with white edges
ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=1.0)  # coastline

# 7. Plot the U.S. boundary data
gdf.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=1.5)

# 8. Plot station locations
scatter = ax.scatter(
    df["gauge_lon"],  # longitude
    df["gauge_lat"],  # latitude
    s=35,  # marker size
    color="red",  # marker color
    alpha=0.8,  # transparency
    transform=ccrs.PlateCarree(),  # geographic coordinate system
    edgecolors='darkred',  # marker edge color
    linewidth=0.5,  # marker edge width
    label="CAMELS Stations"
)

# 9. Set the map extent
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())  # contiguous United States

# 10. Draw the scale bar
scale_lon = -123
scale_lat = 27
scale_length_km = 500
km_per_degree = 95
scale_length_deg = scale_length_km / km_per_degree

# Draw the main scale-bar line
ax.plot(
    [scale_lon, scale_lon + scale_length_deg],
    [scale_lat, scale_lat],
    transform=ccrs.PlateCarree(),
    color='black',
    linewidth=3
)

# Draw the vertical ticks at both ends of the scale bar
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

# 11. Add the scale-bar label
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


# 12. Add a north arrow
def add_north_arrow(ax, x, y, size=30):
    """
    在指定位置添加指北针
    """
    ax.text(x, y, 'N', transform=ccrs.PlateCarree(),
            fontsize=16, fontweight='bold',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', edgecolor='black'))

    # Add the arrow
    ax.annotate('', xy=(x, y + 0.5), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                transform=ccrs.PlateCarree())


# Add the north arrow in the upper-right corner
add_north_arrow(ax, -68, 48, size=30)

# 13. Create a cleaner legend and move it to the lower-right corner
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

# Add the legend to the lower-right corner
legend = ax.legend(handles=legend_elements,
                   loc='lower right',  # changed to the lower-right corner
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   facecolor='white',
                   edgecolor='black',
                   fontsize=18)

# 14. Add longitude and latitude gridlines
try:
    # Configure longitude and latitude gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Configure label positions and styles
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    print("经纬度边框和刻度添加成功！")
except Exception as e:
    print(f"添加经纬度边框时出错: {e}")
    raise

# 15. Add a title if needed
# plt.title("Locations of watersheds in the CAMELS dataset",
#          fontsize=14, fontweight='bold', pad=20)

# 16. Add an optional data-source note in the lower-left corner
ax.text(0.02, 0.02, 'Data Source: CAMELS Dataset',
        transform=ax.transAxes, fontsize=18,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 17. Adjust the layout and display the figure
plt.tight_layout()
plt.show()

