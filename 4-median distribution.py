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

# Read and merge the KGE data files
def read_kge_data(files):

    all_data = []
    for file in files:
        df = pd.read_csv(file)
        model_name = file.split("\\")[-1].split(".")[0]  # Extract the model name
        df['model'] = model_name  # Add the model-name column
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Data preprocessing: strip list brackets and convert values to numeric
def preprocess_kge_data(df):

    df['KGE'] = df['KGE'].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith('[') else x
    )
    df['KGE'] = pd.to_numeric(df['KGE'], errors='coerce').fillna(0)
    df['KGE'] = df['KGE'].apply(lambda x: max(x, 0))  # Replace negative KGE values with 0
    return df

# Merge the KGE data with station coordinates
def merge_kge_and_station_data(kge_data, station_info):
    """
    合并KGE数据和站点信息
    :param kge_data: KGE数据
    :param station_info: 站点信息数据
    :return: 合并后的数据
    """
    merged_data = pd.merge(kge_data, station_info, on='gauge_id', how='inner')
    return merged_data

# Read the boundary file and plot the map using Cartopy gridline labels
def plot_kge_distribution_on_map(merged_data, boundary_file, output_file, title):

    # Load the boundary file
    us_boundary = gpd.read_file(boundary_file)
    # If the CRS is missing or not WGS84, assign/convert it to EPSG:4326
    if us_boundary.crs is None:
        us_boundary = us_boundary.set_crs(epsg=4326, allow_override=True)
    elif us_boundary.crs.to_epsg() != 4326:
        us_boundary = us_boundary.to_crs(epsg=4326)

    # Set the map projection
    proj = ccrs.PlateCarree()

    # Create the figure canvas
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj})

    # Plot the U.S. map boundary
    ax.set_title(title, fontsize=18)
    us_boundary.plot(ax=ax, edgecolor='black', facecolor='lightgray', linewidth=0.5)

    # Restrict the view to the contiguous United States extent
    ax.set_extent([-125.0, -66.93457, 24.396308, 49.384358], crs=proj)

    # Plot each gauge_id location and color it by KGE
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

    # Add the color bar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('KGE Median Value', fontsize=18)

    # Key change: use Cartopy Gridliner for longitude/latitude labels and font sizes
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
    gl.xlabel_style = {"size": 18}   # longitude-label font size
    gl.ylabel_style = {"size": 18}   # latitude-label font size
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Axis labels (optional; removing them will not affect gridline tick labels)
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)

    # Save and display the figure
    plt.tight_layout()
    plt.savefig(output_file, format='png', dpi=300)
    plt.show()

# =============================
# File paths for all KGE files and the station metadata file
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

# Station metadata file path, including columns such as 'gauge_id', 'gauge_lat', and 'gauge_lon'
station_info_file = r"../data/csv/filtered_camels_10day_attributes_cleaned_update.csv"

# Read and preprocess the KGE data
all_geo_data = read_kge_data(geo_files)
all_geo_data = preprocess_kge_data(all_geo_data)

all_similarity_data = read_kge_data(similarity_files)
all_similarity_data = preprocess_kge_data(all_similarity_data)

# Read the station metadata
station_info = pd.read_csv(station_info_file)

# Merge the KGE data and station metadata
geo_merged_data = merge_kge_and_station_data(all_geo_data, station_info)
similarity_merged_data = merge_kge_and_station_data(all_similarity_data, station_info)

# Compute the KGE median for each gauge_id
geo_kge_median_data = geo_merged_data.groupby('gauge_id')['KGE'].median().reset_index()
geo_kge_median_data.columns = ['gauge_id', 'KGE_median']

similarity_kge_median_data = similarity_merged_data.groupby('gauge_id')['KGE'].median().reset_index()
similarity_kge_median_data.columns = ['gauge_id', 'KGE_median']

# Merge the median values with the station metadata
geo_merged_data_with_median = pd.merge(geo_merged_data, geo_kge_median_data, on='gauge_id', how='inner')
similarity_merged_data_with_median = pd.merge(similarity_merged_data, similarity_kge_median_data, on='gauge_id', how='inner')

# Boundary file path
boundary_file = r"../data/shp/cb_2018_us_state_20m.shp"

# Output file paths
geo_output_file = r"geo_kge_median_distribution.png"
similarity_output_file = r"similarity_kge_median_distribution.png"

# Plot the geographic distribution of KGE median values for the Geo model
plot_kge_distribution_on_map(
    geo_merged_data_with_median,
    boundary_file,
    geo_output_file,
    ""
)

# Plot the geographic distribution of KGE median values for the Similarity model
plot_kge_distribution_on_map(
    similarity_merged_data_with_median,
    boundary_file,
    similarity_output_file,
    ""
)
