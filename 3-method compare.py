# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D
from math import radians, sin, cos, asin, sqrt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import os
# ===============================================
# I. Configuration and constants
# ===============================================

# Global Matplotlib configuration
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 28
mpl.rcParams['pdf.fonttype'] = 42

# File paths (adjust them to your environment if needed)
USA_SHAPEFILE = r"../data/shp/cb_2018_us_state_20m.shp"
OUTPUT_PATH = r'figure3_refactored_final.pdf'

# Plot-style constants
COLOR_MAPS = {"Aridity": "sandybrown", "area": "dodgerblue", "fao_pm": "forestgreen"}
TARGET_COLOR = "blue"
DONOR_COLOR = "orange"
LINE_CLOSE = 4.0
LINE_FAR = 0.8
SIZE_SCALE = 2.0
N_DONORS = 8


# ===============================================
# II. Utility functions and classes
# ===============================================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        return np.min(zs)


def scale_01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)


def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


# ===============================================
# III. Data-generation functions
# ===============================================

def generate_data(n_donors, size_scale):
    np.random.seed(42)
    target_attr = np.array([0.5, 0.5, 0.5])
    donor_attr = np.random.rand(n_donors, 3)
    areas = np.random.randint(70, 220, size=n_donors)
    target_area = 220
    donor_sizes = areas * size_scale
    target_size = target_area * size_scale
    climates = np.random.choice(list(COLOR_MAPS.keys()), size=n_donors)
    attr_dist = np.linalg.norm(donor_attr - target_attr, axis=1)
    attr_sim = scale_01(1.0 - attr_dist)
    target_geo = np.array([-98.0, 39.0])
    donor_geo = np.column_stack([
        np.random.uniform(-105, -85, n_donors),
        np.random.uniform(30, 45, n_donors)
    ])
    geo_dist_km = np.array([
        haversine(target_geo[0], target_geo[1], donor_geo[i, 0], donor_geo[i, 1])
        for i in range(n_donors)
    ])
    geo_weight = scale_01(1.0 - geo_dist_km)

    return {
        'target_attr': target_attr, 'donor_attr': donor_attr, 'climates': climates,
        'target_geo': target_geo, 'donor_geo': donor_geo,
        'donor_sizes': donor_sizes, 'target_size': target_size,
        'geo_weight': geo_weight, 'attr_sim': attr_sim
    }


# ===============================================
# IV. Plotting helper functions
# ===============================================

def draw_cube_frame(ax, color='k', linestyle='-', linewidth=1.5, alpha=0.5):
    LOWER, UPPER = 0.0, 1.0
    verts = [
        (LOWER, LOWER, LOWER), (UPPER, LOWER, LOWER), (UPPER, UPPER, LOWER), (LOWER, UPPER, LOWER),
        (LOWER, LOWER, UPPER), (UPPER, LOWER, UPPER), (UPPER, UPPER, UPPER), (LOWER, UPPER, UPPER)
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    EXCLUDED_AXES = [(0, 1), (0, 3), (0, 4)]
    for edge in edges:
        if edge not in EXCLUDED_AXES:
            p1, p2 = verts[edge[0]], verts[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=0)


def draw_internal_grid(ax, ticks=np.arange(0.2, 1.0, 0.2), **kwargs):
    for val in ticks:
        ax.plot([val, val], [0, 1], [0, 0], zorder=0, **kwargs)
        ax.plot([0, 1], [val, val], [0, 0], zorder=0, **kwargs)
        ax.plot([val, val], [0, 0], [0, 1], zorder=0, **kwargs)
        ax.plot([0, 1], [0, 0], [val, val], zorder=0, **kwargs)
        ax.plot([0, 0], [val, val], [0, 1], zorder=0, **kwargs)
        ax.plot([0, 0], [0, 1], [val, val], zorder=0, **kwargs)


def setup_legend(target_size):
    return [
        plt.Line2D([0], [0], marker="o", color="w", label="Target",
                   markerfacecolor=TARGET_COLOR, markeredgecolor="k",
                   markersize=np.sqrt(target_size) * 0.7),
        plt.Line2D([0], [0], marker="o", color="w", label="Donor",
                   markerfacecolor=DONOR_COLOR, markeredgecolor="k", markersize=8),
        plt.Line2D([0], [0], color="gray", lw=LINE_CLOSE, label="Closer distance"),
        plt.Line2D([0], [0], color="gray", lw=LINE_FAR, label="Farther distance"),
    ]


# ===============================================
# V. Core plotting functions
# ===============================================

def plot_3d_axes(ax, data):
    

    # 1. Force equal scaling for the 3D axes
    ax.set_box_aspect((1, 1, 1))

    # 2. Set the axis limits
    ax.set_xlim(0, 1);
    ax.set_ylim(0, 1);
    ax.set_zlim(0, 1)

    # Change A: keep the axes visible instead of fully turning them off
    # Remove the default gray panes and grid while keeping the ticks

    # Remove the pane background color (make the panes transparent)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Remove the default grid and use a custom grid instead
    ax.grid(False)

    # Change B: set explicit tick locations and sizes
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Set the tick-label font size
    ax.tick_params(axis='both', which='major', labelsize=18, pad=2)

    # 3. Configure the viewing angle
    ax.dist = 8.0  # Pull the camera back slightly to avoid clipped tick labels
    ax.view_init(elev=30, azim=45)

    # 4. Draw custom internal gridlines and the cube frame
    grid_params = {'color': 'gray', 'linestyle': '-', 'linewidth': 0.8, 'alpha': 0.3}
    draw_internal_grid(ax, ticks=np.arange(0.2, 1.0, 0.2), **grid_params)

    # Draw thicker black axes for better visual emphasis
    ax.plot([0, 1], [0, 0], [0, 0], color='k', linewidth=2, zorder=1)
    ax.plot([0, 0], [0, 1], [0, 0], color='k', linewidth=2, zorder=1)
    ax.plot([0, 0], [0, 0], [0, 1], color='k', linewidth=2, zorder=1)
    draw_cube_frame(ax, color='k', linestyle='-', linewidth=1.5, alpha=0.5)

    # 5. Plot the points
    ax.scatter(*data['target_attr'], s=data['target_size'], c=TARGET_COLOR, edgecolor="k", zorder=5)
    for i in range(N_DONORS):
        donor_color = COLOR_MAPS[data['climates'][i]]
        ax.scatter(*data['donor_attr'][i], s=data['donor_sizes'][i], c=donor_color, edgecolor="k")
        ax.plot([data['target_attr'][0], data['donor_attr'][i, 0]],
                [data['target_attr'][1], data['donor_attr'][i, 1]],
                [data['target_attr'][2], data['donor_attr'][i, 2]],
                c="gray", alpha=0.7)

    # 6. Configure axis labels and increase label padding to avoid overlap
    ax.set_xlabel("Aridity", fontsize=20, labelpad=15)
    ax.set_ylabel("Area", fontsize=20, labelpad=15)
    ax.set_zlabel("fao_pm", fontsize=20, labelpad=15)

    # 7. Draw the arrows
    arrow_prop = dict(mutation_scale=15, arrowstyle='-|>', color='k', lw=1.5)
    ax.add_artist(Arrow3D([1, -0.05], [1, 1], [0, 0], **arrow_prop))
    ax.add_artist(Arrow3D([1, 1], [1, -0.05], [0, 0], **arrow_prop))
    ax.add_artist(Arrow3D([1, 1], [1, 1], [0, 1.05], **arrow_prop))

    # 8. Add the legend
    ax.legend(handles=setup_legend(data['target_size']), loc="upper left",
              bbox_to_anchor=(-0.1, 1.05), fontsize=16, frameon=True, title="Legend", title_fontsize=16)


def plot_map_axes(ax, data, usa_gdf):
    """绘制地理空间地图图"""
    # Keep the 2D panel in a square aspect ratio
    ax.set_box_aspect(1)

    usa_gdf.boundary.plot(ax=ax, edgecolor="black")
    ax.set_xlim(-125, -65)
    ax.set_ylim(25, 50)

    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.scatter(*data['target_geo'], s=data['target_size'], c=TARGET_COLOR, edgecolor="k", zorder=5)
    for i in range(N_DONORS):
        ax.scatter(*data['donor_geo'][i], s=data['donor_sizes'][i], c=DONOR_COLOR, edgecolor="k", zorder=4)
        lw = LINE_FAR + (LINE_CLOSE - LINE_FAR) * float(data['geo_weight'][i])
        ax.plot([data['target_geo'][0], data['donor_geo'][i, 0]],
                [data['target_geo'][1], data['donor_geo'][i, 1]],
                c="gray", alpha=0.7, linewidth=lw, zorder=3)

    ax.legend(handles=setup_legend(data['target_size']), loc="lower right",
              fontsize=16, frameon=True, title="Legend", title_fontsize=16)


# ===============================================
# VI. Main execution function
# ===============================================
def main():
    data = generate_data(N_DONORS, SIZE_SCALE)
    try:
        usa_gdf = gpd.read_file(USA_SHAPEFILE).to_crs(epsg=4326)
    except Exception as e:
        print(f"Error loading shapefile: {e}. Check path.")
        return

    # Create the figure
    fig = plt.figure(figsize=(16, 8))

    # Create the left panel (3D)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    plot_3d_axes(ax1, data)

    # Create the right panel (2D map)
    ax2 = fig.add_subplot(1, 2, 2)
    plot_map_axes(ax2, data, usa_gdf)

    plt.subplots_adjust(wspace=0.1, left=0.05, right=0.95, top=0.9, bottom=0.1)

    fig.text(0.25, 0.05, '(a)', ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.75, 0.05, '(b)', ha='center', va='center', fontsize=24, fontweight='bold')

    # ---------------------------------------------------------
    # Fix section
    # ---------------------------------------------------------

    # 1. Make sure the output directory exists; create it if needed
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Save the figure before calling show()
    # Use dpi=300 for clarity and bbox_inches='tight' to avoid clipped labels
    fig.savefig(OUTPUT_PATH, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Figure successfully saved to: {OUTPUT_PATH}")

    # 3. Display the figure after saving it
    plt.show()


if __name__ == "__main__":
    main()
