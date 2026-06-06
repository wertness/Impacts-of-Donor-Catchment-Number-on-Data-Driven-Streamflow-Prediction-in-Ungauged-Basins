import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
from scipy.stats import spearmanr
import numpy as np

# ===== Use Times New Roman as the global font =====
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "axes.unicode_minus": False,   # Prevent minus signs from being rendered as squares
})
sns.set_theme(style="white", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
})

# Read the data
df = pd.read_csv(r"../data/csv/Merged_with_Attributes.csv")
save_path = r'figure9.png'

# Make sure the output path exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Static attributes on the x-axis
attributes = ['area', 'aridity_FAO_PM', 'frac_snow', 'kar_pc_sse', 'slp_dg_sav', 'for_pc_sse']

# Method columns on the y-axis
methods = {
    'Geo Method': 'max_geo_N',
    'Similarity Method': 'max_similarity_N'
}

# Initialize the result tables
correlation_df = pd.DataFrame(index=methods.keys(), columns=attributes, dtype=float)
annot_matrix = pd.DataFrame(index=methods.keys(), columns=attributes, dtype=object)

# Print the significance-level legend
print("===  (p-value, Spearman) ===")
print("  *     : p < 0.1")
print("  **    : p < 0.05")
print("  ***   : p < 0.01\n")

# Compute Spearman correlations and significance after dropping NaN pairs
for method_label, method_col in methods.items():
    for attr in attributes:
        x = pd.to_numeric(df[attr], errors='coerce')
        y = pd.to_numeric(df[method_col], errors='coerce')
        mask = x.notna() & y.notna()
        if mask.sum() >= 3:
            corr, pval = spearmanr(x[mask], y[mask])
        else:
            corr, pval = np.nan, np.nan

        correlation_df.loc[method_label, attr] = corr

        # Significance annotation
        if pd.notna(pval) and pval < 0.01:
            stars = '***'
        elif pd.notna(pval) and pval < 0.05:
            stars = '**'
        elif pd.notna(pval) and pval < 0.1:
            stars = '*'
        else:
            stars = ''

        annot_matrix.loc[method_label, attr] = (f"{corr:.2f}{stars}" if pd.notna(corr) else np.nan)

        # Print the summary information
        print(f"{method_label:>18} vs {attr:<20}: ρ = {corr:.3f}, p = {pval:.4f} {stars}")

# Plot the figure
# Plot the figure
plt.figure(figsize=(12, 6))  # Use a larger figure size to avoid clipping when saving

# Set the color scale midpoint to 0
vmin = float(np.nanmin(correlation_df.values))
vmax = float(np.nanmax(correlation_df.values))
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Draw the heatmap
heatmap = sns.heatmap(
    correlation_df.astype(float),
    annot=annot_matrix,
    fmt='',
    cmap='coolwarm',
    norm=norm,
    linewidths=1,
    linecolor='white',
    cbar_kws={'label': 'Spearman Correlation'},
    square=False,
    annot_kws={"size": 22, "weight": "bold"}
)

# Title and axes (Times New Roman)
plt.title("Spearman Correlation between Static Attributes and Optimal N",
          fontsize=22, weight='bold', loc='center', fontname='Times New Roman')
plt.xlabel("", fontname='Times New Roman')
plt.ylabel("", fontname='Times New Roman')

# Tick-label font settings
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=0, ha='center', fontsize=22, weight='bold', fontname='Times New Roman'
)
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0, fontsize=22, weight='bold', fontname='Times New Roman'
)

# Use a consistent font for the color bar
cbar = heatmap.collections[0].colorbar
cbar.set_label('Spearman Correlation', fontsize=22, weight='bold', fontname='Times New Roman')
for t in cbar.ax.get_yticklabels():
    t.set_fontname('Times New Roman')
    t.set_fontsize(22)

# Save the figure
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

