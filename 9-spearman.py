import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rcParams
from scipy.stats import spearmanr
import numpy as np

# ===== 全局字体设为 Times New Roman =====
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "axes.unicode_minus": False,   # 避免负号显示为方块
})
sns.set_theme(style="white", rc={
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
})

# 读取数据
df = pd.read_csv(r"../data/csv/Merged_with_Attributes.csv")
save_path = r'figure9.png'

# 确保路径存在
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 静态属性作为横坐标
attributes = ['area', 'aridity_FAO_PM', 'frac_snow', 'kar_pc_sse', 'slp_dg_sav', 'for_pc_sse']

# 方法列作为纵坐标
methods = {
    'Geo Method': 'max_geo_N',
    'Similarity Method': 'max_similarity_N'
}

# 初始化表
correlation_df = pd.DataFrame(index=methods.keys(), columns=attributes, dtype=float)
annot_matrix = pd.DataFrame(index=methods.keys(), columns=attributes, dtype=object)

# 打印显著性等级标题
print("=== 显著性等级 (p-value, Spearman) ===")
print("  *     : p < 0.1")
print("  **    : p < 0.05")
print("  ***   : p < 0.01\n")

# 计算 Spearman 相关与显著性（逐对去除 NaN）
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

        # 显著性标注
        if pd.notna(pval) and pval < 0.01:
            stars = '***'
        elif pd.notna(pval) and pval < 0.05:
            stars = '**'
        elif pd.notna(pval) and pval < 0.1:
            stars = '*'
        else:
            stars = ''

        annot_matrix.loc[method_label, attr] = (f"{corr:.2f}{stars}" if pd.notna(corr) else np.nan)

        # 打印信息
        print(f"{method_label:>18} vs {attr:<20}: ρ = {corr:.3f}, p = {pval:.4f} {stars}")

# 绘图
# 绘图
plt.figure(figsize=(12, 6))  # 更大的图像尺寸，确保保存时显示全屏

# 设置色带中值为 0
vmin = float(np.nanmin(correlation_df.values))
vmax = float(np.nanmax(correlation_df.values))
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# 绘制热力图
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

# 标题与坐标轴（Times New Roman）
plt.title("Spearman Correlation between Static Attributes and Optimal N",
          fontsize=22, weight='bold', loc='center', fontname='Times New Roman')
plt.xlabel("", fontname='Times New Roman')
plt.ylabel("", fontname='Times New Roman')

# 坐标刻度字体
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=0, ha='center', fontsize=22, weight='bold', fontname='Times New Roman'
)
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0, fontsize=22, weight='bold', fontname='Times New Roman'
)

# 统一 colorbar 字体
cbar = heatmap.collections[0].colorbar
cbar.set_label('Spearman Correlation', fontsize=22, weight='bold', fontname='Times New Roman')
for t in cbar.ax.get_yticklabels():
    t.set_fontname('Times New Roman')
    t.set_fontsize(22)

# 保存图像
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

