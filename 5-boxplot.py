import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import re

# Set the global Matplotlib font size to 18 and use Times New Roman
plt.rcParams.update({
    'font.size': 18,         # Set the global font size to 18
    'axes.labelsize': 18,    # Set the axis-label font size
    'xtick.labelsize': 18,   # Set the x-axis tick-label font size
    'ytick.labelsize': 18,   # Set the y-axis tick-label font size
    'legend.fontsize': 18,   # Set the legend font size
    'axes.titlesize': 18,    # Set the figure-title font size
    'font.family': 'Times New Roman'  # Use Times New Roman as the global font family
})

# The seaborn style also inherits the Matplotlib settings
sns.set_context("notebook", rc={"font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 18, "font.family": "Times New Roman"})

def plot_median_difference(df_long, save_path=None):
    """
    绘制中位数差值随 N 变化的折线图
    """
    # Compute the median for each (N, method) pair
    median_values = df_long.groupby(['N', 'method'])['KGE'].median().reset_index()

    # Convert the data to wide format to simplify the difference calculation
    df_median = median_values.pivot(index='N', columns='method', values='KGE')

    # Compute the difference in medians between the geo and similarity methods
    df_median['median_diff'] = df_median['geo'] - df_median['similarity']

    # Plot the median-difference line chart
    plt.figure(figsize=(14, 4))
    sns.set(style="whitegrid")
    sns.lineplot(data=df_median, x=df_median.index, y='median_diff', marker='o', linewidth=2)

    # Set the title and axis labels
    plt.title('Median Difference between geo and similarity Methods', fontsize=16, weight='bold')
    plt.xlabel('Number of Nearest Stations (N)', fontsize=14)
    plt.ylabel('Median Difference (geo - similarity)', fontsize=14)

    # Save the figure in SVG format
    if save_path:
        plt.savefig(save_path, format='svg')

    plt.tight_layout()
    plt.show()


def plot_kge_boxplots_comparison(df_long, save_path=None):
    """
    绘制不同 N 值下 geo 和 similarity 方法的 KGE 箱线图对比
    将 N 值一分为二，分别绘制两个子图（2 行 × 1 列）
    """
    df_small_n = df_long[df_long['N'] <= 100]
    df_large_n = df_long[df_long['N'] > 100]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    sns.set(style="whitegrid")

    # N <= 100
    if not df_small_n.empty:
        sns.boxplot(
            data=df_small_n, x='N', y='KGE', hue='method',
            showfliers=False, ax=axes[0],
            palette={"geo": "#F09BA0", "similarity": "#9BBBE1"}, width=0.8
        )
        axes[0].set_title('KGE Comparison for N <= 100', fontsize=18, weight='bold')
        axes[0].set_xlabel('Number of Nearest Stations (N)', fontsize=18, weight='bold')
        axes[0].set_ylabel('KGE', fontsize=18, weight='bold')
        axes[0].tick_params(axis='x', rotation=45, labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].legend(title='Method', title_fontsize=18, fontsize=18, prop={'weight': 'bold'})
    else:
        axes[0].set_visible(False)

    # N > 100
    if not df_large_n.empty:
        sns.boxplot(
            data=df_large_n, x='N', y='KGE', hue='method',
            showfliers=False, ax=axes[1],
            palette={"geo": "#F09BA0", "similarity": "#9BBBE1"}, width=0.8
        )
        axes[1].set_title('KGE Comparison for N > 100', fontsize=18, weight='bold')
        axes[1].set_xlabel('Number of Nearest Stations (N)', fontsize=18, weight='bold')
        axes[1].set_ylabel('KGE', fontsize=18, weight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=18)
        axes[1].tick_params(axis='y', labelsize=18)
        axes[1].legend(
            title='Method', title_fontsize=18, fontsize=18,
            prop={'weight': 'bold'}, loc='lower right', bbox_to_anchor=(1, 0)
        )
    else:
        axes[1].set_visible(False)

    fig.subplots_adjust(hspace=0.15, top=0.95, bottom=0.05)
    plt.tight_layout()

    # Save the figure in SVG format
    if save_path:
        plt.savefig(save_path, format='svg')
    plt.show()


def extract_n_from_filename(filename):
    """
    从文件名中提取 N 值（匹配第一个数字串）
    """
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None


def read_kge_with_method(files, method_label):
    """
    读取KGE CSV文件，添加方法标签（geo 或 similarity）和 N 值
    """
    records = []
    for file in files:
        if not os.path.exists(file):
            print(f"[WARN] 文件不存在，已跳过：{file}")
            continue

        df = pd.read_csv(file)

        # Process the KGE column: if a value looks like "[0.85]", extract the first element
        df['KGE'] = df['KGE'].apply(
            lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith('[') else x
        )
        df['KGE'] = pd.to_numeric(df['KGE'], errors='coerce')

        N = extract_n_from_filename(os.path.basename(file))

        for _, row in df.iterrows():
            records.append({
                'gauge_id': row.get('gauge_id'),
                'KGE': row.get('KGE'),
                'N': N,
                'method': method_label
            })

    out = pd.DataFrame(records)
    return out


def compute_and_save_iqr(df_long, out_csv="kge_iqr_by_N_method.csv"):
    """
    基于长表 df_long（含列：N, method, KGE）计算每个 (N, method) 的
    Q1, Q3, IQR，并保存为CSV。
    """
    df = df_long.dropna(subset=['KGE']).copy()

    # Use groupby and quantile to compute the quartiles
    q25 = df.groupby(['N', 'method'])['KGE'].quantile(0.25)
    q75 = df.groupby(['N', 'method'])['KGE'].quantile(0.75)
    med = df.groupby(['N', 'method'])['KGE'].median()
    cnt = df.groupby(['N', 'method'])['KGE'].size()

    stat = pd.concat(
        [q25.rename('Q1'), q75.rename('Q3'), med.rename('median'), cnt.rename('count')],
        axis=1
    ).reset_index()
    stat['IQR'] = stat['Q3'] - stat['Q1']

    # Save the outputs and print the summary
    stat = stat.sort_values(['N', 'method']).reset_index(drop=True)
    stat.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("\n=== IQR 统计（每个 N × method）===\n")
    print(stat)

    return stat


def plot_combined_kge_and_iqr(df_long, df_iqr, save_path=None):
    """
    将箱线图和IQR折线图合并到同一张图里，使用双Y轴
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    sns.set(style="whitegrid")

    # Plot boxplots with N on the x-axis, KGE on the y-axis, and colors by method
    sns.boxplot(
        data=df_long, x='N', y='KGE', hue='method', showfliers=False,
        palette={"geo": "#F09BA0", "similarity": "#9BBBE1"}, width=0.8, ax=ax1
    )
    ax1.set_title('KGE and IQR Comparison', fontsize=16, weight='bold')
    ax1.set_xlabel('Number of Nearest Stations (N)', fontsize=14)
    ax1.set_ylabel('KGE', fontsize=14)
    ax1.legend(title='Method', title_fontsize=14, fontsize=12)

    # Create a second y-axis for the IQR line plot
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_iqr, x='N', y='IQR', hue='method', marker='o', linewidth=2, ax=ax2
    )
    ax2.set_ylabel('IQR', fontsize=14)
    ax2.legend(title='Method', title_fontsize=14, fontsize=12, loc='upper left')

    # Adjust the legend position to avoid overlap
    ax1.legend(loc='upper right')

    # Save the figure in SVG format
    if save_path:
        plt.savefig(save_path, format='svg')

    plt.tight_layout()
    plt.show()


def plot_q3_q1_difference(df_iqr, save_path=None):
    """
    绘制75分位数（Q3）和25分位数（Q1）差值的折线图
    """
    # Compute the Q3-Q1 difference
    df_iqr['Q3_Q1_diff'] = df_iqr['Q3'] - df_iqr['Q1']

    # Create a new figure
    plt.figure(figsize=(14, 4))
    sns.set(style="whitegrid")

    # Plot the Q3-Q1 difference against N and color by method
    sns.lineplot(
        data=df_iqr, x='N', y='Q3_Q1_diff', hue='method', marker='o', linewidth=2
    )

    # Set the title and axis labels
    plt.title('Difference between 75th and 25th Percentiles (Q3 - Q1)', fontsize=16, weight='bold')
    plt.xlabel('Number of Nearest Stations (N)', fontsize=14)
    plt.ylabel('Q3 - Q1 (Difference)', fontsize=14)
    plt.legend(title='Method', title_fontsize=14, fontsize=12)

    # Save the figure in SVG format
    if save_path:
        plt.savefig(save_path, format='svg')

    plt.tight_layout()
    plt.show()


def main():
    # Split the provided mixed list into geo/similarity groups based on filenames
    all_list = sorted([
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
    ])

    geo_files = [f for f in all_list if "_geo" in os.path.basename(f)]
    similarity_files = [f for f in all_list if "_similarity" in os.path.basename(f)]

    df_geo = read_kge_with_method(geo_files, method_label="geo")
    df_sim = read_kge_with_method(similarity_files, method_label="similarity")

    df_all = pd.concat([df_geo, df_sim], ignore_index=True)

    # Compute and export the IQR values (saved as CSV)
    df_iqr = compute_and_save_iqr(df_all, out_csv="kge_iqr_by_N_method.csv")

    # Draw the boxplots
    plot_kge_boxplots_comparison(df_all, save_path="kge_comparison_by_n_range.svg")

    # Plot the line chart of the 75th-25th percentile difference
    plot_q3_q1_difference(df_iqr, save_path="q3_q1_difference.svg")

    # Plot the line chart of median differences
    plot_median_difference(df_all, save_path="median_difference.svg")


if __name__ == "__main__":
    main()
