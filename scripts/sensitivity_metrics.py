import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np

# ================= 配置区 =================
RESULT_DIR = "../sensitivity_results"  # 请根据实际路径修改
RAW_METRICS_FILE = os.path.join(RESULT_DIR, "sensitivity_metrics_raw.csv")
CASE_STUDY_FILE = os.path.join(RESULT_DIR, "case_study_paths_final.csv")

# 指定药物ID
TARGET_DRUG_ID = "HEB4619"

# 输出图片名称
OUTPUT_IMG = f"Figure_Sensitivity_Analysis_{TARGET_DRUG_ID}_Styled.png"
OUTPUT_IMG_TIFF = f"Figure_Sensitivity_Analysis_{TARGET_DRUG_ID}_Styled.tiff"

# 【优化】手动定义高对比度颜色列表，避免颜色相近
DISTINCT_COLORS = [
    '#1f77b4',  # 蓝
    '#ff7f0e',  # 橙
    '#2ca02c',  # 绿
    '#9467bd',  # 紫
    '#8c564b',  # 棕
    '#e377c2',  # 粉
    '#17becf',  # 青
    '#bcbd22',  # 橄榄黄
    '#7f7f7f',  # 深灰
    '#d62728'  # 红 (备用)
]


def clean_experiment_names(df):
    """美化实验名称"""
    name_map = {
        'Baseline': 'Baseline\n(Proposed)',
        'Exp1_NoDistinction': 'Weight\n(No Dist.)',
        'Exp1_StrongDistinction': 'Weight\n(Strong)',
        'Exp2_NoDecay': 'Decay\n(No Decay)',
        'Exp2_StrongDecay': 'Decay\n(Strong)',
        'Exp3_NoBonus': 'Bonus\n(No Bonus)',
        'Exp3_HighBonus': 'Bonus\n(High)',
        'Exp4_NoISI': 'Specificity\n(No ISI)'
    }
    df['Experiment_Label'] = df['Experiment'].map(name_map).fillna(df['Experiment'])

    order = [
        'Baseline\n(Proposed)',
        'Weight\n(No Dist.)', 'Weight\n(Strong)',
        'Decay\n(No Decay)', 'Decay\n(Strong)',
        'Bonus\n(No Bonus)', 'Bonus\n(High)',
        'Specificity\n(No ISI)'
    ]
    valid_order = [o for o in order if o in df['Experiment_Label'].unique()]
    return df, valid_order


def prepare_slope_data(df_cases, target_drug_id):
    """准备右图数据：以Baseline为锚点"""
    print(f"正在准备右图数据 (Target: {target_drug_id})...")

    # 1. 获取 Baseline 数据
    df_baseline = df_cases[
        (df_cases['Drug'] == target_drug_id) &
        (df_cases['Experiment'] == 'Baseline')
        ].copy()

    # 2. 严格筛选成分路径
    df_baseline = df_baseline[df_baseline['Path'].str.contains("INT")]

    if df_baseline.empty:
        print(f"警告：药物 {target_drug_id} 无 Baseline 成分路径。")
        return None

    target_paths = df_baseline['Path'].unique()

    # 3. 获取 No-ISI 数据
    df_noisi = df_cases[
        (df_cases['Drug'] == target_drug_id) &
        (df_cases['Experiment'] == 'Exp4_NoISI')
        ].set_index('Path')

    plot_rows = []

    def extract_label(path):
        import re
        match = re.search(r'(INT\d+)', path)
        if match: return match.group(1)
        return "Unknown INT"

    for path in target_paths:
        rank_baseline = df_baseline[df_baseline['Path'] == path]['Rank'].values[0]

        if path in df_noisi.index:
            rank_noisi = df_noisi.loc[path, 'Rank']
        else:
            rank_noisi = 12  # 榜外

        plot_rows.append({
            'Path': path,
            'Label': extract_label(path),
            'Baseline': rank_baseline,
            'Exp4_NoISI': rank_noisi
        })

    return pd.DataFrame(plot_rows)


def main():
    sns.set_theme(style="ticks", font_scale=1.1)
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.5, 1]})

    # ================= 左图 (A) =================
    if os.path.exists(RAW_METRICS_FILE):
        print("绘制图 A...")
        df_raw = pd.read_csv(RAW_METRICS_FILE)
        df_raw, order = clean_experiment_names(df_raw)

        palette_colors = []
        for label in order:
            if 'Baseline' in label:
                palette_colors.append('#95a5a6')
            elif 'No ISI' in label:
                palette_colors.append('#c0392b')
            elif 'Weight' in label:
                palette_colors.append('#3498db')
            else:
                palette_colors.append('#bdc3c7')

        sns.boxplot(
            x='Experiment_Label', y='IoU', hue='Experiment_Label',
            data=df_raw, order=order, palette=palette_colors,
            ax=axes[0], legend=False, showfliers=False, width=0.6, linewidth=1.5
        )
        sns.stripplot(
            x='Experiment_Label', y='IoU', hue='Experiment_Label',
            data=df_raw, order=order, palette=palette_colors,
            ax=axes[0], legend=False,
            edgecolor='black', linewidth=0.5, alpha=0.5, size=4, jitter=0.2
        )

        axes[0].set_ylim(-0.05, 1.1)
        axes[0].set_ylabel("Ranking Consistency (IoU with Baseline)", fontweight='bold')
        axes[0].set_xlabel("")
        axes[0].set_title("(A) Global Parameter Sensitivity (N=20)", fontweight='bold', loc='left')
        axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        for tick in axes[0].get_xticklabels(): tick.set_rotation(45)
    else:
        axes[0].text(0.5, 0.5, "Raw Data Not Found", ha='center')

    # ================= 右图 (B) =================
    if os.path.exists(CASE_STUDY_FILE):
        df_cases = pd.read_csv(CASE_STUDY_FILE)
        pivot_data = prepare_slope_data(df_cases, TARGET_DRUG_ID)

        if pivot_data is not None and not pivot_data.empty:
            print(f"开始绘制右图，路径数量: {len(pivot_data)}")
            ax = axes[1]

            x_left, x_right = 0, 1

            # 【优化】生成节点颜色映射 (Identity)
            unique_labels = pivot_data['Label'].unique()
            node_color_map = {}
            for i, label in enumerate(unique_labels):
                # 循环使用高对比度颜色列表
                node_color_map[label] = DISTINCT_COLORS[i % len(DISTINCT_COLORS)]

            legend_handles = {}

            for _, row in pivot_data.iterrows():
                label = row['Label']
                y_left = row['Exp4_NoISI']  # No ISI
                y_right = row['Baseline']  # With ISI

                # 获取该成分的专属节点颜色
                node_color = node_color_map.get(label, 'gray')

                # === 【核心优化】线条颜色与样式逻辑 (Trend) ===
                # diff > 0: 排名数值变大 (e.g. 1 -> 8) -> 下降/抑制
                # diff < 0: 排名数值变小 (e.g. 12 -> 2) -> 上升/挖掘
                diff = y_right - y_left

                if diff > 0:
                    # 排名下降 -> 绿色实线
                    line_color = '#2ca02c'  # Red  #d62728
                    linestyle = '-'
                    linewidth = 2.5
                    alpha = 0.9
                    zorder = 3
                elif diff == 0:
                    # 排名稳定 -> 绿色加粗
                    line_color = '#bbbbbb'  # Red
                    linestyle = '-'
                    linewidth = 3.5  # Bold
                    alpha = 0.8
                    zorder = 2
                else:
                    # 排名上升 -> 红色虚线
                    line_color = '#d62728'  # Light Gray#bbbbbb
                    linestyle = '--'
                    linewidth = 2.0
                    alpha = 0.8
                    zorder = 1

                # 1. 画线 (Trend)
                ax.plot([x_left, x_right], [y_left, y_right],
                        color=line_color, linestyle=linestyle, linewidth=linewidth,
                        alpha=alpha, zorder=zorder)

                # 2. 画点 (Identity) - 点覆盖在线上
                # 分别画左右两个点，确保颜色对应成分
                ax.plot([x_left], [y_left], marker='o', markersize=9,
                        color=node_color, markeredgecolor='white', markeredgewidth=1.0, zorder=4)
                ax.plot([x_right], [y_right], marker='o', markersize=9,
                        color=node_color, markeredgecolor='white', markeredgewidth=1.0, zorder=4)

                # 收集图例句柄 (只显示点的颜色)
                if label not in legend_handles:
                    legend_handles[label] = mlines.Line2D([], [], color='white', marker='o',
                                                          markerfacecolor=node_color, markeredgecolor=node_color,
                                                          linestyle='None', label=label, markersize=8)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Without ISI\n(Ablation)', 'With ISI\n(Proposed)'], fontweight='bold')
            ax.set_xlim(-0.2, 1.2)

            ax.invert_yaxis()
            ax.set_yticks(np.arange(1, 11, 1))
            ax.set_ylim(10.5, 0.5)

            ax.set_ylabel("Path Ranking", fontweight='bold')
            ax.set_xlabel("")
            ax.set_title(f"(B) ISI Impact on Ingredients: {TARGET_DRUG_ID}", fontweight='bold', loc='left')

            ax.grid(axis='x', linestyle='--', alpha=0.5)

            # 图例 (只展示成分颜色)
            ax.legend(handles=legend_handles.values(), title="Ingredient ID",
                      bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            # 说明文本 (解释线条含义)
            info_text = (
                "Lines (Trend):\n"
                "—— Dropped (Green)\n"
                "—— Stable   (Gray)\n"
                "- - -   Risen  (Red)\n"
                "\nNodes (Identity):\n"
                "● Color by Ingredient"
            )
            ax.text(1.05, 0.3, info_text, transform=ax.transAxes,
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        else:
            axes[1].text(0.5, 0.5, f"No Ingredient Paths in Baseline for {TARGET_DRUG_ID}", ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_IMG_TIFF, dpi=300, bbox_inches='tight')

    print(f"绘图完成！已保存: {OUTPUT_IMG}")
    plt.show()


if __name__ == "__main__":
    main()