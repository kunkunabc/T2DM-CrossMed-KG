import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据
data = {
    'DFS_Penalty': [0.1, 0.2, 0.3, 0.6, 0.8, 1.0],
    'Rule_Ratio': [0.8718, 0.589744, 0.2718, 0.0256, 0.0, 0.0],  # 规则覆盖率
    'Avg_Score': [0.2170, 0.2316, 0.2856, 0.5277, 0.7030, 0.8512] # 平均得分
}
df = pd.DataFrame(data)

# 2. 设置画布风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(8, 6))

# 3. 绘制左轴 (规则覆盖率 - 核心指标)
color_left = '#1f77b4' # 蓝色
line1 = ax1.plot(df['DFS_Penalty'], df['Rule_Ratio'], color=color_left, marker='o',
                 linewidth=2.5, markersize=8, label='Rule Coverage Ratio')
ax1.set_xlabel('DFS Penalty Factor ($\lambda_{DFS}$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Rule Coverage Ratio', color=color_left, fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_left, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim(-0.05, 1.05) # 稍微留白

# 4. 绘制右轴 (平均得分 - 辅助指标)
ax2 = ax1.twinx()
color_right = '#ff7f0e' # 橙色
line2 = ax2.plot(df['DFS_Penalty'], df['Avg_Score'], color=color_right, marker='s',
                 linestyle='--', linewidth=2, markersize=8, label='Average Path Score')
ax2.set_ylabel('Average Path Score', color=color_right, fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_right, labelsize=12)
ax2.set_ylim(0, 1.0)

# 5. 添加标注和垂直线 (Highlight 0.2)
selected_x = 0.2
selected_y = df.loc[df['DFS_Penalty'] == 0.2, 'Rule_Ratio'].values[0]

# 垂直虚线
plt.axvline(x=selected_x, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)

# 标注框
bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1, alpha=0.9)
ax1.annotate(f'Selected Point\n$\lambda_{{DFS}}={selected_x}$\nRatio={selected_y:.2%}',
             xy=(selected_x, selected_y), xycoords='data',
             xytext=(selected_x + 0.15, selected_y + 0.15), textcoords='data',
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, bbox=bbox_props)

# 6. 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=12, frameon=True, framealpha=0.9)

# 7. 标题和布局
plt.title('Calibration of DFS Penalty Factor ($\lambda_{DFS}$)', fontsize=16, pad=20, fontweight='bold')
plt.tight_layout()

# 8. 保存与展示
save_path = 'dfs_penalty_calibration_chart.tiff'
plt.savefig(save_path, dpi=300)
plt.show()

print(f"图表已生成并保存至: {save_path}")