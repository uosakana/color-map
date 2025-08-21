import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import re

# 配置字体
plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.edgecolor"] = "#000000"  # 黑色边框

def parse_raw_data(raw_str):
    """Convert raw text data to numpy matrix"""
    lines = [line.strip() for line in raw_str.strip().split('\n') if line.strip()]
    matrix = []
    for i, line in enumerate(lines):
        elements = re.split(r'[\t\s]+', line)
        try:
            row = [float(elem) for elem in elements]
            matrix.append(row)
        except ValueError as e:
            raise ValueError(f"Row {i+1} format error: {str(e)}\nOriginal content: {line}")
    row_lengths = [len(row) for row in matrix]
    if len(set(row_lengths)) > 1:
        raise ValueError(f"Inconsistent row lengths: {row_lengths}")
    return np.array(matrix, dtype=np.float64)

def normalize_columns(data):
    """Normalize data by column (x' = x / max_value)"""
    col_max = np.max(data, axis=0)
    col_max[col_max == 0] = 1.0  # Avoid division by zero
    return data / col_max

def main():
    # --------------------------
    # 数据和配置
    # --------------------------
    raw_data = """
20.14191632	2.796680498	1.002457298	0	0	0
3.395191931	13.26075562	2.310005059	0.474936756	0.082722826	0.362522057
1.273447394	2.744631288	13.79679106	1.164339923	0.685138828	0.730148727
0.713611348	0.831913809	1.749885567	11.50070271	1.5439221	1.238372826
0.22614963	0.354250079	0.599653087	0.687904057	24.65739614	6.268086715
0	0	0	0.047596739	5.221436281	33.52230905
    """
    
    # 通道之间的间距（按顺序对应）
    # channel2与3之间:400nm, channel3与4之间:500nm, 以此类推
    channel_spacings = [400, 500, 600, 700, 800]
    
    try:
        # 1. 解析数据
        data = parse_raw_data(raw_data)
        print(f"Original data shape: {data.shape[0]} rows x {data.shape[1]} columns")
        
        # 2. 归一化处理
        normalized_data = normalize_columns(data)
        print("Completed column-wise normalization (max normalization)")
        
        # 3. 定义标签
        num_rows, num_cols = normalized_data.shape
        row_labels = [f"irradiate point {i+2}" for i in range(num_rows)]
        col_labels = [f"channel {i+2}" for i in range(num_cols)]  # channel2到channel7
        
        # 4. 蓝色系颜色映射
        blue_cmap = LinearSegmentedColormap.from_list(
            'smooth_blue',
            [
                (0.7, 0.85, 1.0, 0.9),    # 浅蓝色
                (0.4, 0.65, 0.95, 0.9),   # 中浅蓝色
                (0.2, 0.45, 0.85, 0.9),   # 中蓝色
                (0.1, 0.3, 0.7, 0.9)      # 深蓝色
            ]
        )
        norm = Normalize(
            vmin=max(normalized_data.min(), 0.05),
            vmax=min(normalized_data.max(), 0.95)
        )
        
        # 5. 创建画布，调整宽度以容纳所有标签和间距
        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        
        # 6. 绘制矩阵方格
        for i in range(num_rows):
            for j in range(num_cols):
                value = normalized_data[i, j]
                rect = plt.Rectangle(
                    (j, i), 1, 1, 
                    facecolor=blue_cmap(norm(value)),
                    edgecolor='none',
                    linewidth=0
                )
                ax.add_patch(rect)
                
                # 显示三位小数
                ax.text(
                    j + 0.5, i + 0.5, 
                    f'{value:.3f}',
                    ha='center', va='center', 
                    fontsize=12,
                    fontweight='bold',
                    color='white'
                )
        
        # 7. 配置x轴，为标签和间距创建复合刻度
        # 主刻度位置（channel标签）
        major_ticks = [j + 0.5 for j in range(num_cols)]
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(
            col_labels, 
            fontsize=11, 
            fontweight='bold',
            rotation=0,
            ha='center'
        )
        
        # 在channel标签之间添加间距标签
        for j in range(num_cols - 1):
            # 间距标签位置：两个channel标签的正中间
            spacing_pos = j + 1.0  # 正好在两个channel中间
            
            # 添加垂直线分隔（可选）
            ax.axvline(x=spacing_pos, ymin=0, ymax=-0.05, color='#999999', linestyle=':')
            
            # 添加间距值标签
            ax.text(
                spacing_pos, -0.15,  # 位于x轴下方，两个channel标签之间
                f'{channel_spacings[j]}nm',
                ha='center', 
                va='top',
                fontsize=10,
                fontweight='medium',
                color='#333333',
                rotation=0
            )
        
        # 配置y轴
        ax.set_yticks([i + 0.5 for i in range(num_rows)])
        ax.set_yticklabels(
            row_labels, 
            fontsize=11, 
            fontweight='bold'
        )
        
        # 设置轴范围
        ax.set_xlim(0, num_cols)
        ax.set_ylim(0, num_rows)
        
        # 反转y轴使第一行在上方
        ax.invert_yaxis()
        
        # 8. 添加颜色条
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=blue_cmap), 
            ax=ax,
            aspect=20,
            pad=0.03
        )
        cbar.set_label(
            'Normalized Gain', 
            fontsize=12, 
            fontweight='bold',
            labelpad=10
        )
        cbar.ax.tick_params(labelsize=10, width=1.5)
        
        # 9. 整体美化
        ax.set_title(
            '6×6 Matrix Heatmap', 
            fontsize=16, 
            pad=20, 
            fontweight='bold'
        )
        
        # 调整底部边距
        plt.subplots_adjust(bottom=0.15)
        ax.set_facecolor('#f0f0f0')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
