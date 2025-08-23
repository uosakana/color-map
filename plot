import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
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
    
    # 通道间距（已修正顺序）
    channel_spacings = [900, 800, 700, 600, 500]
    square_width = 150  # 方块宽度（相对单位）
    
    try:
        # 1. 解析数据
        data = parse_raw_data(raw_data)
        num_rows, num_cols = data.shape
        print(f"Original data shape: {num_rows} rows x {num_cols} columns")
        
        # 2. 归一化处理
        normalized_data = normalize_columns(data)
        print("Completed column-wise normalization (max normalization)")
        
        # 3. 分析数据分布（辅助颜色映射优化）
        flat_data = normalized_data.flatten()
        print(f"数据分布: 最小值={flat_data.min():.3f}, 最大值={flat_data.max():.3f}")
        print(f"0-0.2区间数据占比: {np.mean(flat_data <= 0.2):.1%}")
        
        # 4. 定义标签
        row_labels = [f"channel{i+2}" for i in range(num_rows)]
        col_labels = [f"channel {i+2}" for i in range(num_cols)]
        
        # 5. 计算方块位置（物理间距可视化）
        x_positions = [0]
        for spacing in channel_spacings:
            next_pos = x_positions[-1] + square_width + (spacing / 10)
            x_positions.append(next_pos)
        total_width = x_positions[-1] + square_width
        
        # 6. 优化颜色映射（解决过渡问题的核心）
        # 自定义蓝色系，增加低数值区域的颜色节点
        blue_cmap = LinearSegmentedColormap.from_list(
            'optimized_blue',
            [
                (0.9, 0.95, 1.0, 0.95),   # 接近0的值（更浅）
                (0.7, 0.85, 1.0, 0.95),   # 低数值（0.1左右）
                (0.5, 0.75, 1.0, 0.95),   # 中低数值（0.2左右）
                (0.3, 0.6, 0.95, 0.95),   # 中高数值（0.5左右）
                (0.15, 0.4, 0.85, 0.95)   # 高数值（1.0）- 避免过深
            ],
            N=100  # 增加插值点数，使过渡更平滑
        )
        
        # 非线性归一化：对低数值区域进行颜色拉伸，高数值区域压缩
        # gamma < 1 会增强低数值区域的颜色区分度
        gamma = 0.6  # 调整此值以控制高值色彩压缩程度
        norm = PowerNorm(gamma=gamma,
                         vmin=flat_data.min(),
                         vmax=flat_data.max())
        
        # 7. 创建画布
        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        
        # 8. 绘制矩阵方格
        for i in range(num_rows):
            for j in range(num_cols):
                x = x_positions[j]
                value = normalized_data[i, j]
                
                rect = plt.Rectangle(
                    (x, i), square_width, 1,
                    facecolor=blue_cmap(norm(value)),
                    edgecolor='none',
                    linewidth=0
                )
                ax.add_patch(rect)
                
                # 显示三位小数
                ax.text(
                    x + square_width/2, i + 0.5,
                    f'{value:.3f}',
                    ha='center', va='center', 
                    fontsize=12,
                    fontweight='bold',
                    color='white' if value > 0.3 else 'darkblue'  # 动态调整文字颜色
                )
        
        # 9. 配置坐标轴（交替显示通道和间距标签）
        tick_positions = []
        tick_labels = []
        for j in range(num_cols):
            channel_center = x_positions[j] + square_width / 2
            tick_positions.append(channel_center)
            tick_labels.append(col_labels[j])

            if j < num_cols - 1:
                spacing_x = x_positions[j] + square_width + (x_positions[j + 1] - (x_positions[j] + square_width)) / 2
                tick_positions.append(spacing_x)
                tick_labels.append(f"{channel_spacings[j]}nm")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            fontsize=11,
            fontweight='bold',
            rotation=0,
            ha='center'
        )
        ax.set_xlabel('test', fontsize=13, fontweight='bold')

        
        # Y轴配置
        ax.set_yticks([i + 0.5 for i in range(num_rows)])
        ax.set_yticklabels(
            row_labels,
            fontsize=11,
            fontweight='bold'
        )
        ax.set_ylabel('irradiate', fontsize=12, fontweight='bold')

        ax.set_xlim(0, total_width)
        ax.set_ylim(0, num_rows)
        ax.invert_yaxis()
        
        # 10. 添加优化的颜色条（反映非线性映射）
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
        
        # 设置颜色条刻度与标签
        N = 5  # 颜色条刻度数量
        tick_values = np.linspace(flat_data.min(), flat_data.max(), N)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])

        # 11. 整体美化
        ax.set_title(
            'Heatmap with Optimized Color Transition', 
            fontsize=16, 
            pad=20, 
            fontweight='bold'
        )
        

        ax.set_facecolor('#f0f0f0')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
    
