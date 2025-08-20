# ---------3D瀑布图---------
# Python版本：3.6及以上
# 依赖库安装：
# pip install numpy pandas matplotlib openpyxl
# 或使用清华镜像加速安装：
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas matplotlib openpyxl

import numpy as np  # 数值计算
import pandas as pd  # 数据处理
import matplotlib.pyplot as plt  # 绘图基础
from matplotlib import cm  # 颜色映射
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 3D多边形绘制
from matplotlib.lines import Line2D  # 图例补丁

# ---------1. 从Excel读取数据---------
# 替换为**实际Excel文件路径**
file_path = r'F:\input.xlsx'  
# 读取Excel（指定openpyxl引擎，第一行为表头）
data = pd.read_excel(file_path, engine='openpyxl', sheet_name='Sheet1', header=0)

# 定义**分组列的绘图顺序**（需与Excel中列名一致）
drawing_order = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6', 'Group7']

# ---------2. 颜色映射配置---------
# 使用plasma渐变色生成每组的颜色
plasma_colors = cm.plasma(np.linspace(0, 1, len(drawing_order)))  
# 构建“组名→颜色”的字典
colors = {drawing_order[i]: tuple(plasma_colors[i]) for i in range(len(drawing_order))}

# 提取x轴数据（从“Count”列获取）
x_values = data['Count'].values  


# ---------3. 创建3D画布与坐标轴---------
fig = plt.figure(figsize=(10, 10))  # 画布大小
ax = fig.add_subplot(111, projection='3d')  # 添加3D子图
legend_patches = []  # 用于存储图例元素


# ---------4. 逐组绘制图形---------
for i, y_label in enumerate(reversed(drawing_order)):  # 反向遍历，实现“从下到上”绘图
    color = colors[y_label]  # 当前组的颜色
    z_values = data[y_label].values  # 当前组的z轴数据（数值列）
    x_valid = x_values  # 有效x值（可根据需求筛选）
    z_valid = z_values  # 有效z值（可根据需求筛选）

    # 生成y轴坐标（用组在drawing_order中的索引表示）
    y_val = drawing_order.index(y_label)
    y_points = np.full_like(x_valid, y_val)  # 与x长度一致的y值数组

    # 绘制折线与散点
    ax.plot(x_valid, y_points, z_valid, color=color, alpha=1, linewidth=1.5)
    ax.scatter(x_valid, y_points, z_valid, color=color, marker='o', s=20)

    # 为**奇数x值**的点添加数值标签
    for x, y, z in zip(x_valid, y_points, z_valid):
        if int(x) % 2 == 1:
            label = f'{z:.1f}' if isinstance(z, float) else str(z)  # 格式化标签
            ax.text(x, y, z + 0.5, label, ha='center', va='bottom', fontsize=8)

    # 构建3D多边形（实现瀑布“填充”效果）
    verts = []
    for j in range(len(x_valid)):
        if j == 0:
            verts.append([x_valid[j], y_val, 0])    # 第一个点的“底部”
            verts.append([x_valid[j], y_val, z_valid[j]])  # 第一个点的“顶部”
        else:
            verts.append([x_valid[j], y_val, z_valid[j]])    # 当前点“顶部”
            verts.append([x_valid[j-1], y_val, z_valid[j-1]])  # 前一点“顶部”
            verts.append([x_valid[j-1], y_val, 0])        # 前一点“底部”
            verts.append([x_valid[j], y_val, 0])          # 当前点“底部”
    # 闭合多边形（连接回第一个点的底部和顶部）
    verts.append([x_valid[0], y_val, 0])
    verts.append([x_valid[0], y_val, z_valid[0]])

    # 添加3D多边形到画布
    poly = Poly3DCollection([verts], alpha=0.2, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)

    # 为当前组创建图例元素
    legend_patches.append(Line2D([0], [0], color=color, lw=2, label=y_label))


# ---------5. 图形美化与显示---------
# 坐标轴标签
ax.set_xlabel('Count', fontsize=12, labelpad=10)
ax.set_ylabel('Group', fontsize=12, labelpad=10)
ax.set_zlabel('Value', fontsize=12, labelpad=10)

# y轴刻度与标签（对应分组）
ax.set_yticks(range(len(drawing_order)))
ax.set_yticklabels(drawing_order)

# 坐标轴范围（添加边距）
margin = 0.1
x_min, x_max = min(x_values), max(x_values)
y_min, y_max = 0, len(drawing_order) - 1
z_max = data[drawing_order].values.max() * 1.1  # z轴最大值适当放大（避免顶边拥挤）
ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_zlim(0, z_max)

# 3D视角调整（仰角、方位角）
ax.view_init(elev=30, azim=-35)

# 图例与标题
ax.legend(handles=legend_patches, loc='upper right')
ax.set_title('3D Waterfall Plot', fontsize=15, pad=20)

# 自动调整布局并显示
plt.tight_layout()
plt.show()
