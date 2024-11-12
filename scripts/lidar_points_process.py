import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy import stats


# 加载点云文件
point_cloud = o3d.io.read_point_cloud("04_0586-sample.ply")
points = np.asarray(point_cloud.points)


valid_mask = np.isfinite(points).all(axis=1)
points = points[valid_mask]

# ######################################################
# #########   GTA-UAV-Lidar
# points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
# points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
# points[:, 0] = -points[:, 0]
# points[:, 1] = points[:, 1]
# points[:, 2] = -points[:, 2]
# #######################################################

######################################################
#########   UAV-VisLoc-Lidar
# points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
# points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
# points[:, 0] = -points[:, 0]
# points[:, 1] = points[:, 1]
# points[:, 2] = -points[:, 2]
#######################################################

# 计算 Z-score，过滤掉 Z-score 绝对值大于设定阈值（如 3）的数据点
z_scores = np.abs(stats.zscore(points))
threshold = 3  # 通常使用 3 作为阈值
valid_mask = (z_scores < threshold).all(axis=1)

# 保留非异常值的点
points = points[valid_mask]


# 定义颜色映射（基于高度）
z_vals = points[:, 2]  # 假设z轴为高度
norm = Normalize(vmin=z_vals.min(), vmax=z_vals.max())

cmap = cm.get_cmap('viridis')    # 原始代码中的颜色映射
# cmap = cm.get_cmap('plasma')     # 深红到黄色
# cmap = cm.get_cmap('inferno')    # 深蓝到黄白
# cmap = cm.get_cmap('coolwarm')   # 冷暖渐变
# cmap = cm.get_cmap('jet')          # 蓝到红色渐变

colors = cmap(norm(z_vals))

# 创建图像
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=90, azim=-90)  # 俯视视角（从上向下）

# 设置缩放因子，进一步放大点云显示
zoom_factor = 0.4  # 可以根据需要调整这个值，值越小放大效果越明显

# 获取点云范围并缩放
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()
z_min, z_max = points[:, 2].min(), points[:, 2].max()


# 获取每个轴的范围
max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min])

# 计算中心点
x_center, y_center, z_center = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2

# 设置各轴范围，使得比例一致
ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)

# 绘制点云
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5, marker='o', linewidth=0)

ax.set_box_aspect([1, 1, 1]) 

# 隐藏坐标轴
ax.set_axis_off()

ax.dist = 6

plt.show()
