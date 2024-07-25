import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 设置图形背景
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([0, 150])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])


# 设置z轴在左边
ax.zaxis.set_rotate_label(True)  # disable automatic rotation
ax.set_zlabel('Z', rotation=0)

# # 强化格子线条
# ax.xaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)
# ax.yaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)
# ax.zaxis._axinfo['grid'].update(color='k', linestyle='-', linewidth=1.0)

# 强化坐标轴的粗线
ax.xaxis._axinfo['axisline'].update(color='k', linewidth=2)
ax.yaxis._axinfo['axisline'].update(color='k', linewidth=2)
ax.zaxis._axinfo['axisline'].update(color='k', linewidth=2)

# 保留格子线条
ax.grid(True)

# 去掉背景
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# 保存为无背景的png图片，高分辨率
plt.savefig('3d_background.png', transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()
