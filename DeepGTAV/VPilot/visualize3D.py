import numpy as np
import open3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# filename = "E:/Bachelorarbeit/Datasets/Dataset Pregenerated DeepGTAV-PreSIL/velodyne/001171.bin"
filename = "C:/EXPORTDIR/object/velodyne/000001.bin"

with open(filename, "r") as file:
    a = np.fromfile(file, dtype = np.float32)

a = a.reshape((-1, 4))
points3d = np.delete(a, 3, 1)

point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(points3d)

open3d.visualization.draw_geometries([point_cloud], mesh_show_wireframe=True)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(50, - 90 - 90)
col = np.maximum(np.zeros(points3d[:,2].size), (points3d[:,2]/max(np.absolute(points3d[:,2])) * 255).astype(int))
ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], c=points3d[:,2], s=2)
# ax.scatter(points3d[:,1], points3d[:,0], points3d[:,2])
