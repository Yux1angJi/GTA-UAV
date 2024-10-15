import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

data_root = '/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/'

# 定义目录路径
lidar_dir = data_root + "lidars"
image_dir = data_root + "images"
meta_dir = data_root + "meta_data"

# 获取文件名
lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith(".ply")]

for lidar_file in tqdm(lidar_files, desc="Processing lidar files"):
    lidar_path = os.path.join(lidar_dir, lidar_file)
    
    # 读取点云文件
    pcd = o3d.io.read_point_cloud(lidar_path)
    points = np.asarray(pcd.points)
    
    # 检查点是否为 NaN 或 inf
    valid_points = points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]
    
    # 如果所有点都无效，删除文件
    if valid_points.shape[0] == 0:
        print(f"Deleting {lidar_file} and corresponding files...")
        
        # 删除lidars, images, meta_data中的文件
        os.remove(lidar_path)
        image_path = os.path.join(image_dir, lidar_file.replace('.ply', '.png'))  # 假设图像为jpg
        meta_path = os.path.join(meta_dir, lidar_file.replace('.ply', '.txt'))  # 假设meta_data为json
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
    
    else:
        # 更新点云文件，去除无效点
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(valid_points)
        
        # 重新保存点云文件
        o3d.io.write_point_cloud(lidar_path, new_pcd)

print("Processing complete.")
