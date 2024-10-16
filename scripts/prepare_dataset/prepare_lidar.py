import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import cv2


data_root = '/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/'


def create_top_down_depth_image(points, bev_height=1080, bev_width=1920):
    """
    将点云数据生成俯视角深度图像（鸟瞰图），自动计算投影范围。

    Args:
        points: 点云数据，形状为 (N, 3)，每个点包含 (x, y, z)
        bev_height: 输出图像的高度（像素）
        bev_width: 输出图像的宽度（像素）

    Returns:
        depth_map_uint8: 归一化后的深度图像，dtype 为 uint8
    """
    # 提取点云坐标
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # 自动计算投影范围
    x_min = np.min(x_points)
    x_max = np.max(x_points)
    y_min = np.min(y_points)
    y_max = np.max(y_points)

    # 如果需要，可以添加边距
    padding = 0  # 根据需要调整边距大小
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # 计算范围和分辨率
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_resolution = x_range / (bev_width - 1)    # 修改此处
    y_resolution = y_range / (bev_height - 1)   # 修改此处

    # 将物理坐标转换为图像坐标
    x_img = ((x_points - x_min) / x_resolution).astype(np.int32)
    y_img = ((y_points - y_min) / y_resolution).astype(np.int32)

    # 确保索引在合法范围内
    x_img = np.clip(x_img, 0, bev_width - 1)
    y_img = np.clip(y_img, 0, bev_height - 1)

    # 将图像坐标系原点设置在左下角
    y_img = bev_height - y_img - 1

    # 初始化深度图
    depth_map = np.zeros((bev_height, bev_width), dtype=np.float32)

    # 处理遮挡关系，保留最高点（z 值最大的点）
    indices = np.lexsort((-z_points, y_img, x_img))
    x_img = x_img[indices]
    y_img = y_img[indices]
    z_points = z_points[indices]
    coords = np.stack((y_img, x_img), axis=1)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    depth_map[y_img[unique_indices], x_img[unique_indices]] = z_points[unique_indices]
    # print(depth_map[0][0])
    # 将深度值归一化到 [0, 255]
    non_zero_depth = depth_map[depth_map != 0]
    if len(non_zero_depth) > 0:
        max_height = np.percentile(non_zero_depth, 99)
        min_height = np.percentile(non_zero_depth, 1)
        depth_map_clipped = np.clip(depth_map, min_height, max_height)
        depth_map_normalized = (depth_map_clipped - min_height) / (max_height - min_height)
        depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
    else:
        depth_map_uint8 = depth_map.astype(np.uint8)
    
    
    colored_depth_map = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_JET)

    return depth_map_uint8, colored_depth_map


def data_clean():

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


def lidar2rgbd():
    lidar_dir = data_root + "lidars"
    image_dir = data_root + "images"
    
    rgbd_dir = data_root + "rgbd"

    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith(".ply")]

    for lidar_file in tqdm(lidar_files, desc="Processing lidar files"):
        lidar_path = os.path.join(lidar_dir, lidar_file)

        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)

        points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
        points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
        points[:, 0] = -points[:, 0]
        points[:, 1] = points[:, 1]
        points[:, 2] = 10000-points[:, 2]

        depth_img, _ = create_top_down_depth_image(points)

        image_path = os.path.join(image_dir, lidar_file.replace('.ply', '.png'))

        bgr_img = cv2.imread(image_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        if rgb_img.shape[:2] != depth_img.shape[:2]:
            print(f"ERROR! Wrong size for {image_path}, shape={image_path.shape}")
            continue

        rgbd_img = np.dstack((rgb_img, depth_img))

        cv2.imwrite(os.path.join(rgbd_dir, lidar_file.replace('.ply', '.png')), rgbd_img)
    
    print('Lidar 2 RGBD Done!')


if __name__ == "__main__":
    # lidar2rgbd()

    rgbd = cv2.imread('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/rgbd/200_0001_0000004776.png', cv2.IMREAD_UNCHANGED)
    bgr = cv2.imread('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/images/200_0001_0000004776.png')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print((rgbd[:, :, 0]==rgb[:, :, 0]).sum(), (rgbd[:, :, 1]==rgb[:, :, 1]).sum(), (rgbd[:, :, 2]==rgb[:, :, 2]).sum())
    print(rgbd[:, :, 0].min(), rgbd[:, :, 0].max())
    print(rgbd[:, :, 3].min(), rgbd[:, :, 3].max())
    print(rgbd.shape, rgb.shape)

