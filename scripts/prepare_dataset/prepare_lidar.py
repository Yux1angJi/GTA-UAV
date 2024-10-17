import os
import numpy as np

import cupy as cp
from cupyx.scipy.signal import convolve2d

import open3d as o3d
from tqdm import tqdm
import cv2
# from scipy.signal import convolve2d


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


def dense_top_down_map(Pts, n=1920, m=1080, grid=5):
    ng = 2 * grid + 1  # 定义滑动窗口的大小

    # 初始化坐标网格，使用 np.inf 作为初始值
    mX = np.full((m, n), np.inf)
    mY = np.full((m, n), np.inf)
    mD = np.zeros((m, n))

    # 将点云的 x, y 坐标映射到网格索引
    x_min, x_max = Pts[:, 0].min(), Pts[:, 0].max()
    y_min, y_max = Pts[:, 1].min(), Pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # 根据 n 和 m 将坐标映射到网格索引
    x_indices = ((Pts[:, 0] - x_min) / x_range * (n - 1)).astype(int)
    y_indices = ((Pts[:, 1] - y_min) / y_range * (m - 1)).astype(int)

    # 确保索引在有效范围内
    x_indices = np.clip(x_indices, 0, n - 1)
    y_indices = np.clip(y_indices, 0, m - 1)

    # 计算偏移量（相对于整数网格的偏移）
    x_offsets = Pts[:, 0] - (x_indices / (n - 1) * x_range + x_min)
    y_offsets = Pts[:, 1] - (y_indices / (m - 1) * y_range + y_min)

    # 更新 mX, mY, mD
    mX[y_indices, x_indices] = x_offsets
    mY[y_indices, x_indices] = y_offsets
    mD[y_indices, x_indices] = Pts[:, 2]

    # 初始化滑动窗口
    KmX = np.zeros((ng, ng, m - 2 * grid, n - 2 * grid))
    KmY = np.zeros((ng, ng, m - 2 * grid, n - 2 * grid))
    KmD = np.zeros((ng, ng, m - 2 * grid, n - 2 * grid))

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i : m - 2 * grid + i, j : n - 2 * grid + j]
            KmY[i, j] = mY[i : m - 2 * grid + i, j : n - 2 * grid + j]
            KmD[i, j] = mD[i : m - 2 * grid + i, j : n - 2 * grid + j]


    S = np.zeros((m - 2 * grid, n - 2 * grid))
    Y = np.zeros((m - 2 * grid, n - 2 * grid))

    for i in range(ng):
        for j in range(ng):
            # 计算距离的平方和，防止除以零
            dist_sq = KmX[i, j]**2 + KmY[i, j]**2 + 1e-6
            s = 1 / dist_sq  # 权重：距离的倒数的平方
            # print(dist_sq)
            # 仅考虑有效的深度值
            valid_mask = ~np.isnan(KmD[i, j])
            s = s * valid_mask
            Y += s * KmD[i, j]
            S += s

    # 避免除以零
    depth_values = Y / (S + 1e-6)

    # 将结果扩展回原始大小的数组
    out = np.zeros((m, n), dtype=np.float32)
    out[grid : m - grid, grid : n - grid] = depth_values

    # 处理无效值，用最小深度值填充
    valid_out = out[np.isfinite(out)]
    if valid_out.size > 0:
        min_depth = valid_out.min()
        max_depth = valid_out.max()
        out = np.nan_to_num(out, nan=min_depth)
        # 将深度值归一化到 0-255，点越高值越大
        depth_normalized = (out - min_depth) / (max_depth - min_depth + 1e-6) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(out, dtype=np.uint8)
    
    # print(depth_normalized.max(), depth_normalized.min())

    # 生成彩色深度图
    # colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    # colored_depth_map = cv2.resize(colored_depth_map, (384, 384), cv2.INTER_NEAREST)

    # depth_vis = cv2.convertScaleAbs(depth_normalized, alpha=255.0 / depth_normalized.max())
    # depth_vis = cv2.resize(depth_vis, (384, 384), cv2.INTER_NEAREST)

    # 可选：保存深度图像
    # cv2.imwrite('test_depth_color_grid3.png', depth_vis)

    return depth_normalized


def dense_top_down_map_conv_gpu(Pts, n=1920, m=1080, grid=5, visualize=False):
    ng = 2 * grid + 1  # Size of the sliding window

    # Transfer Pts to GPU memory
    Pts_gpu = cp.asarray(Pts)

    # Initialize coordinate grids with zeros or NaNs on GPU
    mX = cp.zeros((m, n), dtype=cp.float32)
    mY = cp.zeros((m, n), dtype=cp.float32)
    mD = cp.full((m, n), cp.nan, dtype=cp.float32)

    # Map point cloud x, y coordinates to grid indices
    x_min, x_max = Pts_gpu[:, 0].min(), Pts_gpu[:, 0].max()
    y_min, y_max = Pts_gpu[:, 1].min(), Pts_gpu[:, 1].max()
    x_range = x_max - x_min + 1e-6  # Avoid division by zero
    y_range = y_max - y_min + 1e-6

    x_indices = ((Pts_gpu[:, 0] - x_min) / x_range * (n - 1)).astype(cp.int32)
    y_indices = ((Pts_gpu[:, 1] - y_min) / y_range * (m - 1)).astype(cp.int32)

    # Ensure indices are within valid range
    x_indices = cp.clip(x_indices, 0, n - 1)
    y_indices = cp.clip(y_indices, 0, m - 1)

    # Calculate offsets (relative to integer grid)
    x_offsets = Pts_gpu[:, 0] - (x_indices / (n - 1) * x_range + x_min)
    y_offsets = Pts_gpu[:, 1] - (y_indices / (m - 1) * y_range + y_min)

    # Update mX, mY, mD
    mX[y_indices, x_indices] = x_offsets
    mY[y_indices, x_indices] = y_offsets
    mD[y_indices, x_indices] = Pts_gpu[:, 2]

    # Compute weights only where mD is valid
    valid_mask = ~cp.isnan(mD)
    s = cp.zeros_like(mX)
    s[valid_mask] = 1 / (mX[valid_mask]**2 + mY[valid_mask]**2 + 1e-6)
    sD = cp.zeros_like(mX)
    sD[valid_mask] = s[valid_mask] * mD[valid_mask]

    # Convolve s and sD with a kernel of ones
    kernel = cp.ones((ng, ng), dtype=cp.float32)
    S = convolve2d(s, kernel, mode='same', boundary='fill', fillvalue=0)
    Y = convolve2d(sD, kernel, mode='same', boundary='fill', fillvalue=0)

    # Compute depth values
    depth_values = Y / (S + 1e-6)

    # Handle invalid values, fill with minimum depth
    valid_out = depth_values[cp.isfinite(depth_values)]
    if valid_out.size > 0:
        min_depth = valid_out.min()
        max_depth = valid_out.max()
        depth_values = cp.nan_to_num(depth_values, nan=min_depth)
        # Normalize depth values to 0-255
        depth_normalized = (depth_values - min_depth) / (max_depth - min_depth + 1e-6) * 255
        depth_normalized = depth_normalized.astype(cp.uint8)
    else:
        depth_normalized = cp.zeros_like(depth_values, dtype=cp.uint8)

    # Transfer the result back to CPU memory
    depth_normalized_cpu = cp.asnumpy(depth_normalized)

    # Visualization (optional)
    if visualize:
        max_val = depth_normalized_cpu.max()
        if max_val > 0:
            depth_vis = cv2.convertScaleAbs(depth_normalized_cpu, alpha=255.0 / max_val)
        else:
            depth_vis = depth_normalized_cpu.copy()
        depth_vis = cv2.resize(depth_vis, (384, 384), cv2.INTER_NEAREST)
        cv2.imwrite('test_depth_color_grid_gpu.png', depth_vis)

    return depth_normalized_cpu


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

    img_size = (384, 384)

    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith(".ply")]

    for lidar_file in tqdm(lidar_files, desc="Processing lidar files"):
        lidar_path = os.path.join(lidar_dir, lidar_file)

        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)

        points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
        points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
        points[:, 0] = -points[:, 0]
        points[:, 1] = points[:, 1]
        points[:, 2] = -points[:, 2]
        points[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())

        depth_img = dense_top_down_map_conv_gpu(points, grid=6)

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
    lidar2rgbd()

    # rgbd = cv2.imread('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/rgbd/200_0001_0000004776.png', cv2.IMREAD_UNCHANGED)
    # bgr = cv2.imread('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/images/200_0001_0000004776.png')
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # print((rgbd[:, :, 0]==rgb[:, :, 0]).sum(), (rgbd[:, :, 1]==rgb[:, :, 1]).sum(), (rgbd[:, :, 2]==rgb[:, :, 2]).sum())
    # print(rgbd[:, :, 0].min(), rgbd[:, :, 0].max())
    # print(rgbd[:, :, 3].min(), rgbd[:, :, 3].max())
    # print(rgbd.shape, rgb.shape)

    # pcd = o3d.io.read_point_cloud('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/lidars/200_0001_0000001542.ply')
    # points = np.asarray(pcd.points)

    # points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
    # points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
    # points[:, 0] = -points[:, 0]
    # points[:, 1] = points[:, 1]
    # points[:, 2] = -points[:, 2]
    # points[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())

    # print(points[:, 0].max(), points[:, 0].min())
    # print(points[:, 1].max(), points[:, 1].min())
    # print(points[:, 2].max(), points[:, 2].min())

    # depth = dense_top_down_map_conv_gpu(points, grid=6, visualize=True)
    # print(depth.shape)



