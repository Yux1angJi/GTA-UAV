import os
import open3d as o3d
import cupy as cp
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2
import laspy


from cupyx.scipy.signal import convolve2d

def read_camera_info(file_path):
    """
    读取相机的位置信息和姿态
    """
    camera_info = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            parts = line.strip().split()
            image_name = parts[0]
            x, y, z = map(float, parts[1:4])
            omega, phi, kappa = map(float, parts[4:7])
            camera_info.append({
                'image_name': image_name,
                'position': cp.array([x, y, z]),
                'orientation': (omega, phi, kappa)
            })
    return camera_info

def filter_points_within_frustum_gpu(points, colors, camera_pos, orientation, h_fov, v_fov, max_distance=2000):
    """
    使用GPU加速的方法快速筛选在相机视锥内的点
    :param points: 点的坐标数组 (N, 3)，为cupy数组
    :param colors: 点的颜色数组 (N, 3)，为cupy数组
    :param camera_pos: 相机位置 (x, y, z)，为cupy数组
    :param orientation: 相机的欧拉角 (omega, phi, kappa)
    :param h_fov: 水平视场角 (默认90度)
    :param v_fov: 垂直视场角 (默认60度)
    :param max_distance: 最大距离 (默认500米)
    :return: 筛选出的点和颜色
    """
    # 确保points和colors的大小一致
    if points.shape[0] != colors.shape[0]:
        raise ValueError("Points and colors arrays must have the same length.")

    # 计算相机到点的矢量
    vectors_to_points = points - camera_pos
    distances = cp.linalg.norm(vectors_to_points, axis=1)
    
    # 筛选距离在范围内的点
    within_distance = distances <= max_distance
    if cp.sum(within_distance) == 0:
        return [], []

    vectors_to_points = vectors_to_points[within_distance]
    colors = colors[within_distance]

    # 旋转矢量到相机坐标系（俯拍相机方向向下，光轴沿-Z）
    rotation = R.from_euler('xyz', orientation, degrees=True)
    vectors_camera_space = cp.asarray(rotation.inv().apply(cp.asnumpy(vectors_to_points)))
    
    # 计算水平夹角和垂直夹角（相机光轴沿-Z方向）
    horizontal_angles = cp.degrees(cp.arctan2(vectors_camera_space[:, 0], -vectors_camera_space[:, 2]))
    vertical_angles = cp.degrees(cp.arctan2(vectors_camera_space[:, 1], -vectors_camera_space[:, 2]))
    
    # 筛选夹角在水平和垂直视场范围内的点
    within_h_fov = cp.abs(horizontal_angles) <= h_fov / 2
    within_v_fov = cp.abs(vertical_angles) <= v_fov / 2
    within_fov = cp.logical_and(within_h_fov, within_v_fov)

    if cp.sum(within_fov) == 0:
        return [], []

    return vectors_to_points[within_fov], colors[within_fov]

def process_point_cloud_in_chunks(camera_info, ply_file, output_folder, offset_file, chunk_size=100000, h_fov=60, v_fov=42):
    """
    分块处理点云数据，并将结果累积保存
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(offset_file, 'r') as f:
        lines = f.readlines()
        offset_x, offset_y, offset_z = map(float, lines[0].split())

    # 读取PLY文件
    print(f"Reading PLY file: {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)

    # 提取点和颜色信息
    points = np.asarray(pcd.points)
    print(points[:, 0].min(), points[:, 0].max())
    print(points[:, 1].min(), points[:, 1].max())
    print(points[:, 2].min(), points[:, 2].max())
    colors = np.asarray(pcd.colors)

    num_points = points.shape[0]
    num_chunks = (num_points + chunk_size - 1) // chunk_size

    # 为每个相机创建累积点云
    for camera in tqdm(camera_info, desc="Processing cameras"):
        camera_name = os.path.splitext(camera['image_name'])[0]
        output_path = os.path.join(output_folder, f"{camera_name}.ply")

        selected_points = []
        selected_colors = []

        # 分块处理点云
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_points)

            # 转换当前块为 cupy 数组
            chunk_points = cp.asarray(points[start_idx:end_idx])
            chunk_colors = cp.asarray(colors[start_idx:end_idx])

            # 应用偏移
            # chunk_points[:, 0] += offset_x
            # chunk_points[:, 1] += offset_y
            # chunk_points[:, 2] += offset_z

            # 筛选视锥内的点和颜色
            points_in_frustum, colors_in_frustum = filter_points_within_frustum_gpu(
                chunk_points, chunk_colors, camera['position'], camera['orientation'], h_fov, v_fov
            )

            # 累积筛选出的点
            if len(points_in_frustum) > 0:
                selected_points.append(points_in_frustum)
                selected_colors.append(colors_in_frustum)

        # 合并所有符合条件的点和颜色
        if selected_points:
            selected_points = cp.vstack(selected_points)
            selected_colors = cp.vstack(selected_colors)
            selected_points_np = cp.asnumpy(selected_points)
            selected_colors_np = cp.asnumpy(selected_colors)

            # 创建新的点云对象并保存（带颜色）
            selected_pcd = o3d.geometry.PointCloud()
            selected_pcd.points = o3d.utility.Vector3dVector(selected_points_np)
            selected_pcd.colors = o3d.utility.Vector3dVector(selected_colors_np)
            o3d.io.write_point_cloud(output_path, selected_pcd)
            print(f"Saved: {output_path}")
        else:
            print(f"No points found for camera {camera_name}")


def dense_top_down_map(Pts, omega, phi, kappa, n=3976, m=2652, grid=5):
    """
    Generate a top-down depth map from a point cloud given omega, phi, and kappa angles.

    Parameters:
    - Pts: numpy array of shape (N, 3), the point cloud data.
    - omega: rotation angle around the X-axis in degrees.
    - phi: rotation angle around the Y-axis in degrees.
    - kappa: rotation angle around the Z-axis in degrees.
    - n: width of the output depth map.
    - m: height of the output depth map.
    - grid: size of the grid for the sliding window.

    Returns:
    - depth_normalized: a 2D numpy array representing the depth map.
    """
    # Convert angles from degrees to radians
    omega_rad = np.deg2rad(omega)
    phi_rad = np.deg2rad(phi)
    kappa_rad = np.deg2rad(kappa)

    # Rotation matrices around X-axis (omega), Y-axis (phi), and Z-axis (kappa)
    R_omega = np.array([
        [1, 0, 0],
        [0, np.cos(omega_rad), -np.sin(omega_rad)],
        [0, np.sin(omega_rad),  np.cos(omega_rad)]
    ])

    R_phi = np.array([
        [ np.cos(phi_rad), 0, np.sin(phi_rad)],
        [0, 1, 0],
        [-np.sin(phi_rad), 0, np.cos(phi_rad)]
    ])

    R_kappa = np.array([
        [np.cos(kappa_rad), -np.sin(kappa_rad), 0],
        [np.sin(kappa_rad),  np.cos(kappa_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    # The order of multiplication is important: R = R_kappa * R_phi * R_omega
    # Adjust the order according to the rotation sequence in your application
    R = R_kappa @ R_phi @ R_omega

    # Apply rotation to the point cloud
    rotated_Pts = Pts @ R.T  # Transpose because we are using row vectors

    ng = 2 * grid + 1  # Define the sliding window size

    # Initialize coordinate grids, use np.inf as initial values
    mX = np.full((m, n), np.inf)
    mY = np.full((m, n), np.inf)
    mD = np.zeros((m, n))

    # Map point cloud x, y coordinates to grid indices
    x_min, x_max = rotated_Pts[:, 0].min(), rotated_Pts[:, 0].max()
    y_min, y_max = rotated_Pts[:, 1].min(), rotated_Pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Map coordinates to grid indices based on n and m
    x_indices = ((rotated_Pts[:, 0] - x_min) / x_range * (n - 1)).astype(int)
    y_indices = ((rotated_Pts[:, 1] - y_min) / y_range * (m - 1)).astype(int)

    # Ensure indices are within valid range
    x_indices = np.clip(x_indices, 0, n - 1)
    y_indices = np.clip(y_indices, 0, m - 1)

    # Calculate offsets (offsets relative to integer grid)
    x_offsets = rotated_Pts[:, 0] - (x_indices / (n - 1) * x_range + x_min)
    y_offsets = rotated_Pts[:, 1] - (y_indices / (m - 1) * y_range + y_min)

    # Update mX, mY, mD
    mX[y_indices, x_indices] = x_offsets
    mY[y_indices, x_indices] = y_offsets
    mD[y_indices, x_indices] = rotated_Pts[:, 2]

    # Initialize sliding window
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
            # Calculate sum of squared distances to prevent division by zero
            dist_sq = KmX[i, j]**2 + KmY[i, j]**2 + 1e-6
            s = 1 / dist_sq  # Weight: inverse square of distance
            # Only consider valid depth values
            valid_mask = ~np.isnan(KmD[i, j])
            s = s * valid_mask
            Y += s * KmD[i, j]
            S += s

    # Avoid division by zero
    depth_values = Y / (S + 1e-6)

    # Expand the result back to the original size array
    out = np.zeros((m, n), dtype=np.float32)
    out[grid : m - grid, grid : n - grid] = depth_values

    # Handle invalid values, fill with minimum depth value
    valid_out = out[np.isfinite(out)]
    if valid_out.size > 0:
        min_depth = valid_out.min()
        max_depth = valid_out.max()
        out = np.nan_to_num(out, nan=min_depth)
        # Normalize depth values to 0-255, higher points have higher values
        depth_normalized = (out - min_depth) / (max_depth - min_depth + 1e-6) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(out, dtype=np.uint8)
    
    colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    # colored_depth_map = cv2.resize(colored_depth_map, (384, 384), cv2.INTER_NEAREST)
    
    depth_vis = cv2.convertScaleAbs(depth_normalized, alpha=255.0 / depth_normalized.max())
    # depth_vis = cv2.resize(depth_vis, (384, 384), cv2.INTER_NEAREST)
    cv2.imwrite('test_depth_color_grid3.png', depth_vis)
    cv2.imwrite('test_depth_color.png', colored_depth_map)

    return depth_normalized



def lidar2rgbd(data_root):
    lidar_dir = data_root + "lidar"
    image_dir = data_root + "drone"
    
    depth_dir = data_root + "depth"

    os.makedirs(depth_dir, exist_ok=True)

    external_camera_file = data_root + "calibrated_external_camera_parameters.txt"

    image_files = []
    yaws = []
    with open(external_camera_file, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            info = lines[i].split(' ')
            image_files.append(info[0])
            yaws.append(float(info[6]))

    lidar_files = [f.replace('.JPG', '.ply') for f in image_files]

    for lidar_file, yaw in tqdm(zip(lidar_files, yaws), desc="Processing lidar files", total=len(yaws)):
        lidar_path = os.path.join(lidar_dir, lidar_file)

        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)

        points[:, 2] = points[:, 2]
        points[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())

        depth_img = dense_top_down_map_yaw_conv_gpu(points, yaw_angle=-1.0 * yaw, grid=6)
        # print(lidar_file, yaw)

        cv2.imwrite(os.path.join(depth_dir, lidar_file.replace('.ply', '.png')), depth_img)
    
    print('Lidar 2 RGBD Done!')


def dense_top_down_map_yaw_conv_gpu(Pts, yaw_angle, n=3976, m=2652, grid=5, visualize=False):
    
    # 将点云数据转换为 CuPy 数组
    Pts = cp.asarray(Pts)
    
    # 将偏航角从度数转换为弧度
    theta = cp.deg2rad(yaw_angle)
    # print(yaw_angle, theta)
    
    # 绕 Z 轴的旋转矩阵
    cos_theta = cp.cos(theta).item()
    sin_theta = cp.sin(theta).item()
    Rz = cp.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    
    # 对点云应用旋转
    rotated_Pts = Pts @ Rz.T  # 转置因为我们使用行向量
    
    # 初始化坐标网格，使用 cp.nan 作为初始值
    mD = cp.full((m, n), cp.nan)
    
    # 将点云 x, y 坐标映射到网格索引
    x_min, x_max = rotated_Pts[:, 0].min(), rotated_Pts[:, 0].max()
    y_min, y_max = rotated_Pts[:, 1].min(), rotated_Pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # 根据 n 和 m 将坐标映射到网格索引
    # x_indices = ((x_max - rotated_Pts[:, 0]) / x_range * (n - 1)).astype(cp.int32)
    x_indices = ((rotated_Pts[:, 0] - x_min) / x_range * (n - 1)).astype(cp.int32)
    y_indices = ((y_max - rotated_Pts[:, 1]) / y_range * (m - 1)).astype(cp.int32)
    # y_indices = ((rotated_Pts[:, 1] - y_min) / y_range * (m - 1)).astype(cp.int32)
    
    x_indices = cp.clip(x_indices, 0, n - 1)
    y_indices = cp.clip(y_indices, 0, m - 1)
    
    # 更新深度矩阵 mD
    mD[y_indices, x_indices] = rotated_Pts[:, 2]
    
    # 创建权重核
    ng = 2 * grid + 1  # 定义核的大小
    dy, dx = cp.mgrid[-grid:grid+1, -grid:grid+1]
    dist_sq = dx**2 + dy**2 + 1e-6
    W = 1 / dist_sq  # 权重：距离平方的倒数
    W[grid, grid] = 1 / (1e-6)  # 防止中心点除以零
    
    # 有效像素掩码
    valid_mask = ~cp.isnan(mD)
    
    # 使用卷积计算加权和 S 和深度值和 Y
    S = convolve2d(valid_mask.astype(cp.float32), W, mode='same', boundary='fill', fillvalue=0)
    Y = convolve2d(cp.nan_to_num(mD, nan=0), W, mode='same', boundary='fill', fillvalue=0)
    
    # 计算深度值，避免除以零
    depth_values = Y / (S + 1e-6)
    
    # 处理无效值，用最小深度值填充
    valid_out = depth_values[cp.isfinite(depth_values)]
    if valid_out.size > 0:
        min_depth = valid_out.min()
        max_depth = valid_out.max()
        depth_values = cp.nan_to_num(depth_values, nan=min_depth)
        # 将深度值归一化到 0-255，较高的点具有较高的值
        depth_normalized = (depth_values - min_depth) / (max_depth - min_depth + 1e-6) * 255
        depth_normalized = depth_normalized.astype(cp.uint8)
    else:
        depth_normalized = cp.zeros_like(depth_values, dtype=cp.uint16)
    
    # 如果需要，将结果传回 CPU
    depth_normalized_cpu = cp.asnumpy(depth_normalized)

    if visualize:
        colored_depth_map = cv2.applyColorMap(depth_normalized_cpu, cv2.COLORMAP_JET)
        # colored_depth_map = cv2.resize(colored_depth_map, (384, 384), cv2.INTER_NEAREST)
        
        depth_vis = cv2.convertScaleAbs(depth_normalized_cpu, alpha=255.0 / depth_normalized_cpu.max())
        # depth_vis = cv2.resize(depth_vis, (384, 384), cv2.INTER_NEAREST)
        cv2.imwrite('test_depth_color_grid3.png', depth_vis)
        cv2.imwrite('test_depth_color.png', colored_depth_map)
    
    return depth_normalized_cpu


def dense_top_down_map_yaw(Pts, yaw_angle, n=3976, m=2652, grid=5):
    # Convert yaw_angle from degrees to radians
    theta = np.deg2rad(yaw_angle)
    
    # Rotation matrix around Z-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    Rz = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,          0,         1]
    ])
    
    # Apply rotation to the point cloud
    rotated_Pts = Pts @ Rz.T  # Transpose because we are using row vectors
    
    ng = 2 * grid + 1  # Define the sliding window size
    
    # Initialize coordinate grids, use np.inf as initial values
    mX = np.full((m, n), np.inf)
    mY = np.full((m, n), np.inf)
    mD = np.zeros((m, n))
    
    # Map point cloud x, y coordinates to grid indices
    x_min, x_max = rotated_Pts[:, 0].min(), rotated_Pts[:, 0].max()
    y_min, y_max = rotated_Pts[:, 1].min(), rotated_Pts[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Map coordinates to grid indices based on n and m
    x_indices = ((x_max - rotated_Pts[:, 0]) / x_range * (n - 1)).astype(int)
    y_indices = ((rotated_Pts[:, 1] - y_min) / y_range * (m - 1)).astype(int)
    
    x_indices = np.clip(x_indices, 0, n-1)
    x_offsets = rotated_Pts[:, 0] - ((n - 1 - x_indices) / (n - 1) * x_range + x_min)
    
    y_indices = np.clip(y_indices, 0, m - 1)
    y_offsets = rotated_Pts[:, 1] - (y_indices / (m - 1) * y_range + y_min)
    
    # Update mX, mY, mD
    mX[y_indices, x_indices] = x_offsets
    mY[y_indices, x_indices] = y_offsets
    mD[y_indices, x_indices] = rotated_Pts[:, 2]
    
    # Initialize sliding window
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
            # Calculate sum of squared distances to prevent division by zero
            dist_sq = KmX[i, j]**2 + KmY[i, j]**2 + 1e-6
            s = 1 / dist_sq  # Weight: inverse square of distance
            # Only consider valid depth values
            valid_mask = ~np.isnan(KmD[i, j])
            s = s * valid_mask
            Y += s * KmD[i, j]
            S += s
    
    # Avoid division by zero
    depth_values = Y / (S + 1e-6)
    
    # Expand the result back to the original size array
    out = np.zeros((m, n), dtype=np.float32)
    out[grid : m - grid, grid : n - grid] = depth_values
    
    # Handle invalid values, fill with minimum depth value
    valid_out = out[np.isfinite(out)]
    if valid_out.size > 0:
        min_depth = valid_out.min()
        max_depth = valid_out.max()
        out = np.nan_to_num(out, nan=min_depth)
        # Normalize depth values to 0-255, higher points have higher values
        depth_normalized = (out - min_depth) / (max_depth - min_depth + 1e-6) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
    else:
        depth_normalized = np.zeros_like(out, dtype=np.uint8)

    # colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    # # colored_depth_map = cv2.resize(colored_depth_map, (384, 384), cv2.INTER_NEAREST)
    
    # depth_vis = cv2.convertScaleAbs(depth_normalized, alpha=255.0 / depth_normalized.max())
    # # depth_vis = cv2.resize(depth_vis, (384, 384), cv2.INTER_NEAREST)
    # cv2.imwrite('test_depth_color_grid3.png', depth_vis)
    # cv2.imwrite('test_depth_color.png', colored_depth_map)
    
    return depth_normalized

def las2ply():
    # 定义 .las 文件路径
    las_dir = r"D:/data/pix4d_visloc_1/2_densification/point_cloud/"
    las_files = [os.path.join(las_dir, f) for f in os.listdir(las_dir) if f.endswith(".las")]

    # 初始化点和颜色的数组
    all_points = []
    all_colors = []

    # 读取所有 .las 文件并提取点和颜色数据
    for las_file in las_files:
        las = laspy.read(las_file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        all_points.append(points)
        
        # 如果包含颜色信息，则提取红、绿、蓝通道
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            colors = np.vstack((las.red, las.green, las.blue)).transpose()
            
            # 有些数据会以16位存储颜色值，需要缩放到[0, 255]
            if colors.max() > 255:
                colors = (colors / colors.max() * 255).astype(np.uint8)
                
            all_colors.append(colors)
        else:
            # 若无颜色信息，默认白色
            all_colors.append(np.full((points.shape[0], 3), 255))

    # 将所有点和颜色合并
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # 创建 Open3D 的点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(all_points)
    point_cloud.colors = o3d.utility.Vector3dVector(all_colors / 255.0)  # 颜色值归一化到[0, 1]

    # 保存为 .ply 文件
    o3d.io.write_point_cloud(las_dir + "combined_point_cloud_with_color.ply", point_cloud)

    print("所有 .las 文件已成功合并并包含颜色信息，保存为 combined_point_cloud_with_color.ply 文件")


if __name__ == '__main__':
    # las2ply()

    # camera_info_file = 'D:\\data\\UAV_VisLoc_dataset\\02\\calibrated_external_camera_parameters.txt'
    # point_cloud_file = 'D:\\data\\UAV_VisLoc_dataset\\02\\visloc_2_point_cloud.ply'
    # output_folder = 'D:\\data\\UAV_VisLoc_dataset\\02\\lidar'
    # offset_file = 'D:\\data\\UAV_VisLoc_dataset\\02\\visloc_2_offset.xyz'

    # camera_info = read_camera_info(camera_info_file)
    # process_point_cloud_in_chunks(camera_info, point_cloud_file, output_folder, offset_file, chunk_size=50000, h_fov=52, v_fov=36)

    # root_dir = r"D:/data/UAV_VisLoc_dataset/02/"
    # lidar2rgbd(root_dir)

    # pcd = o3d.io.read_point_cloud('D:\\data\\UAV_VisLoc_dataset\\03\\lidar\\03_0010.ply')
    pcd = o3d.io.read_point_cloud(r'./03_0010-sample.ply')
    points = np.asarray(pcd.points)

    points[:, 2] = -points[:, 2]
    points[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
    
    dense_top_down_map_yaw_conv_gpu(points, yaw_angle=-44.412090, grid=5, visualize=True)

