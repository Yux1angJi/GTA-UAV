### Copy from Uni3D and lidar-image-pretrain-VPR

import numpy as np
from PIL import Image


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
    print(depth_map[0][0])
    # 将深度值归一化到 [0, 255]
    non_zero_depth = depth_map[depth_map != 0]
    if len(non_zero_depth) > 0:
        max_height = np.percentile(non_zero_depth, 99)
        min_height = np.percentile(non_zero_depth, 1)
        depth_map_clipped = np.clip(depth_map, min_height, max_height)
        print(depth_map_clipped[0][0])
        depth_map_normalized = (depth_map_clipped - min_height) / (max_height - min_height)
        print(depth_map_normalized[0][0])
        print(min_height, max_height)
        depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
    else:
        depth_map_uint8 = depth_map.astype(np.uint8)
    
    
    colored_depth_map = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_JET)

    return colored_depth_map 



def random_rotate_z(pc):
    # random roate around z axis
    theta = np.random.uniform(0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    return np.matmul(pc, R)

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def augment_pc(data):
    data = random_point_dropout(data[None, ...])
    data = random_scale_point_cloud(data)
    data = shift_point_cloud(data)
    data = rotate_perturbation_point_cloud(data)
    data = data.squeeze()
    return data


if __name__ == '__main__':
    import open3d as o3d
    import cv2
    points = o3d.io.read_point_cloud('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/lidars/200_0001_0000006317.ply')
    points = np.array(points.points, dtype=np.float32)

    points[:, [0, 2]] = points[:, [2, 0]]  # 交换第0列(x)和第2列(z)
    points[:, [0, 1]] = points[:, [1, 0]]  # 交换第0列(x)和第2列(z)
    points[:, 0] = -points[:, 0]
    points[:, 1] = points[:, 1]
    points[:, 2] = 10000-points[:, 2]

    image = create_top_down_depth_image(points)
    cv2.imwrite('test_lidar_range.png', image)