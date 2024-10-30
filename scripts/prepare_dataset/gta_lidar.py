import os
import cv2
import numpy as np
import copy
from tqdm import tqdm
import time
import random
import shutil
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.validation import make_valid
import concurrent.futures
import itertools
import pickle
import json
import re
import math
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


GAME_TO_SATE_KX = 1.8206
GAME_TO_SATE_BX = 7539.39
GAME_TO_SATE_KY = -1.8220
GAME_TO_SATE_BY = 15287.16
SATE_LENGTH = 24576
TILE_LENGTH = 256

THRESHOLD = 0.39
SEMI_THRESHOLD = 0.14


def euler_to_rotation_matrix(pitch, roll, yaw):
    # Convert angles from degrees to radians
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    yaw = np.radians(yaw)
    
    # Rotation matrix around x-axis (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    
    # Rotation matrix around y-axis (roll)
    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])
    
    # Rotation matrix around z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R


def calculate_projection_points(height, rot_x, rot_y, rot_z, temp_x, temp_y, hfov=74, vfov=59):

    # rot_x, rot_y, rot_z represents pitch, roll, and yaw in eular system

    # Convert angles from degrees to radians
    hfov_rad = math.radians(hfov)
    vfov_rad = math.radians(vfov)
    rot_x = abs(rot_x + 90)
    tilt_angle_rad = math.radians(rot_x)

    # print(hfov_rad, vfov_rad, tilt_angle_rad)
    
    # Calculate the width and length of the projection on the ground
    W = 2 * height * math.tan(hfov_rad / 2)
    L = 2 * height * math.tan(vfov_rad / 2)
    
    # Calculate the shift in the projection center due to the tilt angle
    D = height * math.tan(tilt_angle_rad)
    
    # Calculate the four corner points
    P1 = (-W / 2, -L / 2 + D)
    P2 = (W / 2, -L / 2 + D)
    P3 = (-W / 2, L / 2 + D)
    P4 = (W / 2, L / 2 + D)
    relative_points = [
        [-W / 2, -L / 2 + D, 0],
        [W / 2, -L / 2 + D, 0],
        [-W / 2, L / 2 + D, 0],
        [W / 2, L / 2 + D, 0]
    ]
    # print(relative_points)

    R = euler_to_rotation_matrix(pitch=rot_x, roll=rot_y, yaw=rot_z)

    actual_points = []
    for point in relative_points:
        rotated_point = R @ np.array(point)
        actual_x = temp_x + rotated_point[0]
        actual_y = temp_y + rotated_point[1]
        actual_points.append(actual_x)
        actual_points.append(actual_y)

    return actual_points

def correct_proj_points():
    meta_data_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/drone/meta_data_bk'
    meta_data_correct_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/drone/meta_data'

    for filename in os.listdir(meta_data_dir):
        with open(os.path.join(meta_data_dir, filename), 'r') as file:
            line = file.readline().strip()
            values = line.split()

            height = float(values[3])    
            cam_roll = float(values[7])
            cam_pitch = float(values[8])
            cam_yaw = float(values[9])
            cam_pos_x = float(values[0])
            cam_pos_y = float(values[1])
            proj_points = calculate_projection_points(height, cam_roll, cam_pitch, cam_yaw, cam_pos_x, cam_pos_y)
            meta_correct_text = f"{cam_pos_x} {cam_pos_y} {values[2]} {height - 10} {cam_pos_x} {cam_pos_y} {values[6]} {values[7]} {values[8]} {values[9]} {proj_points[0]} {proj_points[1]} {proj_points[2]} {proj_points[3]} {proj_points[4]} {proj_points[5]} {proj_points[6]} {proj_points[7]}"
            with open(os.path.join(meta_data_correct_dir, filename), 'w') as file_correct:
                file_correct.write(meta_correct_text)


def sate2loc(tile_zoom, tile_x, tile_y, offset):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_x = (tile_pix * (tile_x+1/2+offset/TILE_LENGTH)) * 0.45
    loc_y = (tile_pix * (tile_y+1/2+offset/TILE_LENGTH)) * 0.45
    return loc_x, loc_y


def order_points(points):
    hull = ConvexHull(points)
    ordered_points = [points[i] for i in hull.vertices]
    return ordered_points


def calc_intersect_area(poly1, poly2):
    # 计算交集
    intersection = poly1.intersection(poly2)
    return intersection.area


def game_pos2sate_pos(game_pos_x, game_pos_y):
    sate_pos_x = game_pos_x * GAME_TO_SATE_KX + GAME_TO_SATE_BX
    sate_pos_y = game_pos_y * GAME_TO_SATE_KY + GAME_TO_SATE_BY
    return sate_pos_x, sate_pos_y


def game_pos2tile_pos(game_pos_x, game_pos_y, zoom_list):
    sate_pos_x = game_pos_x * GAME_TO_SATE_KX + GAME_TO_SATE_BX
    sate_pos_y = game_pos_y * GAME_TO_SATE_KY + GAME_TO_SATE_BY
    tile_xy_list = []
    for zoom_level in zoom_list:
        tile_length = SATE_LENGTH // int(2 ** zoom_level)
        tile_x = int(sate_pos_x + tile_length - 1) // tile_length - 1
        tile_y = int(sate_pos_y + tile_length - 1) // tile_length - 1
        # print(tile_x, tile_y)
        tile_xy_list.append((tile_x, tile_y, zoom_level))
    return tile_xy_list


def tile_expand(tile_xy_list, p_xy_list, offset_list, debug=False):
    tile_expand_list_iou = []
    tile_expand_list_semi_iou = []
    tile_expand_list_oc = []
    for tile_x, tile_y, zoom_level in tile_xy_list:
        tile_length = SATE_LENGTH // (2 ** zoom_level)
        tile_size = tile_length ** 2
        tile_max_num = 2 ** zoom_level

        p_xy_list_order = order_points(p_xy_list)
        poly_p = Polygon(p_xy_list_order)
        poly_p_area = poly_p.area


        if debug:
            print(tile_x, tile_y, iou, oc)

        tile_l = max(0, tile_x - 6)
        tile_r = min(tile_x + 6, tile_max_num)
        tile_u = max(0, tile_y - 6)
        tile_d = min(tile_y + 6, tile_max_num)

        # Enumerate all LRUD
        for tile_x_i in range(tile_l, tile_r + 1):
            for tile_y_i in range(tile_u, tile_d + 1):
                for offset in offset_list:
                    tile_tmp = [((tile_x_i    ) * tile_length + offset, (tile_y_i    ) * tile_length + offset), 
                                ((tile_x_i + 1) * tile_length + offset, (tile_y_i    ) * tile_length + offset), 
                                ((tile_x_i    ) * tile_length + offset, (tile_y_i + 1) * tile_length + offset), 
                                ((tile_x_i + 1) * tile_length + offset, (tile_y_i + 1) * tile_length + offset)]
                    tile_tmp_order = order_points(tile_tmp)
                    poly_tile = Polygon(tile_tmp_order)
                    poly_tile_area = poly_tile.area
                    intersect_area = calc_intersect_area(poly_p, poly_tile)

                    oc = intersect_area / min(poly_tile_area, poly_p_area)
                    iou = intersect_area / (poly_p_area + poly_tile_area - intersect_area)
                    loc_xy = sate2loc(zoom_level, tile_x_i, tile_y_i, offset)
                    if iou > THRESHOLD:
                        tile_expand_list_iou.append((tile_x_i, tile_y_i, zoom_level, offset, iou, loc_xy))
                    if iou > SEMI_THRESHOLD:
                        tile_expand_list_semi_iou.append((tile_x_i, tile_y_i, zoom_level, offset, iou, loc_xy))
                    # if oc > THRESHOLD:
                    #     tile_expand_list_oc.append((tile_x_i, tile_y_i, zoom_level, iou, loc_xy))

                    # if debug:
                        # print('enumerate')
                        # print(tile_x_i, tile_y_i, intersect_area, tile_size, intersect_area / tile_size, poly_p_area, intersect_area/poly_p_area)

        # if debug:
        #     print('jyx tile lrud', zoom_level, tile_l, tile_r, tile_u, tile_d)
        #     print(tile_expand_list)

    return tile_expand_list_iou, tile_expand_list_semi_iou


def process_per_drone_image(file_data):
    img_file, dir_img, dir_lidar, dir_meta, dir_satellite, root, save_root, zoom_list, offset_list = file_data

    meta_file_path = os.path.join(dir_meta, img_file.replace('.png', '.txt'))
    #### meta_data format
    #### loc_x, loc_y, loc_z, heightAboveGround, cam_x, cam_y, cam_z, cam_rot_x, cam_rot_y, cam_rot_z, P1_x, P1_y, P2_x, P2_y, P3_x, P3_y, P4_x, P4_y, time_hours, time_min, time_sec
    with open(meta_file_path, 'r') as file:
        meta_data = file.readline().strip().split()
        meta_data = [float(x) for x in meta_data]

    p_xy_game_list = meta_data[10:18]
    p_xy_sate_list = [
        game_pos2sate_pos(p_xy_game_list[0], p_xy_game_list[1]),
        game_pos2sate_pos(p_xy_game_list[2], p_xy_game_list[3]),
        game_pos2sate_pos(p_xy_game_list[4], p_xy_game_list[5]),
        game_pos2sate_pos(p_xy_game_list[6], p_xy_game_list[7])
    ]
    cam_x, cam_y = meta_data[4:6]
    tile_xy_list = game_pos2tile_pos(cam_x, cam_y, zoom_list)

    p_x_sate_mid = (p_xy_sate_list[0][0] + p_xy_sate_list[1][0] + p_xy_sate_list[2][0] + p_xy_sate_list[3][0]) / 4
    p_y_sate_mid = (p_xy_sate_list[0][1] + p_xy_sate_list[1][1] + p_xy_sate_list[2][1] + p_xy_sate_list[3][1]) / 4

    debug = False
    # debug = False
    # if not debug:  
    tile_expand_list_iou, tile_expand_list_semi_iou = tile_expand(tile_xy_list, p_xy_sate_list, offset_list, debug)

    if len(tile_expand_list_semi_iou) == 0:
        return None

    # save_drone_dir = os.path.join(save_root, 'drone', ids)
    # save_sate_dir = os.path.join(save_root, 'satellite', ids)

    drone_img = os.path.join(dir_img, img_file)
    lidar_file = img_file.replace('.png', '.ply')
    h = int(img_file.split('_')[0])
    result = {
        "h": h,
        "drone_img_dir": dir_img,
        "drone_img": img_file,
        "drone_lidar_dir": dir_lidar,
        "drone_lidar": lidar_file,
        "drone_loc_x_y": (p_x_sate_mid * 0.45, p_y_sate_mid * 0.45),
        "sate_img_dir": dir_satellite,
        "pair_iou_sate_img_list": [],
        "pair_iou_sate_weight_list": [],
        "pair_iou_sate_loc_xy_list": [],
        "pair_semi_iou_sate_img_list": [],
        "pair_semi_iou_sate_weight_list": [],
        "pair_semi_iou_sate_loc_xy_list": [],
    }
    for tile_x, tile_y, zoom_level, offset, weight, loc_xy in tile_expand_list_iou:
        # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_iou_sate_img_list"].append(f'{zoom_level}_{offset}_{tile_x}_{tile_y}.png')
        result["pair_iou_sate_weight_list"].append(weight)
        result["pair_iou_sate_loc_xy_list"].append(loc_xy)
    for tile_x, tile_y, zoom_level, offset, weight, loc_xy in tile_expand_list_semi_iou:
        # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_semi_iou_sate_img_list"].append(f'{zoom_level}_{offset}_{tile_x}_{tile_y}.png')
        result["pair_semi_iou_sate_weight_list"].append(weight)
        result["pair_semi_iou_sate_loc_xy_list"].append(loc_xy)
    # for tile_x, tile_y, zoom_level, weight, loc_xy in tile_expand_list_oc:
    #     # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
    #     # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
    #     # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
    #     result["pair_oc_sate_img_list"].append(f'{zoom_level}_{tile_x}_{tile_y}.png')
    #     result["pair_oc_sate_weight_list"].append(weight)
    #     result["pair_oc_sate_loc_xy_list"].append(loc_xy)

    if debug:
        print(p_xy_sate_list)
        print(drone_img)
        print('cam_pos', cam_x, cam_y)
        print('cam_pos sate', game_pos2sate_pos(cam_x, cam_y))
        print('tile_expand_list_iou', tile_expand_list_iou)
    return result


def save_pairs_meta_data(pairs_drone2sate_list, pkl_save_path, pair_save_dir):
    pairs_iou_sate2drone_dict = {}
    pairs_iou_drone2sate_dict = {}
    pairs_semi_iou_sate2drone_dict = {}
    pairs_semi_iou_drone2sate_dict = {}
    pairs_oc_sate2drone_dict = {}
    pairs_oc_drone2sate_dict = {}

    drone_save_dir = os.path.join(pair_save_dir, 'drone')
    sate_iou_save_dir = os.path.join(pair_save_dir, 'satellite', 'iou')
    sate_semi_iou_save_dir = os.path.join(pair_save_dir, 'satellite', 'semi_iou')
    sate_oc_save_dir = os.path.join(pair_save_dir, 'satellite', 'oc')
    os.makedirs(drone_save_dir, exist_ok=True)
    os.makedirs(sate_iou_save_dir, exist_ok=True)
    os.makedirs(sate_semi_iou_save_dir, exist_ok=True)
    os.makedirs(sate_oc_save_dir, exist_ok=True)

    pairs_sate2drone_save = []
    for pairs_drone2sate in pairs_drone2sate_list:
        if pairs_drone2sate == None:
            continue
        pairs_sate2drone_save.append(pairs_drone2sate)
        h = pairs_drone2sate["h"]
        pair_iou_sate_img_list = pairs_drone2sate["pair_iou_sate_img_list"]
        pair_semi_iou_sate_img_list = pairs_drone2sate["pair_semi_iou_sate_img_list"]
        # pair_oc_sate_img_list = pairs_drone2sate["pair_oc_sate_img_list"]
        drone_img = pairs_drone2sate["drone_img"]
        drone_img_dir = pairs_drone2sate["drone_img_dir"]
        sate_img_dir = pairs_drone2sate["sate_img_dir"]

        drone_img_name = drone_img.replace('.png', '')
        drone_img_name = f'{drone_img_name}'
        drone_save_path = os.path.join(drone_save_dir, drone_img_name)
        os.makedirs(drone_save_path, exist_ok=True)
        sate_iou_save_path = os.path.join(sate_iou_save_dir, drone_img_name)
        os.makedirs(sate_iou_save_path, exist_ok=True)
        sate_semi_iou_save_path = os.path.join(sate_semi_iou_save_dir, drone_img_name)
        os.makedirs(sate_semi_iou_save_path, exist_ok=True)
        # sate_oc_save_path = os.path.join(sate_oc_save_dir, drone_img_name)
        # os.makedirs(sate_oc_save_path, exist_ok=True)

        # shutil.copy(os.path.join(drone_img_dir, drone_img), drone_save_path)
        for sate_img in pair_iou_sate_img_list:
            pairs_iou_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
            pairs_iou_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
            # shutil.copy(os.path.join(sate_img_dir, sate_img), sate_iou_save_path)
        for sate_img in pair_semi_iou_sate_img_list:
            pairs_semi_iou_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
            pairs_semi_iou_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
            # shutil.copy(os.path.join(sate_img_dir, sate_img), sate_semi_iou_save_path)
        # for sate_img in pair_oc_sate_img_list:
        #     pairs_oc_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
        #     pairs_oc_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
        #     shutil.copy(os.path.join(sate_img_dir, sate_img), sate_oc_save_path)

    pairs_iou_match_set = set()
    for tile_img, tile2drone in pairs_iou_sate2drone_dict.items():
        pairs_iou_sate2drone_dict[tile_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_iou_drone2sate_dict.items():
        pairs_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for tile_img in pairs_iou_drone2sate_dict[drone_img]:
            pairs_iou_match_set.add((drone_img, tile_img))

    pairs_semi_iou_match_set = set()
    for tile_img, tile2drone in pairs_semi_iou_sate2drone_dict.items():
        pairs_semi_iou_sate2drone_dict[tile_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_semi_iou_drone2sate_dict.items():
        pairs_semi_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for tile_img in pairs_semi_iou_drone2sate_dict[drone_img]:
            pairs_semi_iou_match_set.add((drone_img, tile_img))

    # pairs_oc_match_set = set()
    # for tile_img, tile2drone in pairs_oc_sate2drone_dict.items():
    #     pairs_oc_sate2drone_dict[tile_img] = list(set(tile2drone))
    # for drone_img, drone2tile in pairs_oc_drone2sate_dict.items():
    #     pairs_oc_drone2sate_dict[drone_img] = list(set(drone2tile))
    #     for tile_img in pairs_oc_drone2sate_dict[drone_img]:
    #         pairs_oc_match_set.add((drone_img, tile_img))


    with open(pkl_save_path, 'wb') as f:
        pickle.dump({
            "pairs_drone2sate_list": pairs_sate2drone_save,

            "pairs_iou_sate2drone_dict": pairs_iou_sate2drone_dict,
            "pairs_iou_drone2sate_dict": pairs_iou_drone2sate_dict,
            "pairs_iou_match_set": pairs_iou_match_set,

            "pairs_semi_iou_sate2drone_dict": pairs_semi_iou_sate2drone_dict,
            "pairs_semi_iou_drone2sate_dict": pairs_semi_iou_drone2sate_dict,
            "pairs_semi_iou_match_set": pairs_semi_iou_match_set,

            # "pairs_oc_sate2drone_dict": pairs_oc_sate2drone_dict,
            # "pairs_oc_drone2sate_dict": pairs_oc_drone2sate_dict,
            # "pairs_oc_match_set": pairs_oc_match_set,
        }, f)


def process_gta_data(root, save_root, h_list=[200, 300, 400], zoom_list=[5, 6, 7], offset_list=[0], split_type='same'):

    processed_data = []
    processed_data_train = []
    processed_data_test = []

    file_data_list = []

    dir_img = os.path.join(root, 'drone', 'images')
    dir_lidar = os.path.join(root, 'drone', 'lidars')
    dir_meta = os.path.join(root, 'drone', 'meta_data')
    dir_satellite = os.path.join(root, 'satellite')
    files = [f for f in os.listdir(dir_img)]
    file_data_list.extend([(img_file, dir_img, dir_lidar, dir_meta, dir_satellite, root, save_root, zoom_list, offset_list)for img_file in files])
    file_data_list_h = []
    for file_data in file_data_list:
        if int(file_data[0].split('_')[0]) in h_list:
            file_data_list_h.append(file_data)
    random.shuffle(file_data_list_h)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_per_drone_image, file_data_list_h), total=len(file_data_list_h)):
            processed_data.append(result)
    
    processed_data_wonone = []
    for result in processed_data:
        if result is not None:
            processed_data_wonone.append(result)

    if split_type == 'same':
        processed_data_num = len(processed_data_wonone)
        processed_data_train = processed_data_wonone[:processed_data_num // 5 * 4]
        processed_data_test = processed_data_wonone[processed_data_num // 5 * 4: ]
    elif split_type == 'cross':
        processed_data_train = []
        processed_data_test = []
        for result in processed_data_wonone:
            if result['drone_loc_x_y'][0] < 7500 * 0.45:
                processed_data_train.append(result)
            else:
                processed_data_test.append(result)
    elif split_type == 'cross_test':
        processed_data_train = []
        processed_data_test = []
        for result in processed_data_wonone:
            if int(result['drone_img'].split('_')[-1].replac('.png', '')) < 2500:
                processed_data_train.append(result)
            else:
                processed_data_test.append(result)
    print(f'After spliting, train data len={len(processed_data_train)}, test data len={len(processed_data_test)}')
    
    train_pkl_save_path = os.path.join(save_root, 'train_pair_meta.pkl')
    train_data_save_dir = os.path.join(save_root, 'train')
    save_pairs_meta_data(processed_data_train, train_pkl_save_path, train_data_save_dir)

    test_pkl_save_path = os.path.join(save_root, 'test_pair_meta.pkl')
    test_data_save_dir = os.path.join(save_root, 'test')
    save_pairs_meta_data(processed_data_test, test_pkl_save_path, test_data_save_dir)

    write_json(pickle_root=save_root, root=root, split_type=split_type)


def copy_png_files(src_path, dst_path):
    # 创建目标文件夹
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for root, dirs, files in os.walk(src_path):
        for file_name in files:
            # 检查文件是否为 .png 文件
            if file_name.endswith('.png'):
                # 构建完整的文件路径
                full_file_name = os.path.join(root, file_name)
                if os.path.isfile(full_file_name):
                    # 复制文件到目标文件夹
                    shutil.copy(full_file_name, dst_path)

    print(f"所有 .png 文件已复制到 {dst_path}")


def resize_and_save_image(file, input_dir, output_dir, size):
    # 加载RGBD图像
    rgbd_path = os.path.join(input_dir, file)
    image = Image.open(rgbd_path)
    
    # 将图像resize
    resized_image = image.resize(size, Image.LANCZOS)
    
    # 保存到目标路径
    output_path = os.path.join(output_dir, file)
    resized_image.save(output_path)

def resize_rgbd_images(input_dir, output_dir, size=(960, 540), max_workers=4):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取文件列表
    files = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpg")]
    
    # 使用多进程池执行任务
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(resize_and_save_image, file, input_dir, output_dir, size)
            for file in files
        ]
        
        # 使用tqdm显示进度条
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Resizing images", unit="image"):
            pass


def move_png_files(source_dir, destination_dir):
    # 遍历源目录下的所有文件和子目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".txt"):
                # 源文件路径
                source_file_path = os.path.join(root, file)
                
                # 目标文件路径
                destination_file_path = os.path.join(destination_dir, file)
                
                # 移动文件
                shutil.move(source_file_path, destination_file_path)
                print(f"Moved {source_file_path} to {destination_file_path}")


def move_file():
    import os
    import shutil

    # 定义源目录和目标目录
    source_directory = "/home/xmuairmud/data/GTA-UAV-data/randcam2_std5/train_new/drone"
    target_directory = "/home/xmuairmud/data/GTA-UAV-data/randcam2_std5/test_new/drone"

    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历范围为25到32的所有目录
    for x in range(20, 33):
        for y in range(20, 33):
            directory_name = f"{x}_{y}"
            source_path = os.path.join(source_directory, directory_name)
            # 检查目录是否存在
            if os.path.exists(source_path) and os.path.isdir(source_path):
                # 移动目录到目标目录
                shutil.move(source_path, target_directory)
                print(f"Moved: {source_path} to {target_directory}")

    print("All directories moved successfully.")


def rename_tile():
    directory = '/home/xmuairmud/data/GTA-UAV-data/satellite_overlap'  # 替换为你的目录路径

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            match = re.match(r'(\d+)_(\d+)_(\d+)\.png', filename)
            if match:
                new_filename = f"{match.group(1)}_0_{match.group(2)}_{match.group(3)}.png"
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

    print("文件重命名完成！")


def write_json(pickle_root, root, split_type):
    for type in ['train', 'test']:
        data_drone2sate_json = []
        with open(os.path.join(pickle_root, f'{type}_pair_meta.pkl'), 'rb') as f:
            data_pickle = pickle.load(f)
        for pair_drone2sate in data_pickle['pairs_drone2sate_list']:
            img_name = pair_drone2sate['drone_img']
            meta_file = os.path.join(root, 'drone/meta_data', img_name.replace('.png', '.txt'))
            with open(meta_file, 'r') as file:
                line = file.readline().strip()
                values = line.split()
                if len(values) >= 4:
                    height = float(values[3])           
                    cam_roll = float(values[7])
                    cam_pitch = float(values[8])
                    cam_yaw = float(values[9])
                    drone_roll = cam_roll + 90.0
                    drone_pitch = cam_pitch
                    drone_yaw = cam_yaw
            data_drone2sate_json.append({
                "drone_img_dir": "drone/images",
                "drone_img_name": pair_drone2sate['drone_img'],
                "drone_lidar_dir": "drone/lidars",
                "drone_lidar_name": pair_drone2sate['drone_lidar'],
                "drone_loc_x_y": pair_drone2sate['drone_loc_x_y'],
                "sate_img_dir": "satellite",
                "pair_pos_sate_img_list": pair_drone2sate['pair_iou_sate_img_list'],
                "pair_pos_sate_weight_list": pair_drone2sate['pair_iou_sate_weight_list'],
                "pair_pos_sate_loc_x_y_list": pair_drone2sate['pair_iou_sate_loc_xy_list'],
                "pair_pos_semipos_sate_img_list": pair_drone2sate['pair_semi_iou_sate_img_list'],
                "pair_pos_semipos_sate_weight_list": pair_drone2sate['pair_semi_iou_sate_weight_list'],
                "pair_pos_semipos_sate_loc_x_y_list": pair_drone2sate['pair_semi_iou_sate_loc_xy_list'],
                "drone_metadata": {
                    "height": height,
                    "drone_roll": drone_roll,
                    "drone_pitch": drone_pitch,
                    "drone_yaw": drone_yaw,
                    "cam_roll": cam_roll,
                    "cam_pitch": cam_pitch,
                    "cam_yaw": cam_yaw,
                }
            })
        save_path = os.path.join(root, f'{split_type}-area-drone2sate-{type}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_drone2sate_json, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # correct_proj_points()
    # rename_tile()

    # root = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar'
    # save_root = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar/same_area'
    # process_gta_data(root, save_root, h_list=[100, 200, 300, 400, 500, 600], zoom_list=[4, 5, 6, 7], offset_list=[0], split_type='same')

    input_dir = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar/drone/images'
    output_dir = '/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-Lidar/GTA-UAV-Lidar-LR/drone/images'
    resize_rgbd_images(input_dir, output_dir, size=(960, 540))

    # write_json(pickle_root=save_root, root=root, split_type='same')

    # src_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/randcam2_std0_stable_all_resize'
    # dst_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/drone/meta_data'
    # move_png_files(src_dir, dst_dir)

    # src_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/satellite'
    # dst_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/train_h23456/all_satellite'
    # copy_png_files(src_path, dst_path)
