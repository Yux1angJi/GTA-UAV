import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
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


GAME_TO_SATE_KX = 1.8206
GAME_TO_SATE_BX = 7539.39
GAME_TO_SATE_KY = -1.8220
GAME_TO_SATE_BY = 15287.16
SATE_LENGTH = 24576

THRESHOLD = 0.39
SEMI_THRESHOLD = 0.14

SATE_LENGTH = 24576

NORM_LOC = 10000.


def sate2loc(tile_zoom, tile_x, tile_y):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_x = (tile_pix * (tile_x+1/2)) * 0.45
    loc_y = (tile_pix * (tile_y+1/2)) * 0.45
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


def tile_expand(tile_xy_list, p_xy_list, debug=False):
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
        
        tile_tmp = [((tile_x    ) * tile_length, (tile_y    ) * tile_length), 
                    ((tile_x + 1) * tile_length, (tile_y    ) * tile_length), 
                    ((tile_x    ) * tile_length, (tile_y + 1) * tile_length), 
                    ((tile_x + 1) * tile_length, (tile_y + 1) * tile_length)]
        tile_tmp_order = order_points(tile_tmp)
        poly_tile = Polygon(tile_tmp_order)
        poly_tile_area = poly_tile.area
        intersect_area = calc_intersect_area(poly_p, poly_tile)
        oc = intersect_area / min(poly_tile_area, poly_p_area)
        iou = intersect_area / (poly_p_area + poly_tile_area - intersect_area)

        if debug:
            print(tile_x, tile_y, iou, oc)

        tile_l = max(0, tile_x - 5)
        tile_r = min(tile_x + 5, tile_max_num)
        tile_u = max(0, tile_y - 5)
        tile_d = min(tile_y + 5, tile_max_num)
        
        # # To Left
        # while tile_l >= 1:
        #     tile_tmp = [((tile_l - 1) * tile_length, (tile_y - 1) * tile_length), 
        #                 ((tile_l    ) * tile_length, (tile_y - 1) * tile_length), 
        #                 ((tile_l - 1) * tile_length, (tile_y    ) * tile_length), 
        #                 ((tile_l    ) * tile_length, (tile_y    ) * tile_length)]
        #     intersect_area = calc_intersect_area(p_xy_list, tile_tmp)
        #     if debug:
        #         print('To Left')
        #         print("tile_xy", tile_l, tile_y)
        #         print("tile_tmp", tile_tmp)
        #         print("intersect_area / tile_size", intersect_area, tile_size, intersect_area/tile_size)
        #     tile_l -= 1
        #     if intersect_area / tile_size < 0.4:
        #         break
        # tile_l += 1

        # Enumerate all LRUD
        for tile_x_i in range(tile_l, tile_r + 1):
            for tile_y_i in range(tile_u, tile_d + 1):
                tile_tmp = [((tile_x_i    ) * tile_length, (tile_y_i    ) * tile_length), 
                            ((tile_x_i + 1) * tile_length, (tile_y_i    ) * tile_length), 
                            ((tile_x_i    ) * tile_length, (tile_y_i + 1) * tile_length), 
                            ((tile_x_i + 1) * tile_length, (tile_y_i + 1) * tile_length)]
                tile_tmp_order = order_points(tile_tmp)
                poly_tile = Polygon(tile_tmp_order)
                poly_tile_area = poly_tile.area
                intersect_area = calc_intersect_area(poly_p, poly_tile)

                # max_rate = max(intersect_area / tile_size, intersect_area / poly_p_area)
                # if max_rate > 0.4:
                #     # print('jyxjyx', intersect_area / tile_size, intersect_area / poly_p_area)
                #     tile_expand_list.append((tile_x_i, tile_y_i, zoom_level, max_rate))
                oc = intersect_area / min(poly_tile_area, poly_p_area)
                iou = intersect_area / (poly_p_area + poly_tile_area - intersect_area)
                loc_xy = sate2loc(zoom_level, tile_x_i, tile_y_i)
                if iou > THRESHOLD:
                    tile_expand_list_iou.append((tile_x_i, tile_y_i, zoom_level, iou, loc_xy))
                if iou > SEMI_THRESHOLD:
                    tile_expand_list_semi_iou.append((tile_x_i, tile_y_i, zoom_level, iou, loc_xy))
                if oc > THRESHOLD:
                    tile_expand_list_oc.append((tile_x_i, tile_y_i, zoom_level, iou, loc_xy))

                # if debug:
                    # print('enumerate')
                    # print(tile_x_i, tile_y_i, intersect_area, tile_size, intersect_area / tile_size, poly_p_area, intersect_area/poly_p_area)

        # if debug:
        #     print('jyx tile lrud', zoom_level, tile_l, tile_r, tile_u, tile_d)
        #     print(tile_expand_list)

    return tile_expand_list_iou, tile_expand_list_semi_iou, tile_expand_list_oc


def process_per_drone_image(file_data):
    img_file, dir_img, dir_meta, dir_satellite, root, save_root, zoom_list = file_data

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
    tile_expand_list_iou, tile_expand_list_semi_iou, tile_expand_list_oc = tile_expand(tile_xy_list, p_xy_sate_list, debug)

    if len(tile_expand_list_semi_iou) == 0:
        return None

    # save_drone_dir = os.path.join(save_root, 'drone', ids)
    # save_sate_dir = os.path.join(save_root, 'satellite', ids)

    drone_img = os.path.join(dir_img, img_file)
    h = int(img_file.split('_')[0])
    result = {
        "h": h,
        "drone_img_dir": dir_img,
        "drone_img": img_file,
        "drone_loc_x_y": (p_x_sate_mid * 0.45, p_y_sate_mid * 0.45),
        "sate_img_dir": dir_satellite,
        "pair_iou_sate_img_list": [],
        "pair_iou_sate_weight_list": [],
        "pair_iou_sate_loc_xy_list": [],
        "pair_semi_iou_sate_img_list": [],
        "pair_semi_iou_sate_weight_list": [],
        "pair_semi_iou_sate_loc_xy_list": [],
        "pair_oc_sate_img_list": [],
        "pair_oc_sate_weight_list": [],
        "pair_oc_sate_loc_xy_list": [],
    }
    for tile_x, tile_y, zoom_level, weight, loc_xy in tile_expand_list_iou:
        # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_iou_sate_img_list"].append(f'{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_iou_sate_weight_list"].append(weight)
        result["pair_iou_sate_loc_xy_list"].append(loc_xy)
    for tile_x, tile_y, zoom_level, weight, loc_xy in tile_expand_list_semi_iou:
        # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_semi_iou_sate_img_list"].append(f'{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_semi_iou_sate_weight_list"].append(weight)
        result["pair_semi_iou_sate_loc_xy_list"].append(loc_xy)
    for tile_x, tile_y, zoom_level, weight, loc_xy in tile_expand_list_oc:
        # tile_img = os.path.join(dir_satellite, f'level_{zoom_level}/{zoom_level}_{tile_x}_{tile_y}.png')
        # save_drone_img = os.path.join(save_drone_dir, f'{h}_{img_file}')
        # save_sate_img = os.path.join(save_sate_dir, f'{h}_{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_oc_sate_img_list"].append(f'{zoom_level}_{tile_x}_{tile_y}.png')
        result["pair_oc_sate_weight_list"].append(weight)
        result["pair_oc_sate_loc_xy_list"].append(loc_xy)

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
        pair_oc_sate_img_list = pairs_drone2sate["pair_oc_sate_img_list"]
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
        sate_oc_save_path = os.path.join(sate_oc_save_dir, drone_img_name)
        os.makedirs(sate_oc_save_path, exist_ok=True)

        shutil.copy(os.path.join(drone_img_dir, drone_img), drone_save_path)
        for sate_img in pair_iou_sate_img_list:
            pairs_iou_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
            pairs_iou_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
            shutil.copy(os.path.join(sate_img_dir, sate_img), sate_iou_save_path)
        for sate_img in pair_semi_iou_sate_img_list:
            pairs_semi_iou_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
            pairs_semi_iou_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
            shutil.copy(os.path.join(sate_img_dir, sate_img), sate_semi_iou_save_path)
        for sate_img in pair_oc_sate_img_list:
            pairs_oc_drone2sate_dict.setdefault(drone_img, []).append(sate_img)
            pairs_oc_sate2drone_dict.setdefault(sate_img, []).append(drone_img)
            shutil.copy(os.path.join(sate_img_dir, sate_img), sate_oc_save_path)

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

    pairs_oc_match_set = set()
    for tile_img, tile2drone in pairs_oc_sate2drone_dict.items():
        pairs_oc_sate2drone_dict[tile_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_oc_drone2sate_dict.items():
        pairs_oc_drone2sate_dict[drone_img] = list(set(drone2tile))
        for tile_img in pairs_oc_drone2sate_dict[drone_img]:
            pairs_oc_match_set.add((drone_img, tile_img))


    with open(pkl_save_path, 'wb') as f:
        pickle.dump({
            "pairs_drone2sate_list": pairs_sate2drone_save,

            "pairs_iou_sate2drone_dict": pairs_iou_sate2drone_dict,
            "pairs_iou_drone2sate_dict": pairs_iou_drone2sate_dict,
            "pairs_iou_match_set": pairs_iou_match_set,

            "pairs_semi_iou_sate2drone_dict": pairs_semi_iou_sate2drone_dict,
            "pairs_semi_iou_drone2sate_dict": pairs_semi_iou_drone2sate_dict,
            "pairs_semi_iou_match_set": pairs_semi_iou_match_set,

            "pairs_oc_sate2drone_dict": pairs_oc_sate2drone_dict,
            "pairs_oc_drone2sate_dict": pairs_oc_drone2sate_dict,
            "pairs_oc_match_set": pairs_oc_match_set,
        }, f)


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



def process_gta_data(root, save_root, h_list=[200, 300, 400], zoom_list=[5, 6, 7], split_type='same'):

    processed_data = []
    processed_data_train = []
    processed_data_test = []

    file_data_list = []


    dir_img = os.path.join(root, 'drone', 'images')
    dir_meta = os.path.join(root, 'drone', 'meta_data')
    dir_satellite = os.path.join(root, 'satellite')
    files = [f for f in os.listdir(dir_img)]
    file_data_list.extend([(img_file, dir_img, dir_meta, dir_satellite, root, save_root, zoom_list)for img_file in files])
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
    print(f'After spliting, train data len={len(processed_data_train)}, test data len={len(processed_data_test)}')
    
    train_pkl_save_path = os.path.join(save_root, 'train_pair_meta.pkl')
    train_data_save_dir = os.path.join(save_root, 'train')
    save_pairs_meta_data(processed_data_train, train_pkl_save_path, train_data_save_dir)

    test_pkl_save_path = os.path.join(save_root, 'test_pair_meta.pkl')
    test_data_save_dir = os.path.join(save_root, 'test')
    save_pairs_meta_data(processed_data_test, test_pkl_save_path, test_data_save_dir)


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


def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
    return data


def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


def write_json():
    data_drone2sate_json = []
    with open('/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/same_h123456_z4567/train_pair_meta.pkl', 'rb') as f:
        data_pickle = pickle.load(f)
    for pair_drone2sate in data_pickle['pairs_drone2sate_list']:
        img_name = pair_drone2sate['drone_img']
        meta_file = os.path.join('/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/drone/meta_data', img_name.replace('.png', '.txt'))
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
    save_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/same-area-drone2sate-train.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_drone2sate_json, f, indent=4, ensure_ascii=False)



class GTADatasetTrain(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 group_len=2):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        self.group_len = group_len

        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            # Training with Positive-only data or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))

            # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate['pair_pos_semipos_sate_img_list']
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        query_img_path, gallery_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight    # , query_loc_xy/NORM_LOC, gallery_loc_xy/NORM_LOC

    def __len__(self):
        return len(self.samples)

    def shuffle_group(self, ):
        '''
        Implementation of Mutually Exclusive Sampling process with group
        '''
        print("\nShuffle Dataset in Batches:")
        
        pair_pool = copy.deepcopy(self.pairs)
        # Shuffle pairs order
        random.shuffle(pair_pool)
        
        sate_batch = set()
        drone_batch = set()
        
        # Lookup if already used in epoch
        pairs_epoch = set()   

        # buckets
        batches = []
        current_batch = []
        
        # counter
        break_counter = 0

        while True:
            # pbar.update()
            # print(break_counter)
            if len(pair_pool) > 0:
                if break_counter >= 16384:
                    break

                pair = pair_pool.pop(0)
                
                drone_img_path, sate_img_path, _ = pair
                drone_img_dir = os.path.dirname(drone_img_path)
                sate_img_dir = os.path.dirname(sate_img_path)

                drone_img_name_i = drone_img_path.split('/')[-1]
                sate_img_name_i = sate_img_path.split('/')[-1]

                pair_name = (drone_img_name_i, sate_img_name_i)

                if drone_img_name_i in drone_batch or pair_name in pairs_epoch:
                    if pair_name not in pairs_epoch:
                            pair_pool.append(pair)
                    break_counter += 1
                    continue

                pairs_drone2sate = self.pairs_drone2sate_dict[drone_img_name_i]
                random.shuffle(pairs_drone2sate)

                subset_sate_len = itertools.combinations(pairs_drone2sate, self.group_len)
                
                subset_drone = None
                subset_sate = None
                for subset_sate_i in subset_sate_len:
                    flag = True
                    sate2drone_inter_set = None

                    #### Check for sate
                    for sate_img in subset_sate_i:
                        if sate_img in sate_batch:
                            flag = False
                            break
                        
                        if sate2drone_inter_set == None:
                            sate2drone_inter_set = set(self.pairs_sate2drone_dict[sate_img])
                        else:
                            sate2drone_inter_set = sate2drone_inter_set.intersection(self.pairs_sate2drone_dict[sate_img])
                    
                        
                    if not flag or sate2drone_inter_set == None or len(sate2drone_inter_set) < self.group_len:
                        continue

                    sate2drone_inter_set = list(sate2drone_inter_set)
                    random.shuffle(sate2drone_inter_set)
                    subset_drone_len = itertools.combinations(sate2drone_inter_set, self.group_len)
                    #### Check for drone
                    for subset_drone_i in subset_drone_len:
                        if drone_img_name_i not in subset_drone_i:
                            continue
                        flag = True
                        for drone_img in subset_drone_i:
                            if drone_img in drone_batch or flag == False:
                                flag = False
                                break
                            for sate_img in subset_sate_i:
                                pair_tmp = (drone_img, sate_img)
                                if pair_tmp in pairs_epoch:
                                    flag = False
                                    break
                        if flag:
                            subset_drone = subset_drone_i
                            subset_sate = subset_sate_i
                            break
                
                if subset_drone != None and subset_sate != None:
                    # random.shuffle(subset_drone)
                    # random.shuffle(subset_sate)
                    for drone_img_name, sate_img_name in zip(subset_drone, subset_sate):
                        drone_img_path = os.path.join(self.data_root, drone_img_dir, drone_img_name)
                        sate_img_path = os.path.join(self.data_root, sate_img_dir, sate_img_name)
                        current_batch.append((drone_img_path, sate_img_path, 1.0))
                        pairs_epoch.add((drone_img_name, sate_img_name))
                    for drone_img in subset_drone:
                        pairs_drone2sate = self.pairs_drone2sate_dict[drone_img_name]
                        for sate in pairs_drone2sate:
                            sate_batch.add(sate)
                    for sate_img in subset_sate:
                        pairs_sate2drone = self.pairs_sate2drone_dict[sate_img_name]
                        for drone in pairs_sate2drone:
                            drone_batch.add(drone)
                else:
                    if pair_name not in pairs_epoch:
                            pair_pool.append(pair)        
                    break_counter += 1

                if break_counter >= 16384:
                    break
            else:
                break
            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                sate_batch = set()
                drone_batch = set()
                current_batch = []
        
        self.samples = batches
        
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element: {} - Last Element: {}".format(self.samples[0][0], self.samples[-1][0]))  
        print("First Element: {} - Last Element: {}".format(self.samples[0][1], self.samples[-1][1]))  
    

    def shuffle(self, ):
        '''
        Implementation of Mutually Exclusive Sampling process
        '''
        
        print("\nShuffle Dataset:")
        
        pair_pool = copy.deepcopy(self.pairs)
            
        # Shuffle pairs order
        random.shuffle(pair_pool)
        
        sate_batch = set()
        drone_batch = set()
        
        # Lookup if already used in epoch
        pairs_epoch = set()   

        # buckets
        batches = []
        current_batch = []
            
        # counter
        break_counter = 0
        
        # progressbar
        # pbar = tqdm()

        while True:
            
            # pbar.update()
            
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                
                drone_img, sate_img, _ = pair
                drone_img_name = drone_img.split('/')[-1]
                sate_img_name = sate_img.split('/')[-1]
                # print(sate_img_name)

                pair_name = (drone_img_name, sate_img_name)

                if drone_img_name not in drone_batch and sate_img_name not in sate_batch and pair_name not in pairs_epoch:

                    current_batch.append(pair)
                    pairs_epoch.add(pair_name)
                    
                    pairs_drone2sate = self.pairs_drone2sate_dict[drone_img_name]
                    for sate in pairs_drone2sate:
                        sate_batch.add(sate)
                    pairs_sate2drone = self.pairs_sate2drone_dict[sate_img_name]
                    for drone in pairs_sate2drone:
                        drone_batch.add(drone)
                    
                    break_counter = 0
                    
                else:
                    # if pair fits not in batch and is not already used in epoch -> back to pool
                    if pair_name not in pairs_epoch:
                        pair_pool.append(pair)
                        
                    break_counter += 1
                    
                if break_counter >= 16384:
                    break
            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                sate_batch = set()
                drone_batch = set()
                current_batch = []
    
        # pbar.close()
        
        # wait before closing progress bar
        # time.sleep(0.3)
        
        self.samples = batches
        
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element: {} - Last Element: {}".format(self.samples[0][0], self.samples[-1][0]))  
        print("First Element: {} - Last Element: {}".format(self.samples[0][1], self.samples[-1][1]))  
        

class GTADatasetEval(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=None,
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)    

        self.images_path = []
        self.images_name = []
        self.images_loc_xy = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        if view == 'drone':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.images_path.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.images_name.append(drone_img_name)
                    self.images_loc_xy.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    self.images_loc_xy.append(sate2loc(tile_zoom, tile_x, tile_y))
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    self.images_loc_xy.append(sate2loc(tile_zoom, tile_x, tile_y))

        self.transforms = transforms

    def __getitem__(self, index):
        
        img_path = self.images_path[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return img

    def __len__(self):
        return len(self.images_name)
    
    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    

    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
                                

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*img_size[0]),
                                                               max_width=int(0.2*img_size[0]),
                                                               min_holes=10,
                                                               min_height=int(0.1*img_size[0]),
                                                               min_width=int(0.1*img_size[0]),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.4, p=1.0),
                                                 A.CoarseDropout(max_holes=25,
                                                                 max_height=int(0.2*img_size[0]),
                                                                 max_width=int(0.2*img_size[0]),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*img_size[0]),
                                                                 min_width=int(0.1*img_size[0]),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])
    
    return val_transforms, train_sat_transforms, train_drone_transforms


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


if __name__ == "__main__":
    write_json()

    # root = '/home/xmuairmud/data/GTA-UAV-data/randcam2_5area'
    # save_root = '/home/xmuairmud/data/GTA-UAV-data/randcam2_5area/same_h123456_z4567'
    # process_gta_data(root, save_root, h_list=[100, 200, 300, 400, 500, 600], zoom_list=[4, 5, 6, 7], split_type='same')

    # src_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/randcam2_std0_stable_all_resize'
    # dst_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/drone/meta_data'
    # move_png_files(src_dir, dst_dir)

    # src_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/satellite'
    # dst_path = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable/train_h23456/all_satellite'
    # copy_png_files(src_path, dst_path)

    # x = -100
    # y = 314

    # print(game_pos2sate_pos(x, y))
    # print(game_pos2tile_pos(x, y, [5, 6, 7]))

    # from shapely.geometry import Polygon
    # from shapely.validation import make_valid

    # p_xy_list = [(9485.994310144275, 15556.90363294211), (9693.898883986512, 16153.238791564307), (9848.754275218602, 15429.67346430581), (10056.65884906084, 16026.008622928004)]
    # tile_x = 50
    # tile_y = 81
    # tile_length = SATE_LENGTH // int(2 ** 7)
    # tile_tmp = [((tile_x    ) * tile_length, (tile_y    ) * tile_length), 
    #             ((tile_x + 1) * tile_length, (tile_y    ) * tile_length), 
    #             ((tile_x    ) * tile_length, (tile_y + 1) * tile_length), 
    #             ((tile_x + 1) * tile_length, (tile_y + 1) * tile_length)]
    # print(tile_tmp)

    # poly1_points = order_points(tile_tmp)
    # poly2_points = order_points(p_xy_list)

    # poly1 = Polygon(poly1_points)
    # poly2 = Polygon(poly2_points)


    # print(poly1)
    # print(poly2)
    
    # inter1 = poly1.intersection(poly2).area
    # inter2 = poly2.intersection(poly1).area
    # print(inter1, inter2)

    # move_file()