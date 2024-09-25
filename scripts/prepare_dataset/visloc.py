from operator import length_hint
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
from PIL import Image
import math
from multiprocessing import Pool, cpu_count
import csv
import pickle
import random
import itertools
from geopy.distance import geodesic


Image.MAX_IMAGE_PIXELS = None
FOV_V = 48.44
FOV_H = 33.48

TRAIN_LIST = [1, 3, 5, 8]
TEST_LIST = [2, 4, 11]

# TRAIN_LIST = [1, 2, 3, 4, 5, 8, 11]
# TEST_LIST = [1, 2, 3, 4, 5, 8, 11]

TILE_SIZE = 256

THRESHOLD = 0.39
SEMI_THRESHOLD = 0.14


def tile_center_latlon(left_top_lat, left_top_lon, right_bottom_lat, right_bottom_lon, zoom, x, y, str_i):
    """Calculate the center lat/lon of a tile."""
    sate_h, sate_w = SATE_SIZE[str_i][0], SATE_SIZE[str_i][1]
    max_dim = max(sate_h, sate_w)
    max_zoom = math.ceil(math.log(max_dim / TILE_SIZE, 2))
    scale = 2 ** (max_zoom - zoom)

    scaled_width = math.ceil(sate_w / scale)
    scaled_height = math.ceil(sate_h / scale)

    coe_lon = (x + 0.5) * TILE_SIZE / scaled_width
    coe_lat = (y + 0.5) * TILE_SIZE / scaled_height

    # Calculate the size of each tile in degrees

    lat_diff = left_top_lat - right_bottom_lat
    lon_diff = right_bottom_lon - left_top_lon

    # Calculate the center of the tile in degrees
    center_lat = left_top_lat - coe_lat * lat_diff
    center_lon = left_top_lon + coe_lon * lon_diff

    return center_lat, center_lon

SATE_LATLON = {
    '01': [29.774065,115.970635,29.702283,115.996851],
    '02': [29.817376,116.033769,29.725402,116.064566],
    '03': [32.355491,119.805926,32.29029,119.900052],
    '04': [32.254036,119.90598,32.151018,119.954509],
    '05': [24.666899,102.340055,24.650422,102.365252],
    '06': [32.373177,109.63516,32.346944,109.656837],
    '07': [40.340058,115.791182,40.339604,115.79923],
    '08': [30.947227,120.136489,30.903521,120.252951],
    '10': [40.355093,115.776356,40.341475,115.794041],
    '11': [38.852301,101.013109,38.807825,101.092483],
}
## HW
SATE_SIZE = {
    '01': (26762,  9774),
    '02': (34291, 11482),
    '03': (24308, 35092),
    '04': (38408, 18093),
    '05': (6144,   9394),
    '06': (9780,   8082),
    '07': (170,    3000),
    '08': (16294, 43421),
    '10': (5077,   6593),
    '11': (16582, 29592),
}


def tile2sate(tile_name):
    tile_name = tile_name.replace('.png', '')
    str_i, zoom_level, tile_x, tile_y = tile_name.split('_')
    zoom_level = int(zoom_level)
    tile_x = int(tile_x)
    tile_y = int(tile_y)
    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[str_i]
    return tile_center_latlon(lt_lat, lt_lon, rb_lat, rb_lon, zoom_level, tile_x, tile_y, str_i)


def cp_tile():
    root = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    dst_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset/all_satellite_z31'
    os.makedirs(dst_dir, exist_ok=True)
    for i in range(1, 12):
        if i not in TRAIN_LIST:
            continue
        str_i = f'{i:02}'
        tile_dir = os.path.join(root, str_i, 'tile')
        zoom_list = os.listdir(tile_dir)
        zoom_list = [int(x) for x in zoom_list]
        zoom_list.sort()
        zoom_max = zoom_list[-1]
        zoom_list = zoom_list[-3:-1]
        for zoom in zoom_list:
            tile_zoom_dir = os.path.join(tile_dir, f'{zoom}')
            for _, _, files in os.walk(tile_zoom_dir):
                for file in files:
                    shutil.copy(os.path.join(tile_zoom_dir, file), dst_dir)


def is_point_in_rectangle(point_lat, point_lon, rect_top_left_lat, rect_top_left_lon, rect_bottom_right_lat, rect_bottom_right_lon):
    # 判断点是否在长方形区域内
    if (rect_top_left_lat >= point_lat >= rect_bottom_right_lat) and (rect_top_left_lon <= point_lon <= rect_bottom_right_lon):
        return True
    else:
        return False


def order_points(points):
    hull = ConvexHull(points)
    ordered_points = [points[i] for i in hull.vertices]
    return ordered_points


def calc_intersect_area(poly1, poly2):
    # 计算交集
    intersection = poly1.intersection(poly2)
    return intersection.area


def process_tile(args):
    scaled_image, str_i, zoom_dir, zoom, x, y, tile_size = args
    box = (x, y, min(x + tile_size, scaled_image.width), min(y + tile_size, scaled_image.height))
    tile = scaled_image.crop(box)
    
    # 创建一个透明背景的图像
    transparent_tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    
    # 将裁剪后的瓦片粘贴到透明背景中
    transparent_tile.paste(tile, (0, 0))
    
    transparent_tile.save(os.path.join(zoom_dir, f'{str_i}_{zoom}_{x // tile_size:03}_{y // tile_size:03}.png'))


def tile_satellite():
    root_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    
    for i in range(10, 12):
        if i == 9:
            continue
        file_dir = os.path.join(root_dir, f'{i:02}')
        tile_dir = os.path.join(file_dir, 'tile')
        os.makedirs(tile_dir, exist_ok=True)

        image_path = os.path.join(file_dir, f'satellite{i:02}.tif')
        image = Image.open(image_path)

        # 瓦片大小
        tile_size = TILE_SIZE

        # 计算最大缩放级别
        max_dim = max(image.width, image.height)
        max_zoom = math.ceil(math.log(max_dim / tile_size, 2))


        # 切分瓦片
        for zoom in range(max_zoom + 1):
            zoom_dir = os.path.join(tile_dir, str(zoom))
            if not os.path.exists(zoom_dir):
                os.makedirs(zoom_dir)
            
            scale = 2 ** (max_zoom - zoom)
            scaled_width = math.ceil(image.width / scale)
            scaled_height = math.ceil(image.height / scale)
            
            # 缩放图像
            scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

            tasks = []

            for x in range(0, scaled_width, tile_size):
                for y in range(0, scaled_height, tile_size):
                    tasks.append((scaled_image, f'{i:02}', zoom_dir, zoom, x, y, tile_size))

            with Pool(cpu_count()) as pool:
                pool.map(process_tile, tasks)
            
            # for x in range(0, scaled_width, tile_size):
            #     for y in range(0, scaled_height, tile_size):
            #         box = (x, y, min(x + tile_size, scaled_width), min(y + tile_size, scaled_height))
            #         tile = scaled_image.crop(box)
                    
            #         transparent_tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
            #         transparent_tile.paste(tile, (0, 0))
                    
            #         transparent_tile.save(os.path.join(zoom_dir, f'{x // tile_size}_{y // tile_size}.png'))

    print('瓦片切分完成')


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


def geo_to_image_coords(lat, lon, lat1, lon1, lat2, lon2, H, W):
    R = 6371000  # 地球半径（米）

    # 计算中心纬度
    center_lat = (lat1 + lat2) / 2

    # 计算地理范围（米）
    x_range = R * (lon2 - lon1) * math.cos(math.radians(center_lat))
    y_range = R * (lat2 - lat1)

    # 计算目标点相对于左上角的平面坐标偏移（米）
    x_offset = R * (lon - lon1) * math.cos(math.radians((lat1 + lat) / 2))
    y_offset = R * (lat - lat1)

    # 计算图像中的坐标
    x = (x_offset / x_range) * W
    y = (y_offset / y_range) * H

    return int(x), int(y)


def offset_to_latlon(latitude, longitude, dx, dy):
    # Earth radius in meters
    R = 6378137
    dlat = dy / R
    dlon = dx / (R * math.cos(math.pi * latitude / 180))
    
    lat_offset = dlat * 180 / math.pi
    lon_offset = dlon * 180 / math.pi
    
    return latitude + lat_offset, longitude + lon_offset


def calculate_coverage_endpoints(heading_angle, height, cur_lat, cur_lon, fov_horizontal=FOV_H, fov_vertical=FOV_V, debug=False):
    # Convert angles from degrees to radians for trigonometric functions
    heading_angle_rad = math.radians(heading_angle)
    fov_horizontal_rad = math.radians(fov_horizontal)
    fov_vertical_rad = math.radians(fov_vertical)
    
    # Calculate the half lengths of the coverage area on the ground
    half_coverage_length_h = height * math.tan(fov_horizontal_rad / 2)
    half_coverage_length_v = height * math.tan(fov_vertical_rad / 2)
    
    # Adjust heading angle for coordinate system where East is 0 and North is 90
    adjusted_heading_angle_rad = math.radians((90 - heading_angle) % 360)
    
    # Calculate the offsets for the four endpoints
    offset_top_left_x = -half_coverage_length_h * math.cos(adjusted_heading_angle_rad) - half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_top_left_y = -half_coverage_length_h * math.sin(adjusted_heading_angle_rad) + half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_top_right_x = half_coverage_length_h * math.cos(adjusted_heading_angle_rad) - half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_top_right_y = half_coverage_length_h * math.sin(adjusted_heading_angle_rad) + half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_bottom_left_x = -half_coverage_length_h * math.cos(adjusted_heading_angle_rad) + half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_bottom_left_y = -half_coverage_length_h * math.sin(adjusted_heading_angle_rad) - half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    offset_bottom_right_x = half_coverage_length_h * math.cos(adjusted_heading_angle_rad) + half_coverage_length_v * math.sin(adjusted_heading_angle_rad)
    offset_bottom_right_y = half_coverage_length_h * math.sin(adjusted_heading_angle_rad) - half_coverage_length_v * math.cos(adjusted_heading_angle_rad)
    
    if debug:
        print(
            'offset',
            offset_top_left_x, 
            offset_top_left_y,
            offset_top_right_x, 
            offset_top_right_y,
            offset_bottom_left_x, 
            offset_bottom_left_y,
            offset_bottom_right_x, 
            offset_bottom_right_y
        )

    return {
        "top_left": offset_to_latlon(cur_lat, cur_lon, offset_top_left_x, offset_top_left_y),
        "top_right": offset_to_latlon(cur_lat, cur_lon, offset_top_right_x, offset_top_right_y),
        "bottom_left": offset_to_latlon(cur_lat, cur_lon, offset_bottom_left_x, offset_bottom_left_y),
        "bottom_right": offset_to_latlon(cur_lat, cur_lon, offset_bottom_right_x, offset_bottom_right_y)
    }


def tile_expand(str_i, cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, tile_x_max, tile_y_max, debug=False):
    tile_area = TILE_SIZE ** 2

    tile_u = max(0, cur_tile_y - 5)
    tile_d = min(cur_tile_y + 5, tile_y_max)
    tile_l = max(0, cur_tile_x - 5)
    tile_r = min(cur_tile_x + 5, tile_x_max)

    p_img_xy_scale_order = order_points(p_img_xy_scale)
    poly_p = Polygon(p_img_xy_scale_order)
    poly_p_area = poly_p.area

    tile_tmp = [((cur_tile_x    ) * TILE_SIZE, (cur_tile_y    ) * TILE_SIZE), 
                ((cur_tile_x + 1) * TILE_SIZE, (cur_tile_y    ) * TILE_SIZE), 
                ((cur_tile_x    ) * TILE_SIZE, (cur_tile_y + 1) * TILE_SIZE), 
                ((cur_tile_x + 1) * TILE_SIZE, (cur_tile_y + 1) * TILE_SIZE)]
    tile_tmp_order = order_points(tile_tmp)
    poly_tile = Polygon(tile_tmp_order)
    poly_tile_area = poly_tile.area

    tile_iou_expand_list = []
    tile_iou_expand_weight_list = []
    tile_semi_iou_expand_list = []
    tile_semi_iou_expand_weight_list = []
    tile_oc_expand_list = []
    tile_oc_expand_weight_list = []

    for tile_x_i in range(tile_l, tile_r + 1):
        for tile_y_i in range(tile_u, tile_d + 1):

            tile_tmp = [((tile_x_i    ) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i    ) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE)]
            tile_tmp_order = order_points(tile_tmp)
            poly_tile = Polygon(tile_tmp_order)
            poly_tile_area = poly_tile.area
            intersect_area = calc_intersect_area(poly_p, poly_tile)
            if debug:
                print('zoom=', zoom_level, cur_tile_x, cur_tile_y)
                print(tile_x_i, tile_y_i)
                print(intersect_area, tile_area, poly_p_area, intersect_area/tile_area, intersect_area/poly_p_area)
            oc = intersect_area / min(poly_p_area, poly_tile_area)
            iou = intersect_area / (poly_p_area + poly_tile_area - intersect_area)
            if iou > THRESHOLD:
                tile_iou_expand_list.append(f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png')
                tile_iou_expand_weight_list.append(iou)
            if iou > SEMI_THRESHOLD:
                tile_semi_iou_expand_list.append(f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png')
                tile_semi_iou_expand_weight_list.append(iou)
            if oc > THRESHOLD:
                tile_oc_expand_list.append(f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png')
                tile_oc_expand_weight_list.append(iou)
    return tile_iou_expand_list, tile_iou_expand_weight_list, tile_semi_iou_expand_list, tile_semi_iou_expand_weight_list, tile_oc_expand_list, tile_oc_expand_weight_list


def process_per_image(drone_meta_data):
    file_dir, str_i, drone_img, lat, lon, height, phi, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w = drone_meta_data

    # debug = (drone_img == '01_0015.JPG')
    debug = False

    p_latlon = calculate_coverage_endpoints(heading_angle=phi, height=height, cur_lat=lat, cur_lon=lon, debug=debug)
    
    if debug:
        print(p_latlon)

    zoom_list = os.listdir(os.path.join(file_dir, 'tile'))
    zoom_list = [int(x) for x in zoom_list]
    zoom_list.sort()
    zoom_max = zoom_list[-1]
    zoom_list = zoom_list[-3:-1]

    cur_img_x, cur_img_y = geo_to_image_coords(lat, lon, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
    p_img_xy = [
        geo_to_image_coords(v[0], v[1], sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
            for v in p_latlon.values()
    ]
    if debug:
        print('p_img_xy', p_img_xy)

    result = {
        "str_i": str_i,
        "drone_img_dir": os.path.join(file_dir, 'drone'),
        "drone_img": drone_img,
        "lat": lat,
        "lon": lon,
        "sate_img_dir": os.path.join(file_dir, 'satellite'),
        "pair_iou_sate_img_list": [],
        "pair_iou_sate_weight_list": [],
        "pair_semi_iou_sate_img_list": [],
        "pair_semi_iou_sate_weight_list": [],
        "pair_oc_sate_img_list": [],
        "pair_oc_sate_weight_list": [],
    }

    for zoom_level in zoom_list:
        scale = 2 ** (zoom_max - zoom_level)
        sate_pix_w_scale = math.ceil(sate_pix_w / scale)
        sate_pix_h_scale = math.ceil(sate_pix_h / scale)
        
        tile_x_max = sate_pix_w_scale // TILE_SIZE
        tile_y_max = sate_pix_h_scale // TILE_SIZE

        cur_img_x_scale = math.ceil(cur_img_x / scale)
        cur_img_y_scale = math.ceil(cur_img_y / scale)

        p_img_xy_scale = [
            (math.ceil(v[0] / scale), math.ceil(v[1] / scale)) 
                for v in p_img_xy
        ]

        cur_tile_x = cur_img_x_scale // TILE_SIZE
        cur_tile_y = cur_img_y_scale // TILE_SIZE

        tile_iou_expand_list, tile_iou_expand_weight_list, \
        tile_semi_iou_expand_list, tile_semi_iou_expand_weight_list, \
        tile_oc_expand_list, tile_oc_expand_weight_list, \
            = tile_expand(str_i, cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, tile_x_max, tile_y_max, debug)

        result["pair_iou_sate_img_list"].extend(tile_iou_expand_list)
        result["pair_iou_sate_weight_list"].extend(tile_iou_expand_weight_list)
        result["pair_semi_iou_sate_img_list"].extend(tile_semi_iou_expand_list)
        result["pair_semi_iou_sate_weight_list"].extend(tile_semi_iou_expand_weight_list)
        result["pair_oc_sate_img_list"].extend(tile_oc_expand_list)
        result["pair_oc_sate_weight_list"].extend(tile_oc_expand_weight_list)
    
    if len(result["pair_semi_iou_sate_img_list"]) == 0:
        return None

    if debug:
        print(result)
    
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

    pairs_drone2sate_list_save = []

    for pairs_drone2sate in pairs_drone2sate_list:
        
        str_i = pairs_drone2sate['str_i']
        pair_iou_sate_img_list = pairs_drone2sate["pair_iou_sate_img_list"]
        pair_semi_iou_sate_img_list = pairs_drone2sate["pair_semi_iou_sate_img_list"]
        pair_oc_sate_img_list = pairs_drone2sate["pair_oc_sate_img_list"]

        drone_img = pairs_drone2sate["drone_img"]
        drone_img_dir = pairs_drone2sate["drone_img_dir"]
        drone_img_name = drone_img.replace('.JPG', '')
        sate_img_dir = pairs_drone2sate["sate_img_dir"]

        drone_save_path = os.path.join(drone_save_dir, drone_img_name)
        os.makedirs(drone_save_path, exist_ok=True)
        sate_iou_save_path = os.path.join(sate_iou_save_dir, drone_img_name)
        os.makedirs(sate_iou_save_path, exist_ok=True)
        sate_semi_iou_save_path = os.path.join(sate_semi_iou_save_dir, drone_img_name)
        os.makedirs(sate_semi_iou_save_path, exist_ok=True)
        sate_oc_save_path = os.path.join(sate_oc_save_dir, drone_img_name)
        os.makedirs(sate_oc_save_path, exist_ok=True)

        flag = False
        for sate_img in pair_iou_sate_img_list:
            try:
                shutil.copy(os.path.join(drone_img_dir, drone_img), drone_save_path)
                shutil.copy(os.path.join(sate_img_dir, sate_img), sate_iou_save_path)
                pairs_iou_drone2sate_dict.setdefault(drone_img, []).append(f'{sate_img}')
                pairs_iou_sate2drone_dict.setdefault(f'{sate_img}', []).append(f'{drone_img}')
                flag = True
            except:
                print(f'Warning!! Can\'t find sate {sate_img} for {drone_img}.')
        for sate_img in pair_semi_iou_sate_img_list:
            try:
                shutil.copy(os.path.join(drone_img_dir, drone_img), drone_save_path)
                shutil.copy(os.path.join(sate_img_dir, sate_img), sate_semi_iou_save_path)
                pairs_semi_iou_drone2sate_dict.setdefault(drone_img, []).append(f'{sate_img}')
                pairs_semi_iou_sate2drone_dict.setdefault(f'{sate_img}', []).append(f'{drone_img}')
                flag = True
            except:
                print(f'Warning!! Can\'t find sate {sate_img} for {drone_img}.')
        for sate_img in pair_oc_sate_img_list:
            try:
                shutil.copy(os.path.join(drone_img_dir, drone_img), drone_save_path)
                shutil.copy(os.path.join(sate_img_dir, sate_img), sate_oc_save_path)
                pairs_oc_drone2sate_dict.setdefault(drone_img, []).append(f'{sate_img}')
                pairs_oc_sate2drone_dict.setdefault(f'{sate_img}', []).append(f'{drone_img}')
                flag = True
            except:
                print(f'Warning!! Can\'t find sate {sate_img} for {drone_img}.')
        if flag:
            pairs_drone2sate_list_save.append(pairs_drone2sate)

    pairs_iou_match_set = set()
    for sate_img, tile2drone in pairs_iou_sate2drone_dict.items():
        pairs_iou_sate2drone_dict[sate_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_iou_drone2sate_dict.items():
        pairs_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for sate_img in pairs_iou_drone2sate_dict[drone_img]:
            pairs_iou_match_set.add((drone_img, f'{sate_img}'))
    pairs_semi_iou_match_set = set()
    for sate_img, tile2drone in pairs_semi_iou_sate2drone_dict.items():
        pairs_semi_iou_sate2drone_dict[sate_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_semi_iou_drone2sate_dict.items():
        pairs_semi_iou_drone2sate_dict[drone_img] = list(set(drone2tile))
        for sate_img in pairs_semi_iou_drone2sate_dict[drone_img]:
            pairs_semi_iou_match_set.add((drone_img, f'{sate_img}'))
    pairs_oc_match_set = set()
    for sate_img, tile2drone in pairs_oc_sate2drone_dict.items():
        pairs_oc_sate2drone_dict[sate_img] = list(set(tile2drone))
    for drone_img, drone2tile in pairs_oc_drone2sate_dict.items():
        pairs_oc_drone2sate_dict[drone_img] = list(set(drone2tile))
        for sate_img in pairs_oc_drone2sate_dict[drone_img]:
            pairs_oc_match_set.add((drone_img, f'{sate_img}'))

    with open(pkl_save_path, 'wb') as f:
        pickle.dump({
            "pairs_drone2sate_list": pairs_drone2sate_list_save,
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


def process_visloc_data(root, save_root, split_type):
    processed_data_train = []
    processed_data_test = []

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    sate_meta_file = os.path.join(root, 'satellite_coordinates_range.csv')
    sate_meta_data = {}
    with open(sate_meta_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header = next(csvreader)
        for row in csvreader:
            name_sate = row[0][9: 11]
            sate_meta_data[name_sate] = {
                "LT_lat": float(row[1]),
                "LT_lon": float(row[2]),
                "RB_lat": float(row[3]),
                "RB_lon": float(row[4]),
                "rect_1_LT_lat": float(row[6]),
                "rect_1_LT_lon": float(row[7]),
                "rect_1_RB_lat": float(row[8]),
                "rect_1_RB_lon": float(row[9]),
                "rect_2_LT_lat": float(row[10]),
                "rect_2_LT_lon": float(row[11]),
                "rect_2_RB_lat": float(row[12]),
                "rect_2_RB_lon": float(row[13]),
            }

    train_drone_meta_data_list = []
    test_drone_meta_data_list = []
    drone_meta_data_list = []
    for i in range(1, 12):
        if i not in TRAIN_LIST and i not in TEST_LIST:
            continue
        str_i = f'{i:02}'
        file_dir = os.path.join(root, str_i)

        drone_meta_file = os.path.join(file_dir, f'{str_i}.csv')

        sate_img = cv2.imread(os.path.join(file_dir, f'satellite{str_i}.tif'))
        sate_pix_h, sate_pix_w, _ = sate_img.shape

        with open(drone_meta_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            # 读取文件的头部
            header = next(csvreader)
            # 逐行读取文件
            for row in csvreader:
                cur_lat = float(row[3])
                cur_lon = float(row[4])
                tmp_meta_data = (
                    file_dir,
                    str_i,
                    row[1],
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[-2]),
                    sate_meta_data[str_i]["LT_lat"],
                    sate_meta_data[str_i]["LT_lon"],
                    sate_meta_data[str_i]["RB_lat"],
                    sate_meta_data[str_i]["RB_lon"],
                    sate_pix_h,
                    sate_pix_w,
                )

                if split_type == 'cross_area':
                    if i in TRAIN_LIST:
                        train_drone_meta_data_list.append(tmp_meta_data)
                    else:
                        test_drone_meta_data_list.append(tmp_meta_data)
                else:
                    drone_meta_data_list.append(tmp_meta_data)

    if split_type == 'same_area':
        processed_data = []
        random.shuffle(drone_meta_data_list)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(process_per_image, drone_meta_data_list), total=len(drone_meta_data_list)):
                # 将每个返回值添加到结果列表中
                    processed_data.append(result)
        processed_data_wonone = []
        for result in processed_data:
            if result is not None:
                processed_data_wonone.append(result)
        
        data_num = len(processed_data_wonone)
        processed_data_train = processed_data_wonone[:data_num // 5 * 4]
        processed_data_test = processed_data_wonone[data_num // 5 * 4:]
    
    else:
        processed_data_train = []
        processed_data_test = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(process_per_image, train_drone_meta_data_list), total=len(train_drone_meta_data_list)):
                # 将每个返回值添加到结果列表中
                    processed_data_train.append(result)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(process_per_image, test_drone_meta_data_list), total=len(test_drone_meta_data_list)):
                # 将每个返回值添加到结果列表中
                    processed_data_test.append(result)
        train_processed_data_wonone = []
        for result in processed_data_train:
            if result is not None:
                train_processed_data_wonone.append(result)
        test_processed_data_wonone = []
        for result in processed_data_test:
            if result is not None:
                test_processed_data_wonone.append(result)
        processed_data_train = train_processed_data_wonone
        processed_data_test = test_processed_data_wonone


    train_pkl_save_path = os.path.join(save_root, 'train_pair_meta.pkl')
    train_data_save_dir = os.path.join(save_root, 'train')
    save_pairs_meta_data(processed_data_train, train_pkl_save_path, train_data_save_dir)

    test_pkl_save_path = os.path.join(save_root, 'test_pair_meta.pkl')
    test_data_save_dir = os.path.join(save_root, 'test')
    save_pairs_meta_data(processed_data_test, test_pkl_save_path, test_data_save_dir)


def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


def get_subset(s, group_len):
    # 返回集合 s 的所有子集
    x = len(s)
    subset_len = []
    for i in range(1 << x):
        subset = {s[j] for j in range(x) if (i & (1 << j))}
        if len(subset) == group_len:
            subset_len.append(subset)
    return subset_len


if __name__ == '__main__':
    root = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    save_root = '/home/xmuairmud/data/UAV_VisLoc_dataset/same_area_z31'
    process_visloc_data(root, save_root, 'same_area')
    # cp_tile()

    # tile_satellite()

    # src_path = '/home/xmuairmud/data/UAV_VisLoc_dataset/11/tile'
    # dst_path = '/home/xmuairmud/data/UAV_VisLoc_dataset/11/satellite'
    # copy_png_files(src_path, dst_path)

    # import pickle
    # with open('/home/xmuairmud/data/UAV_VisLoc_dataset/data_all_iou4_oc4_z3/train_pair_meta.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # for i in range(1):
    #     pairs_drone2sate = data['pairs_drone2sate_list'][i]
    #     str_i = pairs_drone2sate['str_i']
    #     lat = pairs_drone2sate['lat']
    #     lon = pairs_drone2sate['lon']
    #     print(pairs_drone2sate['drone_img'])
    #     print(pairs_drone2sate['lat'], pairs_drone2sate['lon'])
    #     for sate_oc_img, oc_weight in zip(pairs_drone2sate['pair_oc_sate_img_list'], pairs_drone2sate['pair_oc_sate_weight_list']):
    #         lat_i, lon_i = tile2sate(sate_oc_img)
    #         print(lat_i, lon_i)
    #         print(i, 'oc', oc_weight, geodesic((lat, lon), (lat_i, lon_i)).meters)
    #     for sate_iou_img, iou_weight in zip(pairs_drone2sate['pair_iou_sate_img_list'], pairs_drone2sate['pair_iou_sate_weight_list']):
    #         lat_i, lon_i = tile2sate(sate_oc_img)
    #         print(i, 'iou', iou_weight, geodesic((lat, lon), (lat_i, lon_i)).meters)
    
        