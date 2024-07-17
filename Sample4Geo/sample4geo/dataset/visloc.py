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

TRAIN_LIST = [1,2,3,4,5,6,7,8]
TEST_LIST = []
# OK_LIST = [1]

TILE_SIZE = 256

THRESHOLD = 0.4


def latlon_to_meters(lat, lon):
    """Convert lat/lon to meters."""
    origin_shift = 2 * math.pi * 6378137 / 2.0
    mx = lon * origin_shift / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my

def meters_to_latlon(mx, my):
    """Convert meters to lat/lon."""
    origin_shift = 2 * math.pi * 6378137 / 2.0
    lon = (mx / origin_shift) * 180.0
    lat = (my / origin_shift) * 180.0
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180.0)) - math.pi / 2.0)
    return lat, lon

def tile_to_meters(tx, ty, zoom):
    """Convert tile coordinates to meters."""
    origin_shift = 2 * math.pi * 6378137 / 2.0
    initial_resolution = 2 * math.pi * 6378137 / 256.0
    resolution = initial_resolution / (2**zoom)
    
    minx = tx * 256 * resolution - origin_shift
    miny = ty * 256 * resolution - origin_shift
    maxx = (tx + 1) * 256 * resolution - origin_shift
    maxy = (ty + 1) * 256 * resolution - origin_shift
    
    return (minx + maxx) / 2, (miny + maxy) / 2

def tile_center_latlon(left_top_lat, left_top_lon, right_bottom_lat, right_bottom_lon, zoom, x, y):
    """Calculate the center lat/lon of a tile."""
    # Convert the corner coordinates to meters
    left_top_mx, left_top_my = latlon_to_meters(left_top_lat, left_top_lon)
    right_bottom_mx, right_bottom_my = latlon_to_meters(right_bottom_lat, right_bottom_lon)
    
    # Calculate the size of the area in meters
    total_width = abs(right_bottom_mx - left_top_mx)
    total_height = abs(right_bottom_my - left_top_my)
    
    # Calculate the number of tiles at the given zoom level
    num_tiles = 2 ** zoom
    
    # Calculate the meters per tile
    meters_per_tile_x = total_width / num_tiles
    meters_per_tile_y = total_height / num_tiles
    
    # Calculate the center of the tile in meters
    tile_center_mx = left_top_mx + (x + 0.5) * meters_per_tile_x
    tile_center_my = right_bottom_my + (num_tiles - y - 0.5) * meters_per_tile_y
    
    # Convert the tile center from meters to lat/lon
    center_lat, center_lon = meters_to_latlon(tile_center_mx, tile_center_my)
    
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


def tile2sate(tile_name):
    tile_name = tile_name.replace('.png', '')
    str_i, zoom_level, tile_x, tile_y = tile_name.split('_')
    zoom_level = int(zoom_level)
    tile_x = int(tile_x)
    tile_y = int(tile_y)
    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[str_i]
    tile_center_latlon(lt_lat, lt_lon, rb_lat, rb_lon, zoom_level, tile_x, tile_y)


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
            if oc > THRESHOLD:
                tile_oc_expand_list.append(f'{str_i}_{zoom_level}_{tile_x_i:03}_{tile_y_i:03}.png')
                tile_oc_expand_weight_list.append(oc)
    return tile_iou_expand_list, tile_iou_expand_weight_list, tile_oc_expand_list, tile_oc_expand_weight_list


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
    zoom_list = zoom_list[-5:]

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

        tile_iou_expand_list, tile_iou_expand_weight_list, tile_oc_expand_list, tile_oc_expand_weight_list \
            = tile_expand(str_i, cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, tile_x_max, tile_y_max, debug)

        result["pair_iou_sate_img_list"].extend(tile_iou_expand_list)
        result["pair_iou_sate_weight_list"].extend(tile_iou_expand_weight_list)
        result["pair_oc_sate_img_list"].extend(tile_oc_expand_list)
        result["pair_oc_sate_weight_list"].extend(tile_oc_expand_weight_list)
    
    if len(result["pair_iou_sate_img_list"]) == 0:
        return None

    if debug:
        print(result)
    
    return result


def save_pairs_meta_data(pairs_drone2sate_list, pkl_save_path, pair_save_dir):
    pairs_iou_sate2drone_dict = {}
    pairs_iou_drone2sate_dict = {}
    pairs_oc_sate2drone_dict = {}
    pairs_oc_drone2sate_dict = {}
    
    drone_save_dir = os.path.join(pair_save_dir, 'drone')
    sate_iou_save_dir = os.path.join(pair_save_dir, 'satellite', 'iou')
    sate_oc_save_dir = os.path.join(pair_save_dir, 'satellite', 'oc')
    os.makedirs(drone_save_dir, exist_ok=True)
    os.makedirs(sate_iou_save_dir, exist_ok=True)
    os.makedirs(sate_oc_save_dir, exist_ok=True)

    pairs_drone2sate_list_save = []

    for pairs_drone2sate in pairs_drone2sate_list:
        
        str_i = pairs_drone2sate['str_i']
        pair_iou_sate_img_list = pairs_drone2sate["pair_iou_sate_img_list"]
        pair_oc_sate_img_list = pairs_drone2sate["pair_oc_sate_img_list"]

        drone_img = pairs_drone2sate["drone_img"]
        drone_img_dir = pairs_drone2sate["drone_img_dir"]
        drone_img_name = drone_img.replace('.JPG', '')
        sate_img_dir = pairs_drone2sate["sate_img_dir"]

        drone_save_path = os.path.join(drone_save_dir, drone_img_name)
        os.makedirs(drone_save_path, exist_ok=True)
        sate_iou_save_path = os.path.join(sate_iou_save_dir, drone_img_name)
        os.makedirs(sate_iou_save_path, exist_ok=True)
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
            "pairs_oc_sate2drone_dict": pairs_oc_sate2drone_dict,
            "pairs_oc_drone2sate_dict": pairs_oc_drone2sate_dict,
            "pairs_oc_match_set": pairs_oc_match_set,
        }, f)


def process_visloc_data(root, save_root):
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

                # if is_point_in_rectangle(
                #     cur_lat, cur_lon, 
                #     sate_meta_data[str_i]["rect_1_LT_lat"],
                #     sate_meta_data[str_i]["rect_1_LT_lon"],
                #     sate_meta_data[str_i]["rect_1_RB_lat"],
                #     sate_meta_data[str_i]["rect_1_RB_lon"],
                # ) or is_point_in_rectangle(
                #     cur_lat, cur_lon, 
                #     sate_meta_data[str_i]["rect_2_LT_lat"],
                #     sate_meta_data[str_i]["rect_2_LT_lon"],
                #     sate_meta_data[str_i]["rect_2_RB_lat"],
                #     sate_meta_data[str_i]["rect_2_RB_lon"],
                # ):
                drone_meta_data_list.append(tmp_meta_data)

    random.shuffle(drone_meta_data_list)
    processed_data = []
    # train_num = len(drone_meta_data_list) * 4 // 5
    # drone_meta_data_train_list = drone_meta_data_list[: train_num]
    # drone_meta_data_test_list = drone_meta_data_list[train_num: ]

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

    train_pkl_save_path = os.path.join(save_root, 'train_pair_meta.pkl')
    train_data_save_dir = os.path.join(save_root, 'train')
    save_pairs_meta_data(processed_data_train, train_pkl_save_path, train_data_save_dir)

    test_pkl_save_path = os.path.join(save_root, 'test_pair_meta.pkl')
    test_data_save_dir = os.path.join(save_root, 'test')
    save_pairs_meta_data(processed_data_test, test_pkl_save_path, test_data_save_dir)

    os.makedirs(os.path.join(save_root, 'all_satellite'))
    shutil.copytree(os.path.join(save_root, 'train', 'satellite'), os.path.join(save_root, 'all_satellite'), dirs_exist_ok=True)
    shutil.copytree(os.path.join(save_root, 'test', 'satellite'), os.path.join(save_root, 'all_satellite'), dirs_exist_ok=True)


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


class VisLocDatasetTrain(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 drone2sate=True,
                 transforms_query=None,
                 transforms_gallery=None,
                 group_len=2,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='iou'):
        super().__init__()
        
        with open(pairs_meta_file, 'rb') as f:
            pairs_meta_data = pickle.load(f)

        self.group_len = group_len

        self.pairs = []

        pairs_drone2sate_list = pairs_meta_data['pairs_drone2sate_list']
        self.pairs_sate2drone_dict = pairs_meta_data[f'pairs_{mode}_sate2drone_dict']
        self.pairs_drone2sate_dict = pairs_meta_data[f'pairs_{mode}_drone2sate_dict']
        self.pairs_match_set = pairs_meta_data[f'pairs_{mode}_match_set']

        self.drone2sate = drone2sate
        if drone2sate:
            for pairs_drone2sate in pairs_drone2sate_list:
                drone_img_dir = pairs_drone2sate['drone_img_dir']
                drone_img = pairs_drone2sate['drone_img']
                sate_img_dir = pairs_drone2sate['sate_img_dir']
                pair_sate_img_list = pairs_drone2sate[f'pair_{mode}_sate_img_list']
                pair_sate_weight_list = pairs_drone2sate[f'pair_{mode}_sate_weight_list']
                str_i = pairs_drone2sate['str_i']
                drone_img_file = f'{drone_img_dir}/{drone_img}'
            
                for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                    sate_img_file = f'{sate_img_dir}/{pair_sate_img}'  
                    self.pairs.append((drone_img_file, sate_img_file, pair_sate_weight))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

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
        
        return query_img, gallery_img, positive_weight

    def __len__(self):
        return len(self.samples)


    def shuffle_group(self, ):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        print("\nShuffle Dataset in Groups:")
        
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
                        drone_img_path = os.path.join(drone_img_dir, drone_img_name)
                        sate_img_path = os.path.join(sate_img_dir, sate_img_name)
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
    
        # pbar.close()
        
        # wait before closing progress bar
        time.sleep(0.3)
        
        self.samples = batches
        
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

    
    def shuffle(self, ):

            '''
            custom shuffle function for unique class_id sampling in batch
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
            pbar = tqdm()

            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    drone_img, sate_img, _ = pair

                    drone_img_name = drone_img.split('/')[-1]
                    sate_img_name = sate_img.split('/')[-1]

                    # print(drone_img_name, sate_img_name)

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
        
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  

class VisLocDatasetEval(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 view,
                 mode='oc',
                 sate_img_dir='',
                 transforms=None,
                 ):
        super().__init__()
        
        with open(pairs_meta_file, 'rb') as f:
            pairs_meta_data = pickle.load(f)         

        self.images = []
        self.images_path = []
        self.images_loc_xy = []

        if view == 'drone':
            pairs_drone2sate_list = pairs_meta_data['pairs_drone2sate_list']
            self.pairs_sate2drone_dict = pairs_meta_data[f'pairs_{mode}_sate2drone_dict']
            self.pairs_drone2sate_dict = pairs_meta_data[f'pairs_{mode}_drone2sate_dict']
            self.pairs_match_set = pairs_meta_data[f'pairs_{mode}_match_set']
            for pairs_drone2sate in pairs_drone2sate_list:
                self.images_path.append(os.path.join(pairs_drone2sate['drone_img_dir'], pairs_drone2sate['drone_img']))
                self.images.append(pairs_drone2sate['drone_img'])
                self.images_loc_xy((pairs_drone2sate['lat'], pairs_drone2sate['lon'])) 
        elif view == 'sate':
            sate_img_dir_list, sate_img_list = get_sate_data(root_dir=sate_img_dir)
            # print('???????', sate_datas['sate_img'])
            for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                self.images_path.append(os.path.join(sate_img_dir, sate_img))
                self.images.append(sate_img)
                self.images_loc_xy.append(tile2sate(sate_img))

        self.transforms = transforms


    def __getitem__(self, index):
        
        img_path = self.images_path[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        #if self.mode == "sat":
        
        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
            
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
            
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return img

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)


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



if __name__ == '__main__':
    root = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    save_root = '/home/xmuairmud/data/UAV_VisLoc_dataset/test'
    process_visloc_data(root, save_root)

    # tile_satellite()

    # src_path = '/home/xmuairmud/data/UAV_VisLoc_dataset/11/tile'
    # dst_path = '/home/xmuairmud/data/UAV_VisLoc_dataset/11/satellite'
    # copy_png_files(src_path, dst_path)
