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


Image.MAX_IMAGE_PIXELS = None
FOV_V = 48.44
FOV_H = 33.48

OK_LIST = [1, 2, 3, 4, 8, 11]
# OK_LIST = [1]

TILE_SIZE = 256

def order_points(points):
    hull = ConvexHull(points)
    ordered_points = [points[i] for i in hull.vertices]
    return ordered_points


def calc_intersect_area(poly1, poly2):
    poly2 = order_points(poly2)
    poly2 = Polygon(poly2)
    # 计算交集
    intersection = poly1.intersection(poly2)
    return intersection.area


def process_tile(args):
    scaled_image, zoom_dir, x, y, tile_size = args
    box = (x, y, min(x + tile_size, scaled_image.width), min(y + tile_size, scaled_image.height))
    tile = scaled_image.crop(box)
    
    # 创建一个透明背景的图像
    transparent_tile = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    
    # 将裁剪后的瓦片粘贴到透明背景中
    transparent_tile.paste(tile, (0, 0))
    
    transparent_tile.save(os.path.join(zoom_dir, f'{x // tile_size:03}_{y // tile_size:03}.png'))


def tile_satellite():
    root_dir = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    
    for i in range(10, 12):
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
                    tasks.append((scaled_image, zoom_dir, x, y, tile_size))

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


def tile_expand(cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, debug=False):
    tile_area = TILE_SIZE ** 2

    tile_u = max(0, cur_tile_y - 5)
    tile_d = cur_tile_y + 5
    tile_l = max(0, cur_tile_x - 5)
    tile_r = cur_tile_x + 5

    p_img_xy_scale_order = order_points(p_img_xy_scale)
    poly_p = Polygon(p_img_xy_scale_order)
    poly_p_area = poly_p.area

    tile_expand_list = [f'{zoom_level}/{cur_tile_x:03}_{cur_tile_y:03}.png']

    for tile_x_i in range(tile_l, tile_r + 1):
        for tile_y_i in range(tile_u, tile_d + 1):
            if tile_x_i == cur_tile_x and tile_y_i == cur_tile_y:
                continue
            tile_tmp = [((tile_x_i    ) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i    ) * TILE_SIZE), 
                        ((tile_x_i    ) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE), 
                        ((tile_x_i + 1) * TILE_SIZE, (tile_y_i + 1) * TILE_SIZE)]
            intersect_area = calc_intersect_area(poly_p, tile_tmp)
            if debug:
                print('zoom=', zoom_level, cur_tile_x, cur_tile_y)
                print(tile_x_i, tile_y_i)
                print(intersect_area, tile_area, poly_p_area, intersect_area/tile_area, intersect_area/poly_p_area)
            if intersect_area / tile_area > 0.4 or intersect_area / poly_p_area > 0.4:
                tile_expand_list.append(f'{zoom_level}/{tile_x_i:03}_{tile_y_i:03}.png')
    return tile_expand_list
            

def process_per_image(drone_meta_data):
    file_dir, str_i, drone_img_path, lat, lon, height, phi, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w = drone_meta_data

    # debug = (drone_img_path == '01_0015.JPG')
    debug = False

    p_latlon = calculate_coverage_endpoints(heading_angle=phi, height=height, cur_lat=lat, cur_lon=lon, debug=debug)
    
    if debug:
        print(p_latlon)

    zoom_list = os.listdir(os.path.join(file_dir, 'tile'))
    zoom_list = [int(x) for x in zoom_list]
    zoom_list.sort()
    zoom_list = zoom_list[-3:]

    cur_img_x, cur_img_y = geo_to_image_coords(lat, lon, sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
    p_img_xy = [
        geo_to_image_coords(v[0], v[1], sate_lt_lat, sate_lt_lon, sate_rb_lat, sate_rb_lon, sate_pix_h, sate_pix_w)
            for v in p_latlon.values()
    ]
    if debug:
        print('p_img_xy', p_img_xy)

    result = {
        "file_dir": file_dir,
        "str_i": str_i,
        "drone_img": drone_img_path,
        "lat": lat,
        "lon": lon,
        "pair_tile_img_list": []
    }

    for zoom_level in zoom_list:
        scale = 2 ** (int(zoom_list[-1]) - zoom_level)
        sate_pix_w_scale = math.ceil(sate_pix_w / scale)
        sate_pix_h_scale = math.ceil(sate_pix_h / scale)

        cur_img_x_scale = math.ceil(cur_img_x / scale)
        cur_img_y_scale = math.ceil(cur_img_y / scale)

        p_img_xy_scale = [
            (math.ceil(v[0] / scale), math.ceil(v[1] / scale)) 
                for v in p_img_xy
        ]

        cur_tile_x = cur_img_x_scale // TILE_SIZE
        cur_tile_y = cur_img_y_scale // TILE_SIZE

        tile_expand_list = tile_expand(cur_tile_x, cur_tile_y, p_img_xy_scale, zoom_level, debug)

        result["pair_tile_img_list"].extend(tile_expand_list)
    
    if drone_img_path == '01_0015.JPG':
        print(result)
    
    return result



def process_visloc_data(root, save_root):
    processed_data = []
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
            }

    for i in range(1, 12):
        if i not in OK_LIST:
            continue
        str_i = f'{i:02}'
        file_dir = os.path.join(root, str_i)

        drone_meta_file = os.path.join(file_dir, f'{str_i}.csv')
        drone_meta_data_list = []

        sate_img = cv2.imread(os.path.join(file_dir, f'satellite{str_i}.tif'))
        sate_pix_h, sate_pix_w, _ = sate_img.shape

        with open(drone_meta_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            # 读取文件的头部
            header = next(csvreader)
            # 逐行读取文件
            for row in csvreader:
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
                drone_meta_data_list.append(tmp_meta_data)
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(process_per_image, drone_meta_data_list), total=len(drone_meta_data_list)):
                # 将每个返回值添加到结果列表中
                    processed_data.append(result)

    pkl_save_path = os.path.join(save_root, 'processed_data_list.pkl')
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(processed_data, f)






if __name__ == '__main__':
    root = '/home/xmuairmud/data/UAV_VisLoc_dataset'
    save_root = '/home/xmuairmud/data/UAV_VisLoc_dataset/preprocess'
    process_visloc_data(root, save_root)
