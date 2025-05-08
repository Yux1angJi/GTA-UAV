from operator import length_hint
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import torch
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
import json
import open3d as o3d
import torch
from transformers import AutoTokenizer

from .pc_utils import *


TILE_SIZE = 256
THRESHOLD = 0.39
SEMI_THRESHOLD = 0.14
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


def tile2sate(tile_name):
    tile_name = tile_name.replace('.png', '')
    str_i, zoom_level, tile_x, tile_y = tile_name.split('_')
    zoom_level = int(zoom_level)
    tile_x = int(tile_x)
    tile_y = int(tile_y)
    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[str_i]
    return tile_center_latlon(lt_lat, lt_lon, rb_lat, rb_lon, zoom_level, tile_x, tile_y, str_i)

def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


class VisLocMMDatasetTrain(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 transforms_drone_img=None,
                 transforms_drone_depth=None,
                 transforms_drone_geo=None,
                 transforms_satellite=None,
                 tokenizer="openai/clip-vit-base-patch32",
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 prob_drop_depth=0.0,
                 prob_drop_text=0.0,
                 prob_drop_pc=0.0,
                 augment_pc=True):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root

        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_depth_dir = pair_drone2sate['drone_depth_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            drone_depth_name = pair_drone2sate['drone_depth_name']
            drone_lidar_dir = pair_drone2sate['drone_lidar_dir']
            drone_lidar_name = pair_drone2sate['drone_lidar_name']
            drone_img_desc = pair_drone2sate['drone_img_desc']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            # Training with Positive-only data or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
            drone_lidar_file = os.path.join(data_root, drone_lidar_dir, drone_lidar_name)
            drone_depth_file = os.path.join(data_root, drone_depth_dir, drone_depth_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, drone_lidar_file, drone_depth_file, drone_img_desc,
                                    sate_img_file, pair_sate_weight))

            # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate['pair_pos_semipos_sate_img_list']
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.transforms_drone_img = transforms_drone_img
        self.transforms_drone_depth = transforms_drone_depth
        self.transforms_drone_geo = transforms_drone_geo
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.transforms_satellite = transforms_satellite
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.prob_drop_depth = prob_drop_depth
        self.prob_drop_text = prob_drop_text
        self.prob_drop_pc = prob_drop_pc
        self.augment_pc = augment_pc

        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]

        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        drone_img_path, drone_lidar_path, drone_depth_path, drone_img_desc, \
            satellite_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        drone_img = cv2.imread(drone_img_path)
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        drone_lidar = o3d.io.read_point_cloud(drone_lidar_path)
        drone_lidar = np.array(drone_lidar.points, dtype=np.float32)

        N = drone_lidar.shape[0]
        if N > 2048:
            indices = np.random.choice(N, 2048, replace=False)
        else:
            indices = np.random.choice(N, 2048, replace=True)

        drone_lidar = drone_lidar[indices]

        drone_lidar = self.pc_norm(drone_lidar)
        drone_lidar = torch.from_numpy(drone_lidar)

        if self.augment_pc:
            drone_lidar = random_scale_point_cloud(drone_lidar[None, ...])
            drone_lidar = shift_point_cloud(drone_lidar)
            drone_lidar = rotate_perturbation_point_cloud(drone_lidar)
            drone_lidar = rotate_point_cloud(drone_lidar)
            drone_lidar = drone_lidar.squeeze()
            drone_lidar = torch.from_numpy(drone_lidar)

        drone_lidar_pts = drone_lidar
        drone_lidar_clr = torch.ones_like(drone_lidar_pts).float() * 0.4

        drone_depth = cv2.imread(drone_depth_path, cv2.IMREAD_UNCHANGED)
        drone_depth = (drone_depth / 256).astype(np.uint8)
        if len(drone_depth.shape) == 2:
            drone_depth = np.expand_dims(drone_depth, axis=2)

        satellite_img = cv2.imread(satellite_img_path)
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            drone_depth = cv2.flip(drone_depth, 1)

        if self.transforms_drone_geo is not None:
            # Do the geo transforms consistently to RGB and Depth
            transformed = self.transforms_drone_geo(image=drone_img, mask=drone_depth)
            drone_img = transformed['image']
            drone_depth = transformed['mask']
        
        # image transforms
        if self.transforms_drone_img is not None:
            drone_img = self.transforms_drone_img(image=drone_img)['image']

        if self.transforms_drone_depth is not None:
            drone_depth = self.transforms_drone_depth(image=drone_depth)['image']

        # drop depth
        if np.random.random() < self.prob_drop_depth:
            drone_depth = torch.zeros_like(drone_depth)

        if np.random.random() < self.prob_drop_text:
            drone_img_desc = ""

        if np.random.random() < self.prob_drop_pc:
            drone_lidar_pts = torch.zeros_like(drone_lidar_pts)
            drone_lidar_clr = torch.zeros_like(drone_lidar_clr)

        # text tokenize
        drone_img_desc = self.tokenizer(
                [drone_img_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=77, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
        drone_img_desc = {k: v.squeeze() for k, v in drone_img_desc.items()}

        satellite_img_desc = self.tokenizer(
                ["satellite image"],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=77, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
        satellite_img_desc = {k: v.squeeze() for k, v in satellite_img_desc.items()}

        if self.transforms_satellite is not None:
            satellite_img = self.transforms_satellite(image=satellite_img)['image']
        
        sample =  {
            "drone_img": drone_img, 
            "drone_lidar_pts": drone_lidar_pts,
            "drone_lidar_clr": drone_lidar_clr, 
            "drone_desc": drone_img_desc,
            "drone_depth": drone_depth,
            "satellite_img": satellite_img, 
            "satellite_desc": satellite_img_desc,
            "positive_weight": positive_weight,
        }

        return sample

    def __len__(self):
        return len(self.samples)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

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
                
                drone_img, drone_lidar, drone_depth, drone_img_desc, sate_img, _ = pair
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


class VisLocMMDatasetEval(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='DImg2SImg',
                 pairs_sate2drone_dict=None,
                 transforms_rgb=None,
                 transforms_depth=None,
                 tokenizer="openai/clip-vit-base-patch32",
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)    

        self.drone_img_paths = []
        self.drone_lidar_paths = []
        self.drone_depth_paths = []
        self.drone_img_desc = []
        self.drone_img_names = []
        self.drone_loc_xys = []

        self.satellite_img_paths = []
        self.satellite_img_names = []
        self.satellite_loc_xys = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        self.view = view

        if view == 'drone':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_depth_name = pair_drone2sate['drone_depth_name']
                drone_depth_dir = pair_drone2sate['drone_depth_dir']
                drone_lidar_name = pair_drone2sate['drone_lidar_name']
                drone_lidar_dir = pair_drone2sate['drone_lidar_dir']
                drone_img_desc = pair_drone2sate['drone_img_desc']
                drone_loc_xy = pair_drone2sate['drone_loc_lat_lon']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.drone_img_paths.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.drone_lidar_paths.append(os.path.join(data_root, drone_lidar_dir, drone_lidar_name))
                    self.drone_depth_paths.append(os.path.join(data_root, drone_depth_dir, drone_depth_name))
                    self.drone_img_desc.append(drone_img_desc)
                    self.drone_img_names.append(drone_img_name)
                    self.drone_loc_xys.append((drone_loc_xy[0], drone_loc_xy[1]))

        elif view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)

                    loc_center = tile2sate(sate_img)
                    self.satellite_loc_xys.append(loc_center)
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)
                    loc_center = tile2sate(sate_img)
                    self.satellite_loc_xys.append(loc_center)
        
        self.transforms_rgb = transforms_rgb
        self.transforms_depth = transforms_depth
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __getitem__(self, index):
        sample = {}
        if self.view == 'drone':
            ## RGB
            img_path = self.drone_img_paths[index]
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transforms_rgb(image=img)['image']
            
            sample["drone_img"] = img
            
            ## LiDAR
            lidar_path = self.drone_lidar_paths[index]
            
            lidar = o3d.io.read_point_cloud(lidar_path)
            lidar = np.array(lidar.points, dtype=np.float32)

            N = lidar.shape[0]
            if N > 2048:
                indices = np.random.choice(N, 2048, replace=False)
            else:
                indices = np.random.choice(N, 2048, replace=True)
            lidar = lidar[indices]

            ## TODO point cloud error
            lidar = lidar[:, [1, 2, 0]]
            lidar[:, 0] = -lidar[:, 0]
            lidar[:, 2] = -lidar[:, 2]
            
            lidar = self.pc_norm(lidar)
            lidar = torch.from_numpy(lidar)

            sample["drone_lidar_pts"] = lidar
            sample["drone_lidar_clr"] = torch.ones_like(lidar).float() * 0.4
            
            ## Depth
            depth_path = self.drone_depth_paths[index]
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=2)
            depth = (depth / 256).astype(np.uint8)

            depth = self.transforms_depth(image=depth)['image']

            sample["drone_depth"] = depth

            ## Description text
            drone_desc = self.drone_img_desc[index]
            drone_desc = self.tokenizer(
                [drone_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=77, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            sample["drone_desc"] = {k: v.squeeze() for k, v in drone_desc.items()}

        elif self.view == 'sate':
            img_path = self.satellite_img_paths[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image transforms
            img = self.transforms_rgb(image=img)['image']
            
            sample["satellite_img"] = img

            ## Description text
            sate_desc = ""
            sate_desc = self.tokenizer(
                [sate_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=77, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            # sample["satellite_desc"] = {k: v.squeeze() for k, v in sate_desc.items()}

        return sample

    def __len__(self):
        if self.view == 'drone':
            return len(self.drone_img_names)
        else:
            return len(self.satellite_img_names)
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   eval_robust=False):

    mean_d = [0.5]
    std_d = [0.5]
    mean_rgbd = mean + mean_d
    std_rgbd = std + std_d
    

    val_sat_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
    if eval_robust:
        val_drone_rgb_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.OneOf([
                                            #   A.GridDropout(ratio=0.8, p=1.0),
                                            # A.CoarseDropout(max_holes=1,
                                            #                    max_height=int(0.6*img_size[0]),
                                            #                    max_width=int(0.6*img_size[0]),
                                            #                    min_holes=1,
                                            #                    min_height=int(0.6*img_size[0]),
                                            #                    min_width=int(0.6*img_size[0]),
                                            #                    p=1.0),
                                            SaltAndPepperNoise(amount=0.02, p=1.0),
                                            ], p=1.0),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    ])
    else:
        val_drone_rgb_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.Normalize(mean, std),
                                    ToTensorV2(),
                                    ])
    val_drone_depth_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean_d, std_d),
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

    train_drone_geo_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                            A.RandomRotate90(p=1.0),
                                           ],
                                           is_check_shapes=False
                                          )

    train_drone_rgb_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
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
                                        ],
                                    )
    train_drone_depth_transforms = A.Compose([
        A.Normalize(mean_d, std_d),
        ToTensorV2(),
    ])
    
    return val_sat_transforms, val_drone_rgb_transforms, val_drone_depth_transforms, train_sat_transforms, train_drone_geo_transforms, train_drone_rgb_transforms, train_drone_depth_transforms

