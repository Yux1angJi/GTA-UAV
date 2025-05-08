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
import open3d as o3d
import torch
from transformers import AutoTokenizer

from .pc_utils import *


SATE_LENGTH = 24576
TILE_LENGTH = 256


def sate2loc(tile_zoom, offset, tile_x, tile_y):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_x = (tile_pix * (tile_x+1/2+offset/TILE_LENGTH)) * 0.45
    loc_y = (tile_pix * (tile_y+1/2+offset/TILE_LENGTH)) * 0.45
    return loc_x, loc_y


def get_sate_data(root_dir):
    sate_img_dir_list = []
    sate_img_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            sate_img_dir_list.append(root)
            sate_img_list.append(file)
    return sate_img_dir_list, sate_img_list


class GTAMMDatasetTrain(Dataset):
    
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
        self.augment_pc = augment_pc

        self.pairs = []
        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        for pair_drone2sate in pairs_meta_data:
            drone_img_dir = pair_drone2sate['drone_img_dir']
            drone_img_name = pair_drone2sate['drone_img_name']
            drone_lidar_dir = pair_drone2sate['drone_lidar_dir']
            drone_lidar_name = pair_drone2sate['drone_lidar_name']
            drone_depth_dir = pair_drone2sate['drone_depth_dir']
            drone_depth_name = pair_drone2sate['drone_depth_name']
            drone_img_desc = pair_drone2sate['drone_img_desc']
            sate_img_dir = pair_drone2sate['sate_img_dir']
            # Training with Positive-only data /or/ Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']
            pair_sate_img_desc_list = pair_drone2sate[f'pair_{mode}_sate_img_desc_list']
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
            drone_lidar_file = os.path.join(data_root, drone_lidar_dir, drone_lidar_name)
            drone_depth_file = os.path.join(data_root, drone_depth_dir, drone_depth_name)

            for pair_sate_img, pair_sate_weight, sate_img_desc in zip(pair_sate_img_list, pair_sate_weight_list, pair_sate_img_desc_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, drone_lidar_file, drone_depth_file, drone_img_desc,
                                    sate_img_file, sate_img_desc, pair_sate_weight))

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

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        drone_img_path, drone_lidar_path, drone_depth_path, drone_img_desc, \
            satellite_img_path, satellite_img_desc, positive_weight = self.samples[index]
        
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

        ## TODO point cloud error
        drone_lidar = drone_lidar[:, [1, 2, 0]]
        drone_lidar[:, 0] = -drone_lidar[:, 0]
        drone_lidar[:, 1] = -drone_lidar[:, 1]
        drone_lidar[:, 2] = -drone_lidar[:, 2]

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
                max_length=300, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
        drone_img_desc = {k: v.squeeze() for k, v in drone_img_desc.items()}

        satellite_img_desc = self.tokenizer(
                [satellite_img_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=300, # 固定最大长度
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
                
                drone_img, drone_lidar, drone_depth, drone_img_desc, sate_img, sate_desc, _ = pair
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
        

class GTAMMDatasetEval(Dataset):
    
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
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
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
                    self.drone_loc_xys.append((drone_loc_x_y[0], drone_loc_x_y[1]))
        
        elif view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.satellite_loc_xys.append(sate2loc(tile_zoom, offset, tile_x, tile_y))
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.satellite_loc_xys.append(sate2loc(tile_zoom, offset, tile_x, tile_y))

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
            lidar[:, 1] = -lidar[:, 1]
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
                max_length=300, # 固定最大长度
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
            sate_desc = self.satellite_img_desc[index]
            sate_desc = self.tokenizer(
                [sate_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=300, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            sample["satellite_desc"] = {k: v.squeeze() for k, v in sate_desc.items()}

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


class GTAMMDatasetEvalUni(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=None,
                 tokenizer="openai/clip-vit-base-patch32",
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)
        sate_txt_file_path = f'/home/xmuairmud/jyx/daily_scripts/GTA-UAV-MM-satellite-description-qwen-vlplus.txt' 

        self.drone_img_paths = []
        self.drone_lidar_paths = []
        self.drone_depth_paths = []
        self.drone_img_desc = []
        self.drone_img_names = []
        self.drone_loc_xys = []
        self.satellite_img_paths = []
        self.satellite_img_names = []
        self.satellite_loc_xys = []
        self.satellite_img_desc = []

        self.pairs_sate2drone_dict = {}
        self.pairs_drone2sate_dict = {}
        self.pairs_match_set = set()

        self.view = view

        if view == 'drone_img':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_lidar_name = pair_drone2sate['drone_lidar_name']
                drone_lidar_dir = pair_drone2sate['drone_lidar_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.drone_img_paths.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.drone_lidar_paths.append(os.path.join(data_root, drone_lidar_dir, drone_lidar_name))
                    self.drone_img_names.append(drone_img_name)
                    self.drone_loc_xys.append((drone_loc_x_y[0], drone_loc_x_y[1]))
        
        elif view == 'drone_pc':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_lidar_name = pair_drone2sate['drone_lidar_name']
                drone_lidar_dir = pair_drone2sate['drone_lidar_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.drone_img_paths.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.drone_lidar_paths.append(os.path.join(data_root, drone_lidar_dir, drone_lidar_name))
                    self.drone_img_names.append(drone_img_name)
                    self.drone_loc_xys.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif view == 'drone_depth':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_dir = pair_drone2sate['drone_img_dir']
                drone_depth_name = pair_drone2sate['drone_depth_name']
                drone_depth_dir = pair_drone2sate['drone_depth_dir']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.drone_depth_paths.append(os.path.join(data_root, drone_depth_dir, drone_depth_name))
                    self.drone_img_names.append(drone_img_name)
                    self.drone_loc_xys.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif view == 'drone_desc':
            for pair_drone2sate in pairs_meta_data:
                drone_img_name = pair_drone2sate['drone_img_name']
                drone_img_desc = pair_drone2sate['drone_img_desc']
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.drone_img_desc.append(drone_img_desc)
                    self.drone_img_names.append(drone_img_name)
                    self.drone_loc_xys.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif view == 'sate_img':
            if "2SImg" in query_mode:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.satellite_loc_xys.append(sate2loc(tile_zoom, offset, tile_x, tile_y))
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.satellite_img_names.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.satellite_loc_xys.append(sate2loc(tile_zoom, offset, tile_x, tile_y))

        elif view == 'sate_desc':
            sate_descriptions = {}
            with open(sate_txt_file_path, 'r') as txt_file:
                for line in txt_file:
                    parts = line.split(".png, ")
                    # print(parts)
                    if len(parts) == 2:
                        # print(parts)
                        image_name = parts[0].strip()
                        description = parts[1].strip().rstrip("]\n").replace("'", "")
                        sate_descriptions[image_name] = description

            sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
            for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                sate_img_name = sate_img.replace(".png", "")
                self.satellite_img_desc.append(sate_descriptions.get(sate_img_name, ""))
                self.satellite_img_paths.append(os.path.join(data_root, sate_img_dir, sate_img))
                self.satellite_img_names.append(sate_img)

                tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                tile_zoom = int(tile_zoom)
                tile_x = int(tile_x)
                tile_y = int(tile_y)
                offset = int(offset)
                self.satellite_loc_xys.append(sate2loc(tile_zoom, offset, tile_x, tile_y))


        self.transforms = transforms
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    def __getitem__(self, index):
        if self.view == 'drone_img':
            img_path = self.drone_img_paths[index]
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image transforms
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            
            sample = {
                "drone_img": img,
            }

        elif self.view == 'drone_pc':
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
            lidar[:, 1] = -lidar[:, 1]
            lidar[:, 2] = -lidar[:, 2]
            
            lidar = self.pc_norm(lidar)
            lidar = torch.from_numpy(lidar)

            sample = {
                "drone_lidar_pts": lidar,
                "drone_lidar_clr": torch.ones_like(lidar).float() * 0.4,
            }

        elif self.view == 'drone_depth':
            depth_path = self.drone_depth_paths[index]

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=2)
            depth = (depth / 256).astype(np.uint8)

            depth = self.transforms(image=depth)['image']

            sample = {
                "drone_depth": depth,
            }

        elif self.view == 'drone_desc':
            drone_desc = self.drone_img_desc[index]
            drone_desc = self.tokenizer(
                [drone_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=300, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            sample = {
                "drone_desc": {k: v.squeeze() for k, v in drone_desc.items()}
            }

        elif self.view == 'sate_img':
            img_path = self.satellite_img_paths[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image transforms
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            
            sample = {
                "satellite_img": img,
            }

        elif self.view == 'sate_desc':
            sate_desc = self.satellite_img_desc[index]
            sate_desc = self.tokenizer(
                [sate_desc],
                padding="max_length",  # 短文本自动填充
                truncation=True,       # 长文本自动截断
                max_length=300, # 固定最大长度
                return_tensors="pt"    # 返回 PyTorch 张量
            )
            sample = {
                "satellite_desc": {k: v.squeeze() for k, v in sate_desc.items()}
            }

        return sample

    def __len__(self):
        if self.view == 'drone_img':
            length = len(self.drone_img_names)
        elif self.view == 'drone_pc':
            length = len(self.drone_img_names)
        elif self.view == 'drone_depth':
            length = len(self.drone_img_names)
        elif self.view == 'drone_desc':
            length = len(self.drone_img_names)
        elif self.view == 'sate_img':
            length = len(self.satellite_img_names)
        elif self.view == 'sate_desc':
            length = len(self.satellite_img_names)
        return length
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


class PoissonNoise(A.ImageOnlyTransform):
    def __init__(self, scale=1.0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.scale = scale  # 控制噪声强度

    def apply(self, img, **params):
        img = img.astype(np.float32)
        noise = np.random.poisson(img / 255.0 * self.scale) / self.scale * 255  # 生成泊松噪声
        img = img + noise  # 添加噪声
        return np.clip(img, 0, 255).astype(np.uint8)  # 限制像素范围


class SaltAndPepperNoise(A.ImageOnlyTransform):
    def __init__(self, amount=0.02, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.amount = amount  # 椒盐噪声比例

    def apply(self, img, **params):
        img = img.copy()
        num_salt = int(self.amount * img.size * 0.5)  # 计算白色（盐）噪声点数
        num_pepper = int(self.amount * img.size * 0.5)  # 计算黑色（椒）噪声点数

        # 添加白色噪声（盐）
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        img[coords[0], coords[1], :] = 255

        # 添加黑色噪声（椒）
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
        img[coords[0], coords[1], :] = 0

        return img


class SpeckleNoise(A.ImageOnlyTransform):
    def __init__(self, mean=0, std=0.1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        noise = np.random.normal(self.mean, self.std, img.shape)  # 生成高斯噪声
        img = img + img * noise  # 计算 I' = I + I * N
        return np.clip(img, 0, 255).astype(np.uint8)  # 限制像素范围


class Pixelization(A.ImageOnlyTransform):
    def __init__(self, ratio=0.1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.ratio = ratio  # 像素化比例（0.1 表示缩小到 10% 大小）

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.ratio), int(w * self.ratio)  # 计算缩小尺寸
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # 缩小
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)  # 放大回去（像素化）
        return img


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
                                            # A.GridDropout(ratio=0.8, p=1.0),
                                            # A.CoarseDropout(max_holes=1,
                                            #                    max_height=int(0.7*img_size[0]),
                                            #                    max_width=int(0.7*img_size[0]),
                                            #                    min_holes=1,
                                            #                    min_height=int(0.7*img_size[0]),
                                            #                    min_width=int(0.7*img_size[0]),
                                            #                    p=1.0),
                                            SaltAndPepperNoise(amount=0.02, p=1.0),
                                            # Pixelization(ratio=0.2),
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


if __name__ == "__main__":
    pass