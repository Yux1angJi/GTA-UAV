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
                 transforms_drone=None,
                 transforms_satellite=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 group_len=2,
                 augment_pc=True):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        self.group_len = group_len
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
            sate_img_dir = pair_drone2sate['sate_img_dir']
            # Training with Positive-only data /or/ Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
            drone_lidar_file = os.path.join(data_root, drone_lidar_dir, drone_lidar_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, drone_lidar_file, sate_img_file, pair_sate_weight))

            # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate['pair_pos_semipos_sate_img_list']
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.transforms_drone = transforms_drone
        self.transforms_satellite = transforms_satellite
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        drone_img_path, drone_lidar_path, satellite_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        drone_img = cv2.imread(drone_img_path)
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        drone_lidar = o3d.io.read_point_cloud(drone_lidar_path)
        drone_lidar = np.array(drone_lidar.points)
        drone_lidar = farthest_point_sample(drone_lidar, 10000)
        drone_lidar = torch.from_numpy(drone_lidar)
        if self.augment_pc:
            drone_lidar = random_scale_point_cloud(drone_lidar[None, ...])
            drone_lidar = shift_point_cloud(drone_lidar)
            drone_lidar = rotate_perturbation_point_cloud(drone_lidar)
            drone_lidar = rotate_point_cloud(drone_lidar)
            drone_lidar = drone_lidar.squeeze()
            drone_lidar = torch.from_numpy(drone_lidar)
        
        satellite_img = cv2.imread(satellite_img_path)
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            satellite_img = cv2.flip(satellite_img, 1) 
        
        # image transforms
        if self.transforms_drone is not None:
            drone_img = self.transforms_drone(image=drone_img)['image']
            
        if self.transforms_satellite is not None:
            satellite_img = self.transforms_satellite(image=satellite_img)['image']
        
        sample =  {
            "drone_img": drone_img, 
            "drone_lidar_pts": drone_lidar,
            "drone_lidar_clr": torch.ones_like(drone_lidar).float() * 0.4, 
            "satellite_img": satellite_img, 
            "positive_weight": positive_weight,
        }

        return sample

    def __len__(self):
        return len(self.samples)
    
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
                
                drone_img, drone_lidar, sate_img, _ = pair
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
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms=None,
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)    

        self.drone_img_paths = []
        self.drone_lidar_paths = []
        self.drone_img_names = []
        self.drone_loc_xys = []
        self.satellite_img_paths = []
        self.satellite_img_names = []
        self.satellite_loc_xys = []
        

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
        
        elif view == 'drone_lidar':
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

        elif view == 'sate_img':
            if query_mode == 'DImg2SImg':
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

        self.transforms = transforms

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

        elif self.view == 'drone_lidar':
            lidar_path = self.drone_lidar_paths[index]
            
            lidar = o3d.io.read_point_cloud(lidar_path)
            lidar = np.array(lidar.points)
            lidar = farthest_point_sample(lidar, 10000)
            lidar = torch.from_numpy(lidar)
            
            lidar = self.pc_norm(lidar)

            sample = {
                "drone_lidar_pts": lidar,
                "drone_lidar_clr": torch.ones_like(lidar).float() * 0.4,
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

        return sample

    def __len__(self):
        if self.view == 'drone_img':
            length = len(self.drone_img_names)
        elif self.view == 'drone_lidar':
            length = len(self.drone_img_names)
        elif self.view == 'sate_img':
            length = len(self.satellite_img_names)
        return length
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    
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


if __name__ == "__main__":
    pass