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
import torch


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


class GTARGBDDatasetTrain(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 transforms_query_geo=None,
                 transforms_query_rgb=None,
                 transforms_query_depth=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='pos_semipos',
                 train_ratio=1.0,
                 prob_drop_depth=0.0,
                 prob_drop_rgb=0.0,
                 ):
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
            sate_img_dir = pair_drone2sate['sate_img_dir']
            # Training with Positive-only data or Positive+Semi-positive data
            pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
            pair_sate_weight_list = pair_drone2sate[f'pair_{mode}_sate_weight_list']
            
            drone_img_file = os.path.join(data_root, drone_img_dir, drone_img_name)
            drone_depth_file = os.path.join(data_root, drone_depth_dir, drone_depth_name)

            for pair_sate_img, pair_sate_weight in zip(pair_sate_img_list, pair_sate_weight_list):
                sate_img_file = os.path.join(data_root, sate_img_dir, pair_sate_img)
                self.pairs.append((drone_img_file, drone_depth_file, sate_img_file, pair_sate_weight))

            # Build Graph with All Edges (drone, sate)
            pair_all_sate_img_list = pair_drone2sate['pair_pos_semipos_sate_img_list']
            for pair_sate_img in pair_all_sate_img_list:
                self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                self.pairs_match_set.add((drone_img_name, pair_sate_img))

        self.transforms_query_geo = transforms_query_geo
        self.transforms_query_rgb = transforms_query_rgb
        self.transforms_query_depth = transforms_query_depth
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.prob_drop_depth = prob_drop_depth
        self.prob_drop_rgb = prob_drop_rgb

        # Training with sparse data
        num_pairs = len(self.pairs)
        num_pairs_train = int(train_ratio * num_pairs)
        random.shuffle(self.pairs)
        self.pairs = self.pairs[:num_pairs_train]
        
        self.samples = copy.deepcopy(self.pairs)
    
    def __getitem__(self, index):
        
        query_img_path, query_depth_path, gallery_img_path, positive_weight = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)

        query_depth = (query_depth / 256).astype(np.uint8)
        if len(query_depth.shape) == 2:
                query_depth = np.expand_dims(query_depth, axis=2)
        
        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            query_depth = cv2.flip(query_depth, 1)
        
        # image transforms
        if self.transforms_query_geo is not None:
            image_rgb = query_img
            image_d = query_depth
            transformed = self.transforms_query_geo(image=image_rgb, mask=image_d)
            image_rgb_transformed = self.transforms_query_rgb(image=transformed['image'])['image']
            image_d_transformed = self.transforms_query_depth(image=transformed['mask'])['image']

            if image_d_transformed.ndim == 2:
                image_d_transformed = image_d_transformed.unsqueeze(0)
            query_img = torch.cat((image_rgb_transformed, image_d_transformed), dim=0)  # 形状为 (4, H, W)

        drop_flag = False
        if np.random.random() < self.prob_drop_depth:
            zero_d = torch.zeros_like(query_img[3:, :, :])
            query_img[3, :, :] = zero_d
            drop_flag = True

        if np.random.random() < self.prob_drop_rgb and not drop_flag:
            zero_rgb = torch.zeros_like(query_img[:3, :, :])
            query_img[:3, :, :] = zero_rgb
            
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']
        
        return query_img, gallery_img, positive_weight    # , query_loc_xy/NORM_LOC, gallery_loc_xy/NORM_LOC

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
                
                drone_img, _, sate_img, _ = pair
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
        print("First RGB Element: {} - Last Element: {}".format(self.samples[0][0], self.samples[-1][0]))  
        print("First Depth Element: {} - Last Element: {}".format(self.samples[0][1], self.samples[-1][1]))  
        print("First Satellite Element: {} - Last Element: {}".format(self.samples[0][2], self.samples[-1][2]))  
        

class GTARGBDDatasetEval(Dataset):
    
    def __init__(self,
                 pairs_meta_file,
                 data_root,
                 view,
                 mode='pos',
                 sate_img_dir='',
                 query_mode='D2S',
                 pairs_sate2drone_dict=None,
                 transforms_rgb=None,
                 transforms_depth=None,
                 ):
        super().__init__()
        
        with open(os.path.join(data_root, pairs_meta_file), 'r', encoding='utf-8') as f:
            pairs_meta_data = json.load(f)
        self.data_root = data_root
        sate_img_dir = os.path.join(data_root, sate_img_dir)    

        self.images_path = []
        self.images_name = []
        self.depth_path = []
        self.images_loc_xy = []

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
                drone_loc_x_y = pair_drone2sate['drone_loc_x_y']
                self.pairs_drone2sate_dict[drone_img_name] = []
                pair_sate_img_list = pair_drone2sate[f'pair_{mode}_sate_img_list']
                for pair_sate_img in pair_sate_img_list:
                    self.pairs_drone2sate_dict.setdefault(drone_img_name, []).append(pair_sate_img)
                    self.pairs_sate2drone_dict.setdefault(pair_sate_img, []).append(drone_img_name)
                    self.pairs_match_set.add((drone_img_name, pair_sate_img))
                if len(pair_sate_img_list) != 0:
                    self.images_path.append(os.path.join(data_root, drone_img_dir, drone_img_name))
                    self.depth_path.append(os.path.join(data_root, drone_depth_dir, drone_depth_name))
                    self.images_name.append(drone_img_name)
                    self.images_loc_xy.append((drone_loc_x_y[0], drone_loc_x_y[1]))

        elif view == 'sate':
            if query_mode == 'D2S':
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.images_loc_xy.append(sate2loc(tile_zoom, offset, tile_x, tile_y))
            else:
                sate_img_dir_list, sate_img_list = get_sate_data(sate_img_dir)
                for sate_img_dir, sate_img in zip(sate_img_dir_list, sate_img_list):
                    if sate_img not in pairs_sate2drone_dict.keys():
                        continue
                    self.images_path.append(os.path.join(data_root, sate_img_dir, sate_img))
                    self.images_name.append(sate_img)

                    sate_img_name = sate_img.replace('.png', '')
                    tile_zoom, offset, tile_x, tile_y = sate_img_name.split('_')
                    tile_zoom = int(tile_zoom)
                    tile_x = int(tile_x)
                    tile_y = int(tile_y)
                    offset = int(offset)
                    self.images_loc_xy.append(sate2loc(tile_zoom, offset, tile_x, tile_y))

        self.transforms_rgb = transforms_rgb
        self.transforms_depth = transforms_depth

    def __getitem__(self, index):
        
        if self.view == 'drone':
            img_path = self.images_path[index]
            depth_path = self.depth_path[index]

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=2)
            depth = (depth / 256).astype(np.uint8)

            img = self.transforms_rgb(image=img)['image']
            depth = self.transforms_depth(image=depth)['image']
            img = torch.cat((img, depth), dim=0)

        else:
            img_path = self.images_path[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transforms_rgb(image=img)['image']
        
        return img

    def __len__(self):
        return len(self.images_name)
    
    
def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):

    mean_d = [0.5]
    std_d = [0.5]
    mean_rgbd = mean + mean_d
    std_rgbd = std + std_d
    

    val_sat_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])
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
                                            # A.RandomRotate90(p=1.0),
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
    _, _, _, transforms_geo, transforms_rgb, transforms_d = get_transforms(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], img_size=(384, 384))

    x = np.random.rand(500, 500, 4) * 255
    x = x.astype(np.uint8)
    print(x.dtype)
    rgb = x[:, :, :3]
    d = x[:, :, 3:]
    print(rgb.shape, d.shape)
    a = transforms_geo(image=rgb, mask=d)
    rgb = a['image']
    d = a['mask']

    rgb = transforms_rgb(image=rgb)['image']
    d = transforms_d(image=d)['image']


    print(rgb.max(), rgb.min())
    print(d.max(), d.min())

    # rgbd = cv2.imread('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/rgbd/200_0001_0000001542.png', cv2.IMREAD_UNCHANGED)

    # rgbd_trans = transforms_rgbd(image=rgbd)['image']

    # rgb = rgbd[:, :, :3]
    # d = rgbd[:, :, 3]

    # d_resize = cv2.resize(d, (384, 384), cv2.INTER_NEAREST)

    # d_color = cv2.applyColorMap(d_resize, cv2.COLORMAP_JET)
    # cv2.imwrite('vis_d_ori.png', d_color)


    # image_rgb_transformed = transforms_rgb(image=rgb)['image']
    # image_d_transformed = transforms_depth(image=d)['image']

    # image_d_transformed_color = cv2.applyColorMap(image_d_transformed, cv2.COLORMAP_JET)

    # cv2.imwrite('vis_rgb.png', image_rgb_transformed)
    # cv2.imwrite('vis_d.png', image_d_transformed_color)


    