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
import math


def get_all_file_paths(root_dir):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 创建文件的绝对路径
            file_path = os.path.join(dirpath, filename)
            all_files.append(file_path)
    return all_files


def tile_to_lat_lon(x, y, zoom):
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lat_rad_top = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_top = math.degrees(lat_rad_top)
    
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_rad_bottom = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_bottom = math.degrees(lat_rad_bottom)
    
    return (lat_top, lon_left), (lat_bottom, lon_right)


class CustomData(Dataset):
    
    def __init__(self,
                 transforms,
                 root_dir,
                 ):
        super().__init__()

        self.transforms = transforms
        self.imgs_list = get_all_file_paths(root_dir)
        # print(self.imgs_list)

    def __getitem__(self, index):
        
        img_path = self.imgs_list[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img

    def __len__(self):
        return len(self.imgs_list)


class CustomDataWithLatlon(Dataset):
    
    def __init__(self,
                 transforms,
                 root_dir,
                 zoom=15,
                 ):
        super().__init__()

        self.transforms = transforms
        self.imgs_list = get_all_file_paths(root_dir)
        self.zoom = zoom   
                    
    def __getitem__(self, index):
        
        img_path = self.imgs_list[index]
        img_name = img_path.split('/')[-1].replace('.jpg', '')
        # print(img_name)

        x, y = int(img_name.split('_')[0]), int(img_name.split('_')[1])
        # print(x, y)
        (lat_top, lon_left), (lat_bottom, lon_right) = tile_to_lat_lon(x, y, self.zoom)
        lat = (lat_top + lat_bottom) / 2.
        lon = (lon_right + lon_left) / 2.
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, (lat, lon), (x, y)

    def __len__(self):
        return len(self.imgs_list)
    
