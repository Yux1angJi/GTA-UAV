import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import get_transforms
from sample4geo.dataset.custom_query import CustomData, CustomDataWithLatlon
from sample4geo.model import TimmModel
from torch.cuda.amp import autocast
from sample4geo.evaluate.query_topn import QueryTopN
import torch.nn.functional as F

import matplotlib.pyplot as plt

import cv2

import pickle
import time
from tqdm import tqdm
import numpy as np


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    img_latlons_list = []
    img_xy_list = []

    with torch.no_grad():
        
        for img, (lat, lon), (x, y) in bar:
        
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32).cpu().numpy())
            latlon = np.column_stack((lat.to(torch.float32).cpu().numpy(), lon.to(torch.float32).cpu().numpy()))
            xy = np.column_stack((x.cpu().numpy(), y.cpu().numpy()))
            img_latlons_list.append(latlon)
            img_xy_list.append(xy)

        # keep Features on GPU
        img_features = np.concatenate(img_features_list, axis=0)
        img_latlons = np.concatenate(img_latlons_list, axis=0)
        img_xys = np.concatenate(img_xy_list, axis=0)
        print(img_features.shape, img_latlons.shape, img_xys.shape)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, img_latlons, img_xys


@dataclass
class Configuration:

    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    dataset: str = 'U1652-D2S'           # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"
    
    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    # checkpoint_start = '/home/xmuairmud/jyx/ExtenGeo/Sample4Geo/university/convnext_base.fb_in22k_ft_in1k_384/0629220257/weights_end.pth'
    checkpoint_start = '/home/xmuairmud/jyx/ExtenGeo/Sample4Geo/work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0703163147/weights_end.pth'

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

# if config.dataset == 'U1652-D2S':
#     config.query_folder_train = './data/U1652/train/satellite'
#     config.gallery_folder_train = './data/U1652/train/drone'   
#     config.query_folder_test = './data/U1652/test/query_drone' 
#     config.gallery_folder_test = './data/U1652/test/gallery_satellite'    
# elif config.dataset == 'U1652-S2D':
#     config.query_folder_train = './data/U1652/train/satellite'
#     config.gallery_folder_train = './data/U1652/train/drone'    
#     config.query_folder_test = './data/U1652/test/query_satellite'
#     config.gallery_folder_test = './data/U1652/test/gallery_drone'
 
# config.query_folder_test = '/home/xmuairmud/data/work/map_data/map2/query_drone' 
  


def gen_feature(zoom_level):

    config.gallery_folder_test = f'/home/xmuairmud/data/work/map_data/map2/{zoom_level}'  

    gallery_dataset = CustomDataWithLatlon(root_dir=config.gallery_folder_test,
                                               transforms=val_transforms,
                                               zoom=zoom_level,
                                               )
    gallery_path_list = gallery_dataset.imgs_list
    
    gallery_dataloader = DataLoader(gallery_dataset,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)                                                                                                               
    
    features, latlons, xys = predict(config, model, gallery_dataloader)

    # pkl_save_dir = f'/home/xmuairmud/data/work/map_data/map2/sample4geo_feature/baseline/{zoom_level}.pkl'
    # pkl_save_dir = f'/home/xmuairmud/data/work/map_data/map2/sample4geo_feature/extend_des_dse/'
    pkl_save_dir = f'/home/xmuairmud/data/work/map_data/map2/sample4geo_feature/gta/from_uni_woweight/'
    
    if not os.path.exists(pkl_save_dir):
        os.makedirs(pkl_save_dir)

    pkl_save_dir += f'{zoom_level}.pkl'
    
    with open(pkl_save_dir, "wb") as f:
        pickle.dump({"features": features, "latlons": latlons, "xys": xys, "paths": gallery_path_list}, f) 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    

    # load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Query:", img_size)
    print("Image Size Ground:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    
    for zoom_level in [15, 16, 17, 18]:
        gen_feature(zoom_level)

