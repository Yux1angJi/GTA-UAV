import faiss
import numpy as np
import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.university import get_transforms
from game4loc.dataset.custom_query import CustomData
from game4loc.models.model import DesModel
from torch.cuda.amp import autocast
# from game4loc.evaluate.query_topn import QueryTopN
import torch.nn.functional as F

import matplotlib.pyplot as plt

import cv2

import pickle
import time
from tqdm import tqdm
import numpy as np


def geo_to_pixel(lon, lat, top_left_geo, bottom_right_geo, img_width, img_height):
    """
    将地理坐标转换为像素坐标。
    
    参数:
    - lon, lat: 预测点的经度和纬度。
    - top_left_geo: 原图左上角的经纬度（经度，纬度）。
    - bottom_right_geo: 原图右下角的经纬度（经度，纬度）。
    - img_width, img_height: 原图的宽度和高度（像素）。
    
    返回:
    - (x, y): 预测点在图像上的像素坐标。
    """
    lat_max, lon_min = top_left_geo
    lat_min, lon_max = bottom_right_geo
    
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    x = (lon - lon_min) * img_width / lon_range
    y = (lat_max - lat) * img_height / lat_range
    
    return int(x), int(y)


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []

    with torch.no_grad():
        
        for img in bar:
        
            with autocast():
         
                img = img.to(train_config.device)
                img_feature = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32).cpu().numpy())

        # keep Features on GPU
        img_features = np.concatenate(img_features_list, axis=0)
        # print(img_features.shape)
        
    if train_config.verbose:
        bar.close()
    
    return img_features


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
    # checkpoint_start = '/home/xmuairmud/jyx/ExtenGeo/Sample4Geo/work_dir/university/convnext_base.fb_in22k_ft_in1k_384/0629220257/weights_end.pth'
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
 
config.query_folder_test = '/home/xmuairmud/data/work/map_data/map2/query_drone' 
# config.gallery_folder_test = '/home/xmuairmud/data/work/map_data/map2/18'  


def queryTopN(query_vectors, query_paths, zoom_list=[17], top_n=10):
    d = 1024  # 维度
    # nb = 100000  # 数据库大小
    # np.random.seed(1234)
    # db_vectors = np.random.random((nb, d)).astype('float32')

    db_vectors = np.empty((0, d))
    db_latlons = np.empty((0, 2))
    db_xys = np.empty((0, 2))
    db_paths = []

    pkl_path = '/home/xmuairmud/data/work/map_data/map2/sample4geo_feature/gta/from_uni_woweight/'
    # pkl_path = '/home/xmuairmud/data/work/map_data/map2/sample4geo_feature/university/extend_des_dse/'

    for zoom in zoom_list:
        pkl_name = pkl_path + f'{zoom}.pkl'
        with open(pkl_name, "rb") as f:
            data = pickle.load(f)
            features = data['features']
            latlons = data['latlons']
            xys = data['xys']
            paths = data['paths']
            db_paths.extend(paths)
            db_vectors = np.concatenate((db_vectors, features), axis=0)
            db_latlons = np.concatenate((db_latlons, latlons), axis=0)
            db_xys = np.concatenate((db_xys, xys), axis=0)
    
    db_vectors = torch.from_numpy(db_vectors).to(dtype=torch.float32).cuda()
    query_features = torch.from_numpy(query_vectors).to(dtype=torch.float32).cuda()

    # print(query_features.dtype, db_vectors.dtype)

    results_xy = []
    results_path = []

    for i in range(len(query_features)):
        query_feature_i = query_features[i]
        # print('query shape', query_feature_i.shape)

        similarity = torch.matmul(db_vectors, query_feature_i)

        sorted_values, sorted_indices = torch.sort(similarity.squeeze(), descending=True)
        top_indices = sorted_indices[:top_n].cpu().numpy()

        result_path_tmp = [query_paths[i]]
        result_xy_tmp = []
        for i in top_indices:
            result_xy_tmp.append((db_xys[i][0], db_xys[i][1]))
            result_path_tmp.append(db_paths[i])
        results_xy.append(result_xy_tmp)
        results_path.append(result_path_tmp)
    # print(results_xy)
    return results_path


def visualize_from_file(ori_img, top_left_geo, bottom_right_geo, predict_list, file_name):
    img = cv2.imread(ori_img)
    img_height, img_width = img.shape[:2]

    for lat, lon in predict_list:
        x, y = geo_to_pixel(lon, lat, top_left_geo, bottom_right_geo, img_width, img_height)
        cv2.circle(img, (x, y), radius=10, color=(0,255,0), thickness=-1)
    cv2.imwrite(f'vis_{file_name}.png', img)


def query_plot(zoom_list):
    results = queryTopN(query_vectors, query_path_list, zoom_list)

    rows = len(results)
    fig, axes = plt.subplots(nrows=rows, ncols=11, figsize=(20, 4))
    for i, result in enumerate(results):
        size = 256, 256

        query_img = result[0]
        query_img = cv2.imread(query_img)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_img = cv2.resize(query_img, size)
        
        ax = axes[i, 0]
        ax.imshow(query_img, cmap='gray')
        ax.set_title("Query")
        ax.axis('off')

        for j in range(10):
            ax = axes[i, j+1]
            topj_img = result[j+1]
            print(topj_img)
            topj_img = cv2.imread(topj_img)
            topj_img = cv2.cvtColor(topj_img, cv2.COLOR_BGR2RGB)
            topj_img = cv2.resize(topj_img, size)
            ax.imshow(topj_img, cmap='gray')
            ax.set_title(f"Top {j+1}")
            ax.axis('off')
    fig.set_dpi(100)
    # plt.savefig(f'demo/demo_{zoom_list}.png')
    plt.savefig(f'demo/demo_{zoom_list}_gta.png')


if __name__ == '__main__':

    print("\nModel: {}".format(config.model))


    model = DesModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
    
    img_size = (config.img_size, config.img_size)
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]

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
    
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    
    query_dataset = CustomData(root_dir=config.query_folder_test,
                                               transforms=val_transforms,
                                               )
    query_path_list = query_dataset.imgs_list
    
    query_dataloader = DataLoader(query_dataset,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True) 
    
    query_vectors = predict(config, model, query_dataloader)
    
    # query_vectors = query_vectors.astype('float32')
    # query_vectors = np.ascontiguousarray(query_vectors)  # 确保数据是 C-连续的
    # print(query_vectors.shape)

    # res_latlon = faiss_gpu_search(query_vectors)
    # print(res_latlon)

    # ori_img = '/home/xmuairmud/data/work/map_data/map2/ori_img.png'
    # top_left_geo = 24.592190, 117.912738
    # bottom_right_geo = 24.156363, 118.238395
    # for i in range(res_latlon.shape[0]):
    #     visualize_from_file(ori_img, top_left_geo, bottom_right_geo, res_latlon[i], f'{i}')

    for zoom_list in [[16], [17], [18], [17, 18], [16, 17], [16, 17, 18]]:
        query_plot(zoom_list)

