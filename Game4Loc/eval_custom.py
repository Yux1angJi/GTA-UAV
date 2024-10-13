import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from game4loc.dataset.gta import get_transforms
from game4loc.dataset.custom_query import CustomData
from game4loc.models.model import DesModel

from game4loc.evaluate.query_topn import QueryTopN

import matplotlib.pyplot as plt

import cv2
import numpy as np


GAME_TO_SATE_KX = 1.8206
GAME_TO_SATE_BX = 7539.39
GAME_TO_SATE_KY = -1.8220
GAME_TO_SATE_BY = 15287.16
SATE_LENGTH = 24576

def sate2loc(tile_zoom, tile_x, tile_y):
    tile_pix = SATE_LENGTH / (2 ** tile_zoom)
    loc_x = (tile_pix * (tile_x+1/2)) * 0.45
    loc_y = (tile_pix * (tile_y+1/2)) * 0.45
    return loc_x, loc_y

def game_pos2loc(game_pos_x, game_pos_y):
    sate_pos_x = game_pos_x * GAME_TO_SATE_KX + GAME_TO_SATE_BX
    sate_pos_y = game_pos_y * GAME_TO_SATE_KY + GAME_TO_SATE_BY
    return sate_pos_x*0.45, sate_pos_y*0.45

@dataclass
class Configuration:

    # Model
    model: str = 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    # Dataset
    
    # Checkpoint to start from
    # checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
    checkpoint_start = 'work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0905033254/weights_end.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


# config.query_folder_test = '/home/xmuairmud/data/work/map_data/map2/query_drone' 
# config.gallery_folder_test = '/home/xmuairmud/data/work/map_data/map2/gallery_satellite'    
config.query_folder_test = '/home/xmuairmud/data/visual_test/visual_test_300/images' 
config.gallery_folder_test = '/home/xmuairmud/data/GTA-UAV-data/satellite_z41'   


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))


    model = DesModel(config.model,
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

    # Reference Satellite Images
    query_dataset_test = CustomData(root_dir=config.query_folder_test,
                                               transforms=val_transforms,
                                               )
    query_path_list = query_dataset_test.imgs_list
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # Query Ground Images Test
    gallery_dataset_test = CustomData(root_dir=config.gallery_folder_test,
                                               transforms=val_transforms,
                                               )
    gallery_path_list = gallery_dataset_test.imgs_list
    
    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)                                                                                                               

    results = QueryTopN(config, model, query_dataloader_test, query_path_list, gallery_dataloader_test, gallery_path_list)

    predict_sate_xy_dict = {}
    loc_sate_xy_dict = {}

    dis_sum = 0.0
    for i in range(len(results)):
        query_i = results[i][0].split('/')[-1].replace('.png', '.txt')
        num = int(query_i.replace('.txt', '').split('_')[-1])
        result_i = results[i][1].split('/')[-1].replace('.png', '')
        zoom, x, y = result_i.split('_')
        zoom = int(zoom)
        x = int(x)
        y = int(y)

        meta_file = os.path.join('/home/xmuairmud/data/visual_test/visual_test_300/meta_data', query_i)
        with open(meta_file, 'r') as file:
            line = file.readline().strip()
            values = line.split()
            loc_x = float(values[0])
            loc_y = float(values[1])
        loc_xy = game_pos2loc(loc_x, loc_y)

        predict_xy = sate2loc(zoom, x, y)

        dis = ((loc_xy[0] - predict_xy[0]) ** 2 + (loc_xy[1] - predict_xy[1]) ** 2) ** 0.5
        print(query_i, dis)
        dis_sum += dis

        predict_sate_xy_dict[num] = (predict_xy[0]/0.45/16, predict_xy[1]/0.45/16)
        loc_sate_xy_dict[num] = (loc_xy[0]/0.45/16, loc_xy[1]/0.45/16)
    print('dis avg', dis_sum/len(results))

    predict_sate_xy = []
    loc_sate_xy = []
    for i in range(1, 13):
        predict_sate_xy.append(predict_sate_xy_dict[i])
        loc_sate_xy.append(loc_sate_xy_dict[i])
    predict_sate_xy = np.array(predict_sate_xy)
    loc_sate_xy = np.array(loc_sate_xy)
    
    # 定义每对点之间插入的帧数
    frames_per_segment = 30  # 每对点之间插入10帧

    # 函数：生成插值点（线性插值）
    def interpolate_points(points, frames_per_segment):
        interpolated_points = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            for j in range(frames_per_segment):
                # 计算插值，使用 (1 - t) * start + t * end 的方式插值
                t = j / frames_per_segment
                intermediate_point = (1 - t) * start + t * end
                interpolated_points.append(intermediate_point)
        interpolated_points.append(points[-1])  # 添加最后一个点
        return np.array(interpolated_points)

    from PIL import Image
    import matplotlib.animation as animation

    # 插值后的所有点
    predict_sate_xy = interpolate_points(predict_sate_xy, frames_per_segment)
    loc_sate_xy = interpolate_points(loc_sate_xy, frames_per_segment)


    bg_image = Image.open('/home/xmuairmud/data/GTA-UAV-data/GTA-UAV-sate-resized.png')
    bg_array = np.asarray(bg_image)
    img_width, img_height = bg_image.size

    fig, ax = plt.subplots(figsize=(img_width/100, img_height/100), dpi=100)

    # 显示背景图片
    ax.imshow(bg_array)
    ax.set_axis_off()

    # 初始化一个空线条对象，用于绘制逐渐连线的动画
    line_loc, = ax.plot([], [], lw=6, color='blue', linestyle=':')
    line_pre, = ax.plot([], [], lw=6, color='red', linestyle=':') 

    # 初始化函数，设置线条的起始状态
    def init():
        line_loc.set_data([], [])
        line_pre.set_data([], [])
        return line_loc, line_pre

    # 更新函数，在每一帧中逐渐连接更多的插值点
    def update(frame):
        x_data = loc_sate_xy[:frame+1, 0]  # 取前 frame+1 个点的 x 坐标
        y_data = loc_sate_xy[:frame+1, 1]  # 取前 frame+1 个点的 y 坐标
        line_loc.set_data(x_data, y_data)  # 更新线条数据

        x_data = predict_sate_xy[:frame+1, 0]  # 取前 frame+1 个点的 x 坐标
        y_data = predict_sate_xy[:frame+1, 1]  # 取前 frame+1 个点的 y 坐标
        line_pre.set_data(x_data, y_data)  # 更新线条数据
        return line_loc, line_pre

    # 创建动画：frames 是逐帧显示的总数
    ani = animation.FuncAnimation(fig, update, frames=len(loc_sate_xy), init_func=init, blit=True)

    # 保存动画为 mp4 文件
    ani.save('line_animation.mp4', writer='ffmpeg', fps=10)

    # rows = len(results)
    # fig, axes = plt.subplots(nrows=rows, ncols=11, figsize=(20, 2))
    # for i, result in enumerate(results):
    #     print(result)
    #     size = 256, 256

    #     query_img = result[0]
    #     query_img = cv2.imread(query_img)
    #     query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    #     query_img = cv2.resize(query_img, size)
        
    #     ax = axes[i, 0]
    #     ax.imshow(query_img, cmap='gray')
    #     ax.set_title("Query")
    #     ax.axis('off')

    #     for j in range(10):
    #         ax = axes[i, j+1]
    #         topj_img = result[j+1]
    #         topj_img = cv2.imread(topj_img)
    #         topj_img = cv2.cvtColor(topj_img, cv2.COLOR_BGR2RGB)
    #         topj_img = cv2.resize(topj_img, size)
    #         ax.imshow(topj_img, cmap='gray')
    #         ax.set_title(f"Top {j+1}")
    #         ax.axis('off')
    # fig.set_dpi(100)

    # plt.tight_layout()
    # plt.savefig('demo_extend_des.png')


        

 
