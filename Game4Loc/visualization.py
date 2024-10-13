import torch
import timm
import os
import cv2
import numpy as np
import pickle

from game4loc.models.model import DesModel
from game4loc.dataset.gta import GTADatasetEval, get_transforms

def query_match(drone_img):
    pickle_dir = '/home/xmuairmud/data/GTA-UAV-data/randcam2_std0_stable_all/same_h23456_z41_iou4_oc4'
    with open(os.path.join(pickle_dir, 'train_pair_meta.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(pickle_dir, 'test_pair_meta.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    for drone2sate in train_data['pairs_drone2sate_list']:
        if drone2sate['drone_img'] == drone_img:
            print(drone2sate)
    for drone2sate in test_data['pairs_drone2sate_list']:
        if drone2sate['drone_img'] == drone_img:
            print(drone2sate)


def visualization():
    model = DesModel(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
    checkpoint_start = 'work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0722010320/weights_end.pth'
    model_state_dict = torch.load(checkpoint_start)  
    model.load_state_dict(model_state_dict, strict=False) 

    model.cuda()
    model.eval()   

    img_size = (384, 384)
    data_config = model.get_config()
    # print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)

    img_dir = '/home/xmuairmud/data/visualization'
    img_list = [
        # '0000_0000001344.png',
        # '0000_0000002647.png',
        # '0000_0000002988.png',
        # '0000_0000003026.png'

        '0000_0000002988.png',
        '6_21_37.png',
        '6_21_38.png',
        '6_22_37.png',
        '6_22_38.png',

        '0000_0000003026.png',
        '5_11_25.png',
        '6_23_51.png',
        '6_22_51.png',
    ]


    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = val_transforms(image=x)['image']

        x = x.cuda()
        x = x.unsqueeze(0)
        with torch.no_grad():
            features = model.model.forward_features(x)[0, 1:]
        print(features.shape)
        patch_size = int(np.sqrt(features.shape[0]))
        features = features.view(patch_size, patch_size, -1)

        heatmap = features.mean(dim=2).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化
        heatmap = np.sqrt(heatmap)  # 使用平方根变换
        heatmap = cv2.resize(heatmap, (w, h))  # 调整大小
        heatmap = np.uint8(255 * heatmap)  # 转换为0-255范围的整数
        equalized_heatmap = cv2.equalizeHist(heatmap)  # 应用直方图均衡化
        heatmap = cv2.applyColorMap(equalized_heatmap, cv2.COLORMAP_JET)  # 应用颜色映射

        superimposed_img = heatmap*0.2+img

        cv2.imwrite(os.path.join(img_dir, f"feature_{img_name}"), superimposed_img)

if __name__ == '__main__':
    # visualization()

    drone_img = '300_0000_0000003026.png'
    query_match(drone_img)

