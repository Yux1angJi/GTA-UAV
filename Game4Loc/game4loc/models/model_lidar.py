import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile

from .point_encoder import PointcloudEncoder


class MLP(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DesModelWithPC(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pc_model_name='eva02_base_patch14_448.mim_in22k_ft_in1k',
                 pretrained=True,
                 drop_path_rate=0.,
                 img_size=384,
                 share_weights=True,
                 train_with_offset=False,
                 ):
                 
        super(DesModelWithPC, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                self.img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.drone_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.satellite_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.drone_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.satellite_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        point_transformer = timm.create_model(pc_model_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
        self.drone_lidar_model = PointcloudEncoder(point_transformer)

        if pretrained:
            checkpoint = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/checkpoints/uni3d.pt')
            state_dict = checkpoint['module']
            state_dict_new = {}
            for k, v in state_dict.items():
                state_dict_new[k.replace('point_encoder.', '')] = v
            for k, v in state_dict_new.items():
                if k in self.drone_lidar_model.state_dict() and v.shape == self.drone_lidar_model.state_dict()[k].shape:
                    self.drone_lidar_model.state_dict()[k] = v
                else:
                    print(f"Skipping layer: {k}")

        if train_with_offset:
            self.MLP = MLP()
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.img_model)
        else:
            data_config = timm.data.resolve_model_data_config(self.drone_img_model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        if self.share_weights:
            self.img_model.set_grad_checkpointing(enable)
        else:
            self.drone_img_model.set_grad_checkpointing(enable)
            self.satellite_img_model.set_grad_checkpointing(enable)
        self.drone_lidar_model.set_grad_checkpointing(enable)

    def forward(
            self, 
            drone_img=None, 
            drone_lidar_pts=None,
            drone_lidar_clr=None,
            satellite_img=None
        ):  

        drone_img_features = None
        drone_lidar_features = None
        satellite_img_features = None

        if self.share_weights:
            if drone_img is not None:
                drone_img_features = self.img_model(drone_img)
            if satellite_img is not None:
                satellite_img_features = self.img_model(satellite_img)
        else:
            if drone_img is not None:
                drone_img_features = self.drone_img_model(drone_img)
            if satellite_img is not None:
                satellite_img_features = self.satellite_img_model(satellite_img)
        if drone_lidar_pts is not None and drone_lidar_clr is not None:
            drone_lidar_features = self.drone_lidar_model(drone_lidar_pts, drone_lidar_clr)
        
        result = {
            "drone_img_features": drone_img_features,
            "drone_lidar_features": drone_lidar_features,
            "satellite_img_features": satellite_img_features,
        }

        return result


if __name__ == '__main__':

    model = DesModelWithPC(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                           pc_model_name='eva02_base_patch14_448.mim_in22k_ft_in1k',
                           img_size=384)
    model.cuda()
    img = torch.rand((1, 3, 384, 384)).cuda()
    points = torch.rand(1, 50000, 3).cuda()
    colors = torch.rand(1, 50000, 3).cuda()
    batch = {'drone_img': img, 'drone_lidar_pts': points, 'drone_lidar_clr': colors}

    result = model(**batch)
    print(result['drone_img_features'].shape)
    print(result['drone_lidar_features'].shape)

    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

