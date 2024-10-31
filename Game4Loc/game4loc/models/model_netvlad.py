import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch
import timm
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile


class TransVLAD(nn.Module):
    """TransVLAD module implementation"""

    def __init__(self, num_clusters=64, dim=512, alpha=100.0, expansion=2, group=8, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(TransVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.expansion = expansion
        self.group = group
        self.normalize_input = normalize_input

        self.linear1 = nn.Linear(dim, expansion * dim, bias=False)
        self.linear2 = nn.Linear(expansion * dim, group, bias=False)

        self.conv = nn.Conv2d(expansion * dim // group, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, expansion * dim // group), requires_grad=True)

        self.clsts = None
        self.traindescs = None


    def _init_params(self):
        clstsAssign = self.clsts / np.linalg.norm(self.clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, self.traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :] # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids.data.copy_(torch.from_numpy(self.clsts))
        self.conv.weight.data.copy_(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))

    def forward(self, x):
        N, _, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        x = self.linear1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dim = self.expansion * self.dim // self.group
        x_group = x.view(N, self.group, dim, H, W)

        attention = self.linear2(x.view(N, self.expansion * self.dim, -1).permute(0, 2, 1)).permute(0, 2, 1)
        attention = F.sigmoid(attention)

        # soft-assignment
        soft_assign = self.conv(x_group.reshape(-1, dim, H, W)).view(N, self.group, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=2)

        x_group_flatten = x_group.view(N, self.group, dim, -1)

        # calculate residuals to each clusters in one loop
        residual = x_group_flatten.expand(self.num_clusters, -1, -1, -1, -1).permute(1, 2, 0, 3, 4) - \
            self.centroids.expand(x_group_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(3) * attention.unsqueeze(2).unsqueeze(2)
        vlad = residual.sum(dim=-1).sum(dim=1)
        vlad = vlad.view(N, -1)

        return vlad

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


class DesModelWithVLAD(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pretrained=True,
                 img_size=384,
                 share_weights=True,
                 train_with_recon=False,
                 train_with_offset=False,
                 model_hub='timm'):

        super(DesModelWithVLAD, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
            else:
                self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
            else:
                self.model1 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.model2 = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if train_with_offset:
            self.MLP = MLP()
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.netvlad = TransVLAD(dim=768)
        

    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.model)
        else:
            data_config = timm.data.resolve_model_data_config(self.model1)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        if self.share_weights:
            self.model.set_grad_checkpointing(enable)
        else:
            self.model1.set_grad_checkpointing(enable)
            self.model2.set_grad_checkpointing(enable)

    def freeze_layers(self, frozen_blocks=10, frozen_stages=[0,0,0,0]):
        pass

    def forward(self, img1=None, img2=None):
        H = self.img_size // 16
        W = self.img_size // 16

        if self.share_weights:
            if img1 is not None and img2 is not None:
                
                image_features1 = self.model.forward_features(img1)[:, 1:, :]
                image_features2 = self.model.forward_features(img2)[:, 1:, :]
                B, _, D = image_features1.shape

                image_features1 = self.netvlad(image_features1.reshape(B, H, W, D).permute(0, 3, 1, 2))
                image_features2 = self.netvlad(image_features2.reshape(B, H, W, D).permute(0, 3, 1, 2))
                return image_features1, image_features2  
            elif img1 is not None:
                image_features = self.model.forward_features(img1)[:, 1:, :]
                B, _, D = image_features.shape
                image_features = self.netvlad(image_features.reshape(B, H, W, D).permute(0, 3, 1, 2))
                return image_features
            else:
                image_features = self.model.forward_features(img2)[:, 1:, :]
                B, _, D = image_features.shape
                image_features = self.netvlad(image_features.reshape(B, H, W, D).permute(0, 3, 1, 2))
                return image_features
        else:
            if img1 is not None and img2 is not None:
                image_features1 = self.model1(img1)     
                image_features2 = self.model2(img2)
                return image_features1, image_features2            
            elif img1 is not None:
                image_features = self.model1(img1)
                return image_features
            else:
                image_features = self.model2(img2)
                return image_features

    def offset_pred(self, img_feature1, img_feature2):
        offset = self.MLP(torch.cat((img_feature1, img_feature2), dim=1))
        return offset


if __name__ == '__main__':

    model = DesModelWithVLAD(model_name='timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k', img_size=384)

    x = torch.rand((1, 3, 384, 384))
    x = x.cuda()
    model.cuda()
    x = model(x)
    print(x.shape)