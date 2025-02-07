import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image
from urllib.request import urlopen
from thop import profile
from transformers import CLIPTextConfig, CLIPTextModel, CLIPModel, AutoTokenizer
from torch.utils.checkpoint import checkpoint
from timm.layers import LayerNorm

from .eva import vit_base_patch16_rope_reg1_gap_256, EvaAttention, EvaBlock, EvaCrossAttention, EvaCrossAttentionBlock

# from pointnet2_ops import pointnet2_utils

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class PointcloudEncoder(nn.Module):
    def __init__(self, 
                 point_transformer,
                 pc_feat_dim=768,
                 embed_dim=768,
                 group_size=32,
                 num_group=512,
                 pc_encoder_dim=256,
                 patch_dropout=0.):
        super().__init__()
        self.trans_dim = pc_feat_dim # 768
        self.embed_dim = embed_dim # 512
        self.group_size = group_size # 32
        self.num_group = num_group # 512
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim = pc_encoder_dim # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim)
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim)
        
        # bridge transformer and clip embedding
        self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer


    def forward(self, pts, colors):
        # divide the point cloud in the same form. This is important
        _, center, features = self.group_divider(pts, colors)

        # encoder the input cloud patches
        group_input_tokens = self.encoder(features)  #  B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = x + pos
        # x = x.half()
        
        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)

        x = self.visual.pos_drop(x)

        # ModuleList not support forward
        for i, blk in enumerate(self.visual.blocks):
            x = blk(x)
        x = self.visual.norm(x[:, 0, :])
        x = self.visual.fc_norm(x)

        x = self.trans2embed(x)
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: b x n x d
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)                
        return x.pow(1. / self.p)  


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


class DesModelWithMM(nn.Module):

    def __init__(self, 
                 model_name='vit',
                 pc_model_name='eva02_base_patch14_448.mim_in22k_ft_in1k',
                 pretrained=True,
                 drop_path_rate=0.,
                 img_size=384,
                 share_weights=True,
                 with_depth=False,
                 with_lidar=False,
                 with_text=False,
                 train_with_offset=False,
                 ):
                 
        super(DesModelWithMM, self).__init__()
        self.share_weights = share_weights
        self.model_name = model_name
        self.img_size = img_size
        self.with_depth = with_depth
        self.with_lidar = with_lidar
        self.with_text = with_text
        if share_weights:
            if "vit" in model_name or "swin" in model_name:
                # automatically change interpolate pos-encoding to img_size
                # self.img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
                self.img_model = vit_base_patch16_rope_reg1_gap_256(in_chans=3, pretrained=pretrained, global_pool='gem')
            else:
                self.img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        else:
            if "vit" in model_name or "swin" in model_name:
                self.drone_img_model = vit_base_patch16_rope_reg1_gap_256(in_chans=3, pretrained=pretrained, global_pool='gem')
                self.satellite_img_model = vit_base_patch16_rope_reg1_gap_256(in_chans=3, pretrained=pretrained, global_pool='gem')
            else:
                self.drone_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
                self.satellite_img_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if with_lidar:
            point_transformer = timm.create_model(pc_model_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
            self.drone_lidar_model = PointcloudEncoder(point_transformer)

            if pretrained:
                checkpoint = torch.load('./pretrained/uni3d/model.pt')
                state_dict = checkpoint['module']
                state_dict_new = {}
                for k, v in state_dict.items():
                    state_dict_new[k.replace('point_encoder.', '')] = v
                for k, v in state_dict_new.items():
                    if k in self.drone_lidar_model.state_dict() and v.shape == self.drone_lidar_model.state_dict()[k].shape:
                        self.drone_lidar_model.state_dict()[k] = v
                    else:
                        print(f"Skipping layer: {k}")

        if share_weights:
            embed_dim = self.img_model.embed_dim
            self.img_model.set_grad_checkpointing()
        else:
            embed_dim = self.drone_img_model.embed_dim
            self.drone_img_model.set_grad_checkpointing()
            self.satellite_img_model.set_grad_checkpointing()
        
        if with_text:
            self.desc_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_projection = nn.Linear(512, embed_dim, bias=False)
        elif with_depth:
            # self.drone_depth_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size, in_chans=1)

            # self.drone_depth_model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, num_classes=0)
            self.depth_model = vit_base_patch16_rope_reg1_gap_256(in_chans=1, pretrained=pretrained, global_pool='gem')
            # conv1_weight = self.drone_depth_model.patch_embed.proj.weight
            # self.drone_depth_model.patch_embed.proj.weight = torch.nn.Parameter(conv1_weight.mean(dim=1, keepdim=True))
            
            self.depth_model.set_grad_checkpointing() 

        self.grad_checkpointing = True

        self.collab_model = EvaCrossAttentionBlock(dim=768, num_heads=12)
        self.fc_norm = LayerNorm(embed_dim)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def get_config(self,):
        if self.share_weights:
            data_config = timm.data.resolve_model_data_config(self.img_model)
        else:
            data_config = timm.data.resolve_model_data_config(self.drone_img_model)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        pass
    
    def forward_uni(
            self, 
            drone_img=None, 
            drone_lidar_pts=None,
            drone_lidar_clr=None,
            drone_depth=None,
            drone_desc=None,
            satellite_img=None
        ):  

        drone_img_features = None
        drone_depth_features = None
        drone_lidar_features = None
        drone_desc_features = None
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
        if self.with_depth and drone_depth is not None:
            drone_depth_features = self.drone_depth_model(drone_depth)
        if self.with_lidar and drone_lidar_pts is not None and drone_lidar_clr is not None:
            drone_lidar_features = self.drone_lidar_model(drone_lidar_pts, drone_lidar_clr)
        if self.with_text and drone_desc is not None:
            text_outputs = self.drone_desc_model(**drone_desc)
            pooled_output = text_outputs[0]
            # print(pooled_output.shape)
            drone_desc_features = self.text_projection(pooled_output)
        
        result = {
            "drone_img_features": drone_img_features,
            "drone_lidar_features": drone_lidar_features,
            "drone_depth_features": drone_depth_features,
            "drone_desc_features": drone_desc_features,
            "satellite_img_features": satellite_img_features,
        }

        return result
    
    def forward(
            self, 
            drone_img=None, 
            drone_lidar_pts=None,
            drone_lidar_clr=None,
            drone_depth=None,
            drone_desc=None,
            satellite_img=None,
            satellite_desc=None,
        ):

        if not self.share_weights:
            if drone_img != None:
                query_features = self.forward_drone_view(
                    img=drone_img,
                    lidar_pts=drone_lidar_pts,
                    lidar_clr=drone_lidar_clr,
                    depth=drone_depth,
                    desc=drone_desc,
                )
            if satellite_img != None:
                gallery_features = self.forward_satellite_view(
                    img=satellite_img,
                )
            
            if drone_img != None and satellite_img != None:
                return query_features, gallery_features
            elif drone_img != None:
                return query_features
            elif satellite_img != None:
                return gallery_features

        else:
            if drone_img != None:
                query_features = self.forward_one_view(
                    img=drone_img,
                    lidar_pts=drone_lidar_pts,
                    lidar_clr=drone_lidar_clr,
                    depth=drone_depth,
                    desc=drone_desc,
                )
            if satellite_img != None:
                gallery_features = self.forward_one_view(
                    img=satellite_img,
                    desc=satellite_desc,
                )
            
            if drone_img != None and satellite_img != None:
                return query_features, gallery_features
            elif drone_img != None:
                return query_features
            elif satellite_img != None:
                return gallery_features


    def forward_satellite_view(
        self,
        img,
    ):
        img_features = self.satellite_img_model(img)
        return img_features
    
        
    def forward_drone_view(
        self, 
        img=None,
        lidar_pts=None,
        lidar_clr=None,
        depth=None,
        desc=None,
    ):
        if self.with_depth:
            if depth == None:
                N, C, H, W = img.shape
                depth = torch.zeros((N, 1, H, W)).to(device=img.device)
            depth_features = self.depth_model.forward_features(depth)
            # depth_features = depth_features.detach()
        img_embed = self.drone_img_model.patch_embed(img)
        img_embed, rot_pos_embed = self.drone_img_model._pos_embed(img_embed)
        for i, blk in enumerate(self.drone_img_model.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                img_embed = checkpoint(blk, img_embed, None, rope=rot_pos_embed, use_reentrant=False)
            else:
                img_embed = blk(img_embed, None, rope=rot_pos_embed)
        img_features = self.drone_img_model.norm(img_embed)
        # img_features = img_features.detach()
        
        if self.with_depth:
            # depth_features = self.drone_depth_model.forward_features(drone_depth)
            img_collab_features = self.collab_model(query=img_features, key_value=depth_features, rope=rot_pos_embed)
        
        elif self.with_text:
            rot_pos_embed = None
            text_outputs = self.desc_model(**desc)
            desc_features = self.text_projection(text_outputs[0])
            img_collab_features = self.collab_model(query=img_features, key_value=desc_features, rope=rot_pos_embed)
        
        img_collab_features = self.drone_img_model.forward_head(img_collab_features)

        return img_collab_features
    

    def forward_one_view(
        self, 
        img=None,
        lidar_pts=None,
        lidar_clr=None,
        depth=None,
        desc=None,
    ):
        
        if self.with_depth:
            if depth == None:
                N, C, H, W = img.shape
                depth = torch.zeros((N, 1, H, W)).to(device=img.device)
            depth_features = self.depth_model.forward_features(depth)
        img_embed = self.img_model.patch_embed(img)
        img_embed, rot_pos_embed = self.img_model._pos_embed(img_embed)
        for i, blk in enumerate(self.img_model.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                img_embed = checkpoint(blk, img_embed, None, rope=rot_pos_embed, use_reentrant=False)
            else:
                img_embed = blk(img_embed, None, rope=rot_pos_embed)
        img_features = self.img_model.norm(img_embed)
        
        if self.with_depth:
            # depth_features = self.drone_depth_model.forward_features(drone_depth)
            img_collab_features = self.collab_model(query=img_features, key_value=depth_features, rope=rot_pos_embed)
        
        elif self.with_text:
            rot_pos_embed = None
            text_outputs = self.desc_model(**desc)
            desc_features = self.text_projection(text_outputs[0])
            img_collab_features = self.collab_model(query=img_features, key_value=desc_features, rope=rot_pos_embed)
        
        img_collab_features = self.img_model.forward_head(img_collab_features)
        # img_collab_features = self.fc_norm(img_collab_features)
        # img_collab_features = img_collab_features[:, 1:].mean(dim=1)
        # img_collab_features = self.fc_norm(img_collab_features)

        return img_collab_features


if __name__ == '__main__':

    def pc_norm(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if m < 1e-6:
            print("警告：最大距离接近零，点云可能集中在一个点上。")
            return pc  # 或者采取其他适当的处理方式
        pc = pc / m
        print('m nan', np.isnan(m).sum())
        return pc

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset.pc_utils import *
    import open3d as o3d

    model = DesModelWithMM(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                           pc_model_name='eva02_base_patch14_448.mim_in22k_ft_in1k',
                           with_lidar=False,
                           with_text=True,
                           img_size=384)
    model.cuda()

    img = torch.rand((1, 3, 384, 384)).cuda()

    points = o3d.io.read_point_cloud('/home/xmuairmud/data/GTA-UAV-data/Lidar/drone/lidars/200_0001_0000006374.ply')
    points = np.array(points.points, dtype=np.float32)
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    
    print(points)
    points = pc_norm(points)
    print(points)

    points = augment_pc(points)
    print(np.isnan(points).sum())
    print(points.shape)
    points = torch.from_numpy(points).cuda()
    points = points[None, ...]
    colors = torch.ones_like(points).cuda()
    print(torch.isnan(points).sum())

    desc = 'This is a test.'
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    desc = tokenizer(
        [desc],
        padding="max_length",  # 短文本自动填充
        truncation=True,       # 长文本自动截断
        max_length=77, # 固定最大长度
        return_tensors="pt"    # 返回 PyTorch 张量
    )
    desc = {key: value[0].cuda() for key, value in desc.items()}
    print('text', desc)

    batch = {'drone_img': img, 'drone_lidar_pts': points, 'drone_lidar_clr': colors, 'drone_desc': desc}

    # result = model(**batch)
    # print(result['drone_img_features'].shape)
    # print(result['drone_lidar_features'].shape)
    # print(torch.isnan(result['drone_lidar_features']).sum())

    # print(result['drone_desc_features'].shape)

    query = model.forward_query(**batch)
    print(query.shape)

    # flops, params = profile(model, inputs=(x,))
    # # print(img.size)
    # # img = transform(img)
    # # print(img.size)

