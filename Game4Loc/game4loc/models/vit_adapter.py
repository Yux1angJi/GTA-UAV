# References:
#   https://github.com/Lu-Feng/SelaVPR


import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.layers import DropPath, Mlp, RotaryEmbeddingCat, LayerNorm, GluMlp, SwiGLU, apply_rot_embed_cat, use_fused_attn

from torch.utils.checkpoint import checkpoint

from .eva import vit_base_patch16_rope_reg1_gap_256, EvaAttention, EvaBlock, EvaCrossAttention, EvaCrossAttentionBlock


def forward_sa_adapter_with_rope(module, x, adapter1, adapter2, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x, adapter1=adapter1, adapter2=adapter2, rope=rope)

def forward_sa_with_rope(module, x, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x, rope=rope)

def forward_ca_with_rope(module, x1, x2, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x1, x2, rope=rope)


class ViTAdapter(nn.Module):
    def __init__(self, 
                 vit_model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                 img_size=(384, 384), 
                 embed_dim=768,
                 num_heads=12,
                 num_blocks=1,
                 lamda_drop_rate=0.,
                 global_pool='avg',
                 pretrained=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # self.depth_encoder = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size, in_chans=1)
        self.depth_encoder = vit_base_patch16_rope_reg1_gap_256(in_chans=1, pretrained=pretrained, global_pool=global_pool)
        # pretrained_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1026065900/weights_end.pth')
        # depth_state_dict = {}
        # for k, v in pretrained_state_dict.items():
        #     if 'drone_depth_model.' in k:
        #         depth_state_dict[k.replace('drone_depth_model.', '')] = v
        # self.depth_encoder.load_state_dict(depth_state_dict)

        # for param in self.depth_encoder.parameters():
        #     param.requires_grad = False
        
        # self.vit_model = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size)
        # self.vit_model = vit_base_patch16_rope_reg1_gap_256(adapter_list=[0,1,2,3,4,5,6,7,8,9])
        self.vit_model = vit_base_patch16_rope_reg1_gap_256(adapter_list=[], pretrained=pretrained, global_pool=global_pool)
        # if True:
            ###  AIRMUD-559
            # model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1016011311/weights_end.pth')  
            ###############
            # ###  AIRMUD
            # model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1029142103/weights_end.pth')  
            # ###############
            # model_state_dict_new = {}
            # for k, v in model_state_dict.items():
            #     model_state_dict_new[k.replace('model.', '')] = v
            # self.vit_model.load_state_dict(model_state_dict_new, strict=False)

        # for name, param in self.vit_model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad = False
            # else:
            #     print('not freeze', name)

        self.depth_adapters = nn.Sequential(*[
            EvaCrossAttentionBlock(dim=embed_dim, num_heads=num_heads)
            for i in range(num_blocks)
            # for i in range(len(self.vit_model.blocks))
        ])

        self.lamda_norm = LayerNorm(embed_dim)
        self.lamda_drop = nn.Dropout(lamda_drop_rate)
        self.lamda = nn.Linear(embed_dim, 1)

        self.fc_norm = LayerNorm(embed_dim)

        self.set_grad_checkpointing()
        self.grad_checkpointing = True

    def set_grad_checkpointing(self, enable=True):
        self.vit_model.set_grad_checkpointing(enable)
        self.depth_encoder.set_grad_checkpointing(enable)
    
    def forward(self, x):

        if x.shape[1] == 3:
            rgb = x[:, :3, :, :]
            N, C, H, W = rgb.shape
            d = torch.zeros((N, 1, H, W)).to(device=x.device, dtype=x.dtype)
        else:
            rgb = x[:, :3, :, :]
            d = x[:, 3:, :, :]
            # d = d.repeat(1, 3, 1, 1)

        d, d_intermediate = self.depth_encoder.forward_features(d, intermediate=True)

        ##################################
        ## Inter Adapter
        # rgb = self.vit_model(rgb, d)
            
        ##################################

        ####################
        ## Mid Adapter
        # for i in range(len(self.vit_model.blocks)):
        #     if d != None:
        #         if self.grad_checkpointing and not torch.jit.is_scripting():
        #             rgb = checkpoint(forward_ca_with_rope, self.depth_adapters[i], rgb, d, rot_pos_embed)
        #         else:
        #             rgb = self.depth_adapters[i](query=rgb, key_value=d, rope=rot_pos_embed)
        #     if self.grad_checkpointing and not torch.jit.is_scripting():
        #         # rgb = checkpoint(forward_sa_with_rope, self.vit_model.blocks[i], rgb, rot_pos_embed)
        #         rgb = checkpoint(
        #                 forward_sa_adapter_with_rope, self.vit_model.blocks[i], rgb, 
        #                 self.serial_adapters[i], self.parallel_adapters[i], 
        #                 rot_pos_embed, use_reentrant=False
        #             )
        #     else:
        #         # rgb = self.vit_model.blocks[i](rgb, rope=rot_pos_embed)
        #         rgb = self.vit_model.blocks[i](rgb, self.serial_adapters[i], self.parallel_adapters[i], rope=rot_pos_embed)
        # rgb = self.vit_model.norm(rgb)
        # rgb = self.vit_model.forward_head(rgb)
        ####################

        #####################
        ## Last Adapter
        # for i in range(len(self.vit_model.blocks)):
        #     if self.grad_checkpointing and not torch.jit.is_scripting():
        #         rgb.requires_grad_(True)
        #         # rgb = checkpoint(forward_sa_with_rope, blk, rgb, rot_pos_embed, use_reentrant=False)
        #         rgb = checkpoint(
        #                 self.vit_model.blocks[i], rgb, d,
        #                 rot_pos_embed, use_reentrant=False
        #             )
        #     else:
        #         # rgb = self.vit_model.blocks[i](rgb, rope=rot_pos_embed)
        #         rgb = self.vit_model.blocks[i](rgb, self.serial_adapters[i], self.parallel_adapters[i], rope=rot_pos_embed)
        
        rgb = self.vit_model.patch_embed(rgb)
        rgb, rot_pos_embed = self.vit_model._pos_embed(rgb)
        for i, blk in enumerate(self.vit_model.blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                rgb = checkpoint(blk, rgb, d_intermediate[i], rope=rot_pos_embed, use_reentrant=False)
            else:
                rgb = blk(rgb, d_intermediate[i], rope=rot_pos_embed)
        rgb = self.vit_model.norm(rgb)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            d.requires_grad_(True)
            rgb = checkpoint(forward_ca_with_rope, self.depth_adapters[0], rgb, d, rot_pos_embed, use_reentrant=False)
            # rgb2 = checkpoint(forward_ca_with_rope, self.depth_adapters[0], d, rgb, rot_pos_embed, use_reentrant=False)
            # rgb = rgb1 + rgb2
        else:
            rgb = self.depth_adapters[0](query=rgb, key_value=d, rope=rot_pos_embed)
            # rgb2 = self.depth_adapters[1](query=d, key_value=rgb, rope=rot_pos_embed)
            # rgb = rgb1 + rgb2

            # lamda = d[:, 1:].mean(dim=1)
            # lamda = self.lamda_norm(lamda)
            # lamda = self.lamda_drop(lamda)
            # lamda = self.lamda(lamda)
            # for i in range(1, len(self.depth_adapters)):
            #     if self.grad_checkpointing and not torch.jit.is_scripting():
            #         rgb = checkpointing(self.depth_adapters[i], query=rgb, key_value=rgb, rope=rot_pos_embed)
            #     else:
            #         rgb = self.depth_adapters[i](rgb, rgb, rope=rot_pos_embed)
        rgb = self.vit_model.forward_head(rgb)

        lamda = self.lamda_drop(rgb)
        lamda = self.lamda(lamda)
        #######################
        
        return rgb, lamda


if __name__ == '__main__':
    # model = timm.create_model(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', pretrained=True, num_classes=0, img_size=(384, 384))
    # print(model.__class__)
    # print(model.blocks[0].__class__)
    # print(model.patch_embed.grid_size)
    # print(model.fc_norm)
    # print(model.head_drop)
    # print(model.global_pool)
    # print(model.head)
    # print(model.num_prefix_tokens)

    model = ViTAdapter(global_pool='avg')
    print(model.vit_model.global_pool, model.vit_model.fc_norm)
    model.cuda()
    x1 = torch.rand((2, 3, 384, 384)).cuda()
    x2 = torch.rand((2, 4, 384, 384)).cuda()

    x1, _ = model(x1)
    x2, _ = model(x2)
    print(x1.shape, x2.shape)
    
