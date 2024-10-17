import torch
import timm
import numpy as np
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.layers import DropPath, Mlp

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int = 768,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Perform cross-attention between x1 and x2
        x1 = x1 + self.drop_path1(self.ls1(self.cross_attn(self.norm1(x1), self.norm1(x2))))
        x1 = x1 + self.drop_path2(self.ls2(self.mlp(self.norm2(x1))))
        return x1

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query 来自 x1，Key 和 Value 来自 x2
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, N, C = x1.shape

        # 从 x1 提取 query
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 从 x2 提取 key 和 value
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 计算注意力得分
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 使用 value 计算加权和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x) + x2


class ViTAdapter(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', img_size=(384, 384), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_encoder = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size)
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        
        self.vit_model = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size)
        if True:  
            model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1016011311/weights_end.pth')  
            model_state_dict_new = {}
            for k, v in model_state_dict.items():
                model_state_dict_new[k.replace('model.', '')] = v
            self.vit_model.load_state_dict(model_state_dict_new, strict=False)
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.depth_adapters = nn.Sequential(*[
            CrossAttentionBlock()
            for i in range(len(self.vit_model.blocks))
        ])
    
    def forward(self, x):
        if x.shape[1] == 3:
            rgb = x[:, :3, :, :]
            d = None
        else:
            rgb = x[:, :3, :, :]
            d = x[:, 3:, :, :]
            d = d.repeat(1, 3, 1, 1)

        rgb = self.vit_model.patch_embed(rgb)
        rgb, rot_pos_embed = self.vit_model._pos_embed(rgb)

        for i in range(len(self.vit_model.blocks)):
            rgb = self.vit_model.blocks[i](rgb, rope=rot_pos_embed)
        rgb = self.vit_model.norm(rgb)
        rgb = self.vit_model.forward_head(rgb)

        if d != None:
            d = self.depth_encoder.forward_features(d)
            rgb = self.depth_adapters[0](d, rgb)
            for i in range(1, len(self.depth_adapters)):
                rgb = self.depth_adapters[i](rgb, rgb)
        
        return rgb



if __name__ == '__main__':
    # model = timm.create_model(model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', pretrained=True, num_classes=0, img_size=(384, 384))
    # print(model.__class__)
    # print(model.blocks[0].attn.q_norm)

    model = ViTAdapter()
    model.cuda()
    x1 = torch.rand((1, 3, 384, 384)).cuda()
    x2 = torch.rand((1, 6, 384, 384)).cuda()

    x1 = model(x1)
    x2 = model(x2)
    print(x1.shape, x2.shape)
    
