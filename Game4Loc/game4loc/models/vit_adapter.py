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

from .eva import vit_base_patch16_rope_reg1_gap_256


def forward_sa_adapter_with_rope(module, x, adapter1, adapter2, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x, adapter1=adapter1, adapter2=adapter2, rope=rope)

def forward_sa_with_rope(module, x, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x, rope=rope)

def forward_ca_with_rope(module, x1, x2, rope):
    # 调用 blk 的 forward 方法并传递 rope 参数
    return module(x1, x2, rope=rope)

def new_forward_eva_block_with_adapter(self,
                                        x, 
                                        adapter1,
                                        adapter2,
                                        rope: Optional[torch.Tensor] = None, 
                                        attn_mask: Optional[torch.Tensor] = None
                                      ):
    def attn_residual_func(x, rope, attn_mask):
        return adapter1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))

    def ffn_residual_func(x):
        return self.mlp(self.norm2(x))+0.2*adapter2(self.norm2(x))  # 0.2 is the scaling factor for Parallel adapter

    if self.gamma_1 is None:
        x = x + self.drop_path1(attn_residual_func(x, rope=rope, attn_mask=attn_mask))
        x = x + self.drop_path2(ffn_residual_func(x))
    else:
        x = x + self.drop_path1(self.gamma_1 * attn_residual_func(x, rope=rope, attn_mask=attn_mask))
        x = x + self.drop_path2(self.gamma_2 * ffn_residual_func(x))
    return x


class Adapter(nn.Module):  # Adapter is used to add to the transformer block for global adaptation
    def __init__(self, D_features, mlp_ratio=0.75, act_layer=nn.ReLU, skip_connect=True):
        # mlp_ratio is the bottleneck ratio of adapters
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class EvaCrossAttentionBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            num_prefix_tokens: int = 1,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """
        Args:
            dim: Dimension of the input.
            num_heads: Number of attention heads.
            qkv_bias: If bias should be used in QKV projections.
            qkv_fused: Whether to fuse QKV into a single projection.
            mlp_ratio: Ratio of MLP hidden dimension to input dimension.
            swiglu_mlp: Whether to use SwiGLU in the MLP.
            scale_mlp: Whether to use normalization in MLP.
            scale_attn_inner: Whether to use normalization inside attention.
            proj_drop: Dropout rate for the projection layers.
            attn_drop: Dropout rate for attention scores.
            drop_path: Drop path rate for stochastic depth.
            init_values: Initial scaling values for residual layers.
            act_layer: Activation function to use in the MLP.
            norm_layer: Normalization layer function.
            attn_head_dim: Dimension of each attention head.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = EvaCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, key_value, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        # Cross-attention: query and key-value are from different inputs
        if self.gamma_1 is None:
            query = query + self.drop_path1(self.cross_attn(self.norm1(query), key_value, rope=rope, attn_mask=attn_mask))
            query = query + self.drop_path2(self.mlp(self.norm2(query)))
        else:
            query = query + self.drop_path1(self.gamma_1 * self.cross_attn(self.norm1(query), key_value, rope=rope, attn_mask=attn_mask))
            query = query + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(query)))
        return query


class EvaCrossAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int = 768,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """
        Args:
            dim: Dimension of the input.
            num_heads: Number of attention heads.
            qkv_bias: If bias should be used in QKV projections.
            qkv_fused: Whether to fuse QKV into a single projection.
            attn_drop: Dropout rate for attention scores.
            proj_drop: Dropout rate for the output projection.
            attn_head_dim: Dimension of each attention head.
            norm_layer: Normalization layer after the attention output.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()

        self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            query,
            key_value,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = query.shape
        _, M, _ = key_value.shape  # M is length of key-value sequence

        # Q, K, V projections
        q = self.q_proj(query).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, head_dim
        k = self.k_proj(key_value).reshape(B, M, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, M, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # Cross-attention on Q and K
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v  # Weighted sum with V

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTAdapter(nn.Module):
    def __init__(self, 
                 vit_model_name='vit_base_patch16_rope_reg1_gap_256.sbb_in1k', 
                 img_size=(384, 384), 
                 embed_dim=768,
                 num_heads=12,
                 num_blocks=2,
                 lamda_drop_rate=0.,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_encoder = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size, in_chans=1)
        # pretrained_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1021075644/weights_end.pth')
        # depth_state_dict = {}
        # for k, v in pretrained_state_dict.items():
        #     if 'drone_depth_model.' in k:
        #         depth_state_dict[k.replace('drone_depth_model.', '')] = v
        # self.depth_encoder.load_state_dict(depth_state_dict)

        # for param in self.depth_encoder.parameters():
        #     param.requires_grad = False
        
        # self.vit_model = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size)
        self.vit_model = vit_base_patch16_rope_reg1_gap_256()
        if True:
            # model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1016011311/weights_end.pth')  
            model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1026155942/weights_end.pth')  
            model_state_dict_new = {}
            for k, v in model_state_dict.items():
                model_state_dict_new[k.replace('model.', '')] = v
            self.vit_model.load_state_dict(model_state_dict_new, strict=False)

        for name, param in self.vit_model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
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
    
    def forward(self, x):

        if x.shape[1] == 3:
            rgb = x[:, :3, :, :]
            N, C, H, W = rgb.shape
            d = torch.zeros((N, 1, H, W)).to(device=x.device, dtype=x.dtype)
        else:
            rgb = x[:, :3, :, :]
            d = x[:, 3:, :, :]
            # d = d.repeat(1, 3, 1, 1)

        d = self.depth_encoder.forward_features(d)

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
        for blk in self.vit_model.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                rgb = checkpoint(blk, rgb, d, rope=rot_pos_embed, use_reentrant=False)
            else:
                rgb = blk(rgb, d, rope=rot_pos_embed)
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

    model = ViTAdapter()
    model.cuda()
    x1 = torch.rand((1, 3, 384, 384)).cuda()
    x2 = torch.rand((1, 4, 384, 384)).cuda()

    x1, _ = model(x1)
    x2, _ = model(x2)
    print(x1.shape, x2.shape)
    
