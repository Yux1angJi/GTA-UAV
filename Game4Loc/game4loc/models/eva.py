""" EVA

EVA from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

This file contains EVA & EVA02 model implementations evolved from BEiT, additional models in vision_transformer.py.

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision
import math
from typing import Callable, List, Optional, Tuple, Union
from matplotlib.colors import rgb2hex

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

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


def resize_pos_embed(pos_embed_checkpoint, new_grid_size):
    """
    通过插值调整位置编码的大小
    """
    cls_token = pos_embed_checkpoint[:, :1, :]  # 提取 [CLS] token
    pos_tokens = pos_embed_checkpoint[:, 1:, :]  # 去掉 [CLS] token

    old_grid_size = int((pos_tokens.shape[1]) ** 0.5)  # 计算旧的 grid_size
    pos_tokens = pos_tokens.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)

    pos_tokens = F.interpolate(pos_tokens, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)

    new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)  # 重新拼接 [CLS] token
    return new_pos_embed      


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
# class Adapter(nn.Module):  # Adapter is used to add to the transformer block for global adaptation
#     def __init__(self, D_features, mlp_ratio=0.75, act_layer=nn.ReLU, skip_connect=True):
#         # mlp_ratio is the bottleneck ratio of adapters
#         super().__init__()
#         self.skip_connect = skip_connect
#         D_hidden_features = int(D_features * mlp_ratio)
        
#         self.scale_mlp = nn.Sequential(
#             nn.Linear(D_features, D_features),
#             nn.ReLU(),
#             nn.Linear(D_features, 1)  # Output 1 value per position
#         )
#         self.shift_mlp = nn.Sequential(
#             nn.Linear(D_features, D_features),
#             nn.ReLU(),
#             nn.Linear(D_features, 1)  # Output 1 value per position
#         )
        
#     def forward(self, rgb_features, depth_features):
#         scale = self.scale_mlp(depth_features)  # Output shape: (B, N, 1)
#         shift = self.shift_mlp(depth_features)  # Output shape: (B, N, 1)

#         # Broadcast scale and shift to match the shape of rgb_features
#         scale = scale.expand(-1, -1, rgb_features.size(-1))  # Shape: (B, N, C)
#         shift = shift.expand(-1, -1, rgb_features.size(-1))  # Shape: (B, N, C)

#         # Apply modulation to the RGB features
#         modulated_features = scale * rgb_features + shift
#         return modulated_features


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


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
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
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
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

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

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
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

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
            adapter: bool=False,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
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
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
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

        self.adapter = adapter
        ############## Adapter
        if self.adapter:
            self.serial_adapter = Adapter(D_features=dim, mlp_ratio=0.5)
            self.parallel_adapter = Adapter(D_features=dim, mlp_ratio=0.5, skip_connect=False)
        ################

    def forward(self, rgb, d=None, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        def attn_residual_func(rgb, rope, attn_mask):
            rgb = self.attn(self.norm1(rgb), rope=rope, attn_mask=attn_mask)
            rgb = self.serial_adapter(rgb)
            return rgb

        def ffn_residual_func(rgb):
            return self.mlp(self.norm2(rgb))+0.2*self.parallel_adapter(self.norm2(rgb))  # 0.2 is the scaling factor for Parallel adapter

        if self.gamma_1 is None:
            if self.adapter:
            ################################
            ## Adapter
                rgb = rgb + self.drop_path1(attn_residual_func(rgb, rope, attn_mask))
                rgb = rgb + self.drop_path2(ffn_residual_func(rgb))
                # rgb = rgb + self.drop_path2(self.mlp(self.norm2(rgb)))
            ################################
            else:
                rgb = rgb + self.drop_path1(self.attn(self.norm1(rgb), rope=rope, attn_mask=attn_mask))
                rgb = rgb + self.drop_path2(self.mlp(self.norm2(rgb)))
        else:
            if self.adapter:
            ################################
            ## Adapter
                rgb = rgb + self.drop_path1(self.gamma_1 * attn_residual_func(rgb, rope, attn_mask))
                rgb = rgb + self.drop_path2(self.gamma_2 * ffn_residual_func(rgb))
                # rgb = rgb + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(rgb)))
            ################################
            else:
                rgb = rgb + self.drop_path1(self.gamma_1 * self.attn(self.norm1(rgb), rope=rope, attn_mask=attn_mask))
                rgb = rgb + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(rgb)))
        return rgb


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
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
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.attn = EvaAttention(
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
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
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
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 0,
            global_pool: str = 'avg',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            num_reg_tokens: int = 0,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            adapter_list=[],
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        r = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, embed_dim)) if num_reg_tokens else None
        self.cls_embed = class_token and self.reg_token is None

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                num_prefix_tokens=self.num_prefix_tokens,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                adapter=(i in adapter_list),
            )
            for i in range(depth)])

        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=r) for i in range(depth)]

        use_fc_norm = self.global_pool != 'token'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.global_pool == 'gem':
            self.gem = GeM()

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        if self.reg_token is not None:
            trunc_normal_(self.reg_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        if self.reg_token is not None:
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            x = torch.cat(to_cat + [x], dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        return x, rot_pos_embed

    def forward_features(self, rgb, d=None, intermediate=False):
        rgb = self.patch_embed(rgb)
        rgb, rot_pos_embed = self._pos_embed(rgb)
        x_intermediate = []
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                rgb = checkpoint(blk, rgb, d, rope=rot_pos_embed, use_reentrant=False)
            else:
                rgb = blk(rgb, d, rope=rot_pos_embed)
            if intermediate:
                x_intermediate.append(rgb)
        x = self.norm(rgb)
        if intermediate:
            return x, x_intermediate
        else:
            return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool == 'max':
            x = x[:, self.num_prefix_tokens:].max(dim=1)[0]
        elif self.global_pool == 'gem':
            x = self.gem(x[:, self.num_prefix_tokens:])
        elif self.global_pool == 'cls':
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, rgb, d=None):
        x = self.forward_features(rgb, d)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'mit', **kwargs
    }


def eva_giant_patch14_224(pretrained=False, **kwargs) -> Eva:
    """ EVA-g model https://arxiv.org/abs/2211.07636 """
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = Eva(**model_args)
    return model


def eva_giant_patch14_336(pretrained=False, **kwargs) -> Eva:
    """ EVA-g model https://arxiv.org/abs/2211.07636 """
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = Eva(**model_args)
    return model


def eva_giant_patch14_560(pretrained=False, **kwargs) -> Eva:
    """ EVA-g model https://arxiv.org/abs/2211.07636 """
    model_args = dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408)
    model = Eva(**model_args)
    return model


def eva02_tiny_patch14_224(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_small_patch14_224(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_base_patch14_224(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_large_patch14_224(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_tiny_patch14_336(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_small_patch14_336(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_base_patch14_448(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva02_large_patch14_448(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def eva_giant_patch14_clip_224(pretrained=False, **kwargs) -> Eva:
    """ EVA-g CLIP model (only difference from non-CLIP is the pooling)  """
    model_args = dict(
        patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408,
        global_pool=kwargs.pop('global_pool', 'token'))
    model = Eva(**model_args)
    return model


def eva02_base_patch16_clip_224(pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_base """
    model_args = dict(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'cls'),
    )
    model = Eva(**model_args)
    if pretrained:
        checkpoint = torch.load('./pretrained/openclip/open_clip_pytorch_model_base.bin', map_location="cpu")
        model_state_dict = {}
        for k, v in checkpoint.items():
            if 'visual.trunk.' in k:
                model_state_dict[k.replace('visual.trunk.', '')] = v
        pos_embed_checkpoint = model_state_dict['pos_embed']  # 适用于 `timm` 结构
        new_grid_size = 24  # 384x384 对应的 `patch grid` 大小
        model_state_dict['pos_embed'] = resize_pos_embed(pos_embed_checkpoint, new_grid_size)
        model.load_state_dict(model_state_dict, strict=False)
    return model


def eva02_large_patch14_clip_224(pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_large """
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = Eva(**model_args)
    return model


def eva02_large_patch14_clip_336(pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_large """
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = Eva(**model_args)
    return model


def eva02_enormous_patch14_clip_224(pretrained=False, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that uses residual post-norm in blocks """
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1792,
        depth=64,
        num_heads=16,
        mlp_ratio=15360 / 1792,
        use_post_norm=True,
        global_pool=kwargs.pop('global_pool', 'token'),
    )
    model = Eva(**model_args)
    return model


def vit_medium_patch16_rope_reg1_gap_256(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=12,
        num_heads=8,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def vit_mediumd_patch16_rope_reg1_gap_256(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=20,
        num_heads=8,
        qkv_fused=True,
        qkv_bias=False,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def vit_betwixt_patch16_rope_reg4_gap_256(pretrained=False, **kwargs) -> Eva:
    model_args = dict(
        img_size=256,
        patch_size=16,
        embed_dim=640,
        depth=12,
        num_heads=10,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=4,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
    )
    model = Eva(**model_args)
    return model


def vit_base_patch16_rope_reg1_gap_256(
        pretrained=False, 
        in_chans=3,
        img_size=384,
        global_pool='avg',
        **kwargs,
    ) -> Eva:
    model_args = dict(
        in_chans=in_chans,
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=True,
        qkv_bias=True,
        init_values=1e-5,
        class_token=False,
        num_reg_tokens=1,
        use_rot_pos_emb=True,
        use_abs_pos_emb=False,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=global_pool,
    )
    model = Eva(**model_args)
    if pretrained:
        checkpoint = torch.load('./pretrained/vit_base_patch16_rope_reg1_gap_256/pytorch_model.bin')
        if in_chans == 1:
            pretrained_weight = checkpoint['patch_embed.proj.weight']
            new_weight = pretrained_weight.mean(dim=1, keepdim=True)  # 平均到1通道
            checkpoint['patch_embed.proj.weight'] = new_weight  # 更新权重
        model.load_state_dict(checkpoint, strict=False)
    return model


if __name__ == '__main__':
    pass
    # from PointcloudEncoder
    # point_transformer = eva02_base_patch14_448()
    # model = PointcloudEncoder(point_transformer)
    # checkpoint = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/pretrained/uni3d/model.pt', map_location="cpu")
    # state_dict = checkpoint['module']
    # model.load_state_dict(state_dict)

    # state_dict_new = {}
    # for k, v in state_dict.items():
    #     state_dict_new[k.replace('point_encoder.', '')] = v
    # for k, v in state_dict_new.items():
    #     if k in model.state_dict().keys():
    #         model.state_dict()[k] = v
    #     else:
    #         print(f"Skipping layer: {k}")
    # print(len(model.blocks))
    # print(state_dict.keys())
    # print(model.state_dict().keys())
    # model.load_state_dict(model_state_dict)
