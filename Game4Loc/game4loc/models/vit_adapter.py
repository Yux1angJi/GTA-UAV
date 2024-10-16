import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.layers import DropPath, Mlp, RotaryEmbeddingCat, LayerNorm, GluMlp, SwiGLU, apply_rot_embed_cat, use_fused_attn

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
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.depth_encoder = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size, in_chans=3)
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        
        self.vit_model = timm.create_model(model_name=vit_model_name, pretrained=True, num_classes=0, img_size=img_size)
        if True:
            model_state_dict = torch.load('/home/xmuairmud/jyx/GTA-UAV/Game4Loc/work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/1016011311/weights_end.pth')  
            model_state_dict_new = {}
            for k, v in model_state_dict.items():
                model_state_dict_new[k.replace('model.', '')] = v
            self.vit_model.load_state_dict(model_state_dict_new, strict=False)
        # for param in self.vit_model.parameters():
        #     param.requires_grad = False
        
        self.depth_adapters = nn.Sequential(*[
            EvaCrossAttentionBlock(dim=embed_dim, num_heads=num_heads)
            for i in range(num_blocks)
        ])

        self.fc_norm = LayerNorm(embed_dim)
    
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

        if d != None:
            d = self.depth_encoder.forward_features(d)

        for i in range(len(self.vit_model.blocks)):
            rgb = self.vit_model.blocks[i](rgb, rope=rot_pos_embed)
            # if d != None:
            #     rgb = self.depth_adapters[i](d, rgb, rope=rot_pos_embed)
        # rgb = self.vit_model.norm(rgb)
        # rgb = self.vit_model.forward_head(rgb)

        if d != None:
            rgb = self.depth_adapters[0](d, rgb, rope=rot_pos_embed)
            for i in range(1, len(self.depth_adapters)):
                rgb = self.depth_adapters[i](rgb, rgb, rope=rot_pos_embed)
        rgb = self.fc_norm(rgb)
        rgb = rgb[:, 1:].mean(dim=1)
        
        return rgb



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

    x1 = model(x1)
    x2 = model(x2)
    print(x1.shape, x2.shape)
    
