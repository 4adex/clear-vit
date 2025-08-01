import torch.nn as nn
import torch
from typing import Callable, OrderedDict, Optional, Dict
from functools import partial
# from models.blocks import MLPBlock
import math

from detectron2.layers import ShapeSpec
# from .backbone import Backbone
from detectron2.modeling.backbone import Backbone
import torch.nn.functional as F



class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for different output features"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, window_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadSelfAttention(dim, num_heads, dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CustomViT(Backbone):
    """Custom Vision Transformer compatible with DETR-X framework"""

    def __init__(self,
                 img_size=768,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 use_abs_pos=True,
                 use_rel_pos=False,  # Note: not used in this class
                 window_size=14,
                 window_block_indexes=(),
                 residual_block_indexes=(),
                 out_feature="last_feat"):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_feature = out_feature

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else 0,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Initialize weights
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embedding with interpolation for variable input sizes
        if self.pos_embed is not None:
            if x.shape[1] == self.pos_embed.shape[1]:
                x = x + self.pos_embed
            else:
                cls_pos_embed = self.pos_embed[:, :1, :]  # (1,1,C)
                patch_pos_embed = self.pos_embed[:, 1:, :]  # (1, num_orig_patches, C)

                # Original grid size of position embeddings
                orig_num_patches = patch_pos_embed.shape[1]
                orig_size = int(math.sqrt(orig_num_patches))

                # Calculate new patch grid size from actual image size
                H_patch = H // self.patch_size
                W_patch = W // self.patch_size

                patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                patch_pos_embed = F.interpolate(
                    patch_pos_embed, size=(H_patch, W_patch), mode='bicubic', align_corners=False
                )
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, H_patch * W_patch, -1)

                # Trim or pad if needed
                new_num_patches = H_patch * W_patch
                if patch_pos_embed.shape[1] != new_num_patches:
                    if patch_pos_embed.shape[1] > new_num_patches:
                        patch_pos_embed = patch_pos_embed[:, :new_num_patches, :]
                    else:
                        pad_len = new_num_patches - patch_pos_embed.shape[1]
                        pad = torch.zeros(1, pad_len, patch_pos_embed.shape[2], device=patch_pos_embed.device)
                        patch_pos_embed = torch.cat([patch_pos_embed, pad], dim=1)

                new_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
                x = x + new_pos_embed

        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Remove cls token and reshape to (B, C, H_patch, W_patch)
        x_spatial = x[:, 1:]
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        x_spatial = x_spatial.reshape(B, H_patch, W_patch, self.embed_dim)
        x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()

        # Return dict with output feature map, compatible with Detectron2 backbone API
        return {self.out_feature: x_spatial}

    @property
    def size_divisibility(self) -> int:
        # Enforce input sizes divisible by patch size
        return self.patch_size

    def output_shape(self):
        grid_size_h = self.img_size // self.patch_size
        grid_size_w = self.img_size // self.patch_size
        return {
            self.out_feature: ShapeSpec(
                channels=self.embed_dim,
                height=grid_size_h,
                width=grid_size_w,
                stride=self.patch_size,
            )
        }
