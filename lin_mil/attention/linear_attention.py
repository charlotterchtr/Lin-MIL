import torch as torch
import torch.nn as nn
from timm.models.layers import DropPath

from typing import Optional
from torch import nn
from torch.nn import functional as F
import torch as torch
import math
from functools import wraps
from math import ceil, pi
from typing import Optional

import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum, nn

# linear attention implementation from https://github.com/ChuanyangZheng/L2ViT/blob/master/classification/models/l2vit.py

class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        return q @ k

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class LinearAttention(nn.Module):
    """
    the linear attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, att_act = nn.ReLU, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * self.scale)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_mul = Mul()
        self.kv_mul = Mul()
        self.extra_mul = Mul()
        self.act = att_act()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.act(q)
        k = self.act(k)
        denom = torch.clamp(self.qk_mul(q, k.transpose(-2, -1).sum(dim=-1, keepdim=True)), 1e2)
        attn = self.kv_mul(k.transpose(-2, -1), v) * self.temperature
        attn1 = self.extra_mul(q, attn)
        attn = attn1.clone() / denom.clone()

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn1

class LinearTransformerBlock(nn.Module):
    """
    linear attention block.
    """
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=False, att_act=nn.ReLU):
        super().__init__()

        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        
        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            att_act=att_act,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer)

    def forward(self, x):

        y = self.norm1(x)
        y, attn = self.attn(y)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
