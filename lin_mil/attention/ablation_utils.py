import torch as torch
import torch.nn as nn

from .linear_attention import DropPath, Mlp, LinearAttention
from .softmax_attention import Attention
from .nystrom_attention import NystromAttention
from .dilated_attention import Dilated_Args, DilatedAttention

class LinearTransformerBlock_Ablation(nn.Module):
    """
    linear global attention block.
    """
    def __init__(self, dim, num_heads, attention,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=False):
        super().__init__()

        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        self.attention_type = attention

        if self.attention_type == "linear":
            self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
            
        elif self.attention_type == "softmax":
            self.attn = Attention(
                dim,
                heads=num_heads,
            )

        elif self.attention_type == "dilated":
            
            args = Dilated_Args()
            self.attn = DilatedAttention(
                embed_dim=dim,
                num_heads=num_heads,
                self_attention=True,
                args=args
            )

        elif self.attention_type == "nystrom":
            self.attn = NystromAttention(
                dim,
                heads=num_heads,
                num_landmarks=64,
            )
        else:
            raise ValueError(f"Invalid attention type: {self.attention_type}")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer)

    def forward(self, x):

        y = self.norm1(x)
    
        if self.attention_type == "dilated":
            self.attn.half() 
            y = y.half()
            y, attn = self.attn(y)
            y = y.float()
        
        else:
            y, attn = self.attn(y)

        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn