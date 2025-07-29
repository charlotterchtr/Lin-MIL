import torch as torch
import torch.nn as nn
from einops import repeat

from .attention.linear_attention import LinearTransformerBlock
from .attention.ablation_utils import LinearTransformerBlock_Ablation


class Config:
    def __init__(self, latent_dim=512, transformer_depth=4, dropout=0., emb_dropout=0.1, act='ReLU', attention='linear', pooling='cls', ablation=False):
        self.latent_dim = latent_dim
        self.transformer_depth = transformer_depth
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.act = act
        self.attention = attention
        self.pooling = pooling
        self.ablation = ablation

class Lin_MIL(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        input_dim,
        config, 
        heads=8,
        pos_enc=None,
    ):
        super(Lin_MIL, self).__init__()

        self.dim = config.latent_dim
        self.pos_enc = pos_enc
        
        dropout = config.dropout
        emb_dropout = config.emb_dropout
        
        if config.act is not None:
            if config.act == "GeLU":
                act = nn.GELU
            elif config.act == "ReLU":
                act = nn.ReLU
            elif config.act == "ELU":
                act = nn.ELU
            elif config.act == "LeakyReLU":
                act = nn.LeakyReLU
            elif config.act == "TanH":
                act = nn.Tanh
            elif config.act == "Softplus":
                act = nn.Softplus
            else:
                raise ValueError(f"Invalid activation function: {config.act}")

        self.projection = nn.Linear(input_dim, self.dim, bias=True)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.depth = config.transformer_depth
        if config.ablation:
            self.transformer = nn.ModuleList([LinearTransformerBlock_Ablation(dim=self.dim, num_heads=heads, attention = config.attention, qkv_bias=True, drop_path=dropout) for _ in range(self.depth)])
        else:
            self.transformer = nn.ModuleList([LinearTransformerBlock(dim=self.dim, num_heads=heads, qkv_bias=True, drop_path=dropout, att_act = act) for _ in range(self.depth)])
        
        self.pool = config.pooling
        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        self.norm = nn.LayerNorm(self.dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes))

    def forward(self, x, coords = None):
        
        b, n, _ = x.shape

        x = self.projection(x)

        if self.pos_enc is not None:
                x = self.pos_enc(x, coords)

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x)

        for layer in self.transformer:
            x, att = layer(x)
        
        # average attention across heads and latent dimension
        att = att.mean(dim=1).mean(dim=-1) if att is not None else None
        
        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]
            att = att[:, 1:] if att is not None else None
        else:
            raise ValueError(f"Invalid pooling type: {self.pool}")
        
        return self.mlp_head(self.norm(x)), att