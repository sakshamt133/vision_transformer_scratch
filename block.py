import torch.nn as nn
from Attention import Attention
from mlp import MLP


class Block(nn.Module):
    def __init__(self, embed_dim=216, n_heads=6, qkv_bias=False, attn_p=0.0, p=0.0, factor=2):
        super(Block, self).__init__()
        self.attention = Attention(embed_dim, n_heads, qkv_bias,  attn_p)
        self.mlp = MLP(embed_dim, factor)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out = x + self.norm1(self.attention(x, x, x))
        out = x + self.norm2(self.mlp(out))
        return out
