import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, embed_dim=216, n_heads=6, qkv_bias=False, atttn_p=0.):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = (embed_dim//n_heads)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.drop = nn.Dropout2d(atttn_p)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, key, query, value):
        # key ,query, value shape is [batch, n_pathes +1, embed_dim]
        key = self.k_linear(key).reshape(key.shape[0], self.n_heads, key.shape[1], self.head_dim)
        query = self.q_linear(query).reshape(query.shape[0], self.n_heads, query.shape[1], self.head_dim)
        value = self.v_linear(value).reshape(value.shape[0], self.n_heads, value.shape[1], self.head_dim)
        attn_score = torch.einsum("bnqd, bnkd -> bnqk", query, key)
        attn_soft = self.soft(attn_score)
        out = torch.einsum("bnqk, bnvd -> bnqd", attn_soft, value)
        out = self.drop(out)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        return out


a = Attention()
img = torch.randn((1, 65, 216))
a(img, img, img)