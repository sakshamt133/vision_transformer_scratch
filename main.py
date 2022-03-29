from block import Block
import torch.nn as nn
from Patch__embed import PatchEmbed
import torch


class VisionTransformer(nn.Module):
    def __init__(self, img_dim=32, in_channels=2, patch_dim=4, embed_dim=216, n_heads=6, attn_p=0.,
                 n_classes=2, qkv_bias=False, n_blocks=2):
        super(VisionTransformer, self).__init__()
        self.block = Block(
            embed_dim, n_heads, qkv_bias, attn_p, p=0., factor=2
        )
        self.layers = nn.ModuleList()
        self.n_layers = n_blocks
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.patch = PatchEmbed(in_channels, img_dim, patch_dim, embed_dim=embed_dim)
        self.position_embedding = nn.Parameter(
            torch.ones(1, 1+self.patch.n_patches, embed_dim)
        )
        self.final = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding

        for i in range(self.n_layers):
            x = self.block(x)

        x = x[:, 0]
        out = self.final(x)
        return out


img = torch.randn((1, 2, 32, 32))
vt = VisionTransformer()
out = vt(img)
print(out.shape)

