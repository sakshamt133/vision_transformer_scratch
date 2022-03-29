import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=2, img_dim=32, patch_dim=4, embed_dim=216):
        super(PatchEmbed, self).__init__()
        self.n_patches = (img_dim // patch_dim) ** 2
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_dim, patch_dim), stride=(patch_dim, patch_dim))

    def forward(self, x):
        # X SHAPE IS (batch, in_channels, img_dim,  img_dim )
        x = self.conv(x)
        # new shape of x is going to be (batch, embed_dim , self.n_patches, self.n_patches)
        x = x.reshape(x.shape[0], self.embed_dim, -1)
        x = x.transpose(2, 1)
        return x
