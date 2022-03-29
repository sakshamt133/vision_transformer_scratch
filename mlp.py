import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=216, factor=2):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features, factor * in_features)
        self.act = nn.GELU()
        self.l2 = nn.Linear(factor * in_features, in_features)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return x

