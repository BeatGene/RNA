"""ABRefine SO(2) graph attention output head: spherical latents -> [N,3]."""
from torch import nn


class ABRefineVectorHead(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, context, z):
        spherical, edge_features, edge_index = context
        return self.attention(spherical, z, edge_features, edge_index).embedding[:, 1:4, 0]
