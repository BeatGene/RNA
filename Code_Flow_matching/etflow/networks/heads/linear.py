"""Scalar-gated equivariant vector projection shared by both backends."""
from torch import nn


class LinearVectorHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.vector = nn.Linear(channels, 1, bias=False)

    def forward(self, scalar, vector):
        return self.vector(vector * self.gate(scalar).unsqueeze(1)).squeeze(-1)
