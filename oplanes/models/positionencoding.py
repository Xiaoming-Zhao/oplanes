import math

import torch
from torch import nn


class PositionEmbeddingSimple(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, scale=1.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self, x):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        pos_x = x * self.scale
        pos_x = pos_x.unsqueeze(-1) / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(x.dim())
        return pos_x
