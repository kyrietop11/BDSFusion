# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : modules.py
# @Software: VScode

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SpatialSize = Tuple[int, int]


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, SpatialSize]:
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError("Input height and width must be divisible by patch_size.")
        y = self.proj(x)
        return y.flatten(2).transpose(1, 2).contiguous(), (y.shape[-2], y.shape[-1])


class PatchReconstruct(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int, patch_size: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, spatial_size: SpatialSize, output_size: Optional[SpatialSize] = None) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = spatial_size
        if h * w != n:
            raise ValueError(f"Token count {n} does not match spatial size {spatial_size}.")
        x = tokens.transpose(1, 2).contiguous().reshape(b, c, h, w)
        if self.patch_size > 1:
            x = F.interpolate(x, scale_factor=self.patch_size, mode="bilinear", align_corners=False)
        if output_size is not None and x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.refine(x))
