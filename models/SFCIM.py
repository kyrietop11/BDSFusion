# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : SFCIM.py
# @Software: VScode

from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from .SS2D import SS2D

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def _resolve_spatial_size(token_count: int, spatial_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if spatial_size is None:
        side = int(token_count ** 0.5)
        spatial_size = (side, side)
    h, w = int(spatial_size[0]), int(spatial_size[1])
    if h * w != token_count:
        raise ValueError(f"Token count {token_count} does not match spatial size {(h, w)}.")
    return h, w


class DifferentiableChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c % self.groups != 0:
            raise ValueError("channels must be divisible by groups")
        x = x.view(b, self.groups, c // self.groups, h, w)
        return x.transpose(1, 2).contiguous().view(b, c, h, w)


class FIM(nn.Module):
    def __init__(self, channels):
        super(FIM, self).__init__()
        self.conv1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.shuffle = DifferentiableChannelShuffle(2)

    def forward(self, mid, long):
        _, _, H, W = mid.shape

        fft_mid = torch.fft.rfft2(mid + 1e-8, norm='backward')
        fft_long = torch.fft.rfft2(long + 1e-8, norm='backward')
        mid_amp = torch.abs(fft_mid)
        mid_pha = torch.angle(fft_mid)
        long_amp = torch.abs(fft_long)
        long_pha = torch.angle(fft_long)
        amp_fuse = self.relu1(self.conv1(torch.cat([mid_amp, long_amp], 1)))
        pha_fuse = self.relu2(self.conv2(torch.cat([mid_pha, long_pha], 1)))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        fuse = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        shallow_fuse = self.shuffle(fuse)

        return shallow_fuse


class SFCIM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.FIM = FIM(hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.ln_3 = norm_layer(hidden_dim)
        self.ln_4 = norm_layer(hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.Mamba = SS2D(d_model=hidden_dim, d_state=d_state)
        self.drop_path = DropPath(drop_path)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def _mamba_bhwc(self, feature: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return self.Mamba(norm(feature))

    @staticmethod
    def _sequence_to_spatial_feature(tokens: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = spatial_size
        if h * w != n:
            raise ValueError(f"Token count {n} does not match spatial size {spatial_size}.")
        return tokens.transpose(1, 2).contiguous().reshape(b, c, h, w)

    @staticmethod
    def _bhwc_to_tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.permute(0, 3, 1, 2).contiguous().flatten(2).transpose(1, 2).contiguous()

    def forward(
            self,
            Feature_M: torch.Tensor,
            Feature_L: torch.Tensor,
            spatial_size: Optional[Tuple[int, int]] = None,
    ):
        B, N, C = Feature_M.shape
        H, W = _resolve_spatial_size(N, spatial_size)
        Feature_M = self._sequence_to_spatial_feature(Feature_M, (H, W))
        Feature_L = self._sequence_to_spatial_feature(Feature_L, (H, W))

        Feature_F = self.FIM(Feature_M, Feature_L)

        Feature_M = Feature_M.permute(0, 2, 3, 1).contiguous()
        Feature_L = Feature_L.permute(0, 2, 3, 1).contiguous()
        Feature_F = Feature_F.permute(0, 2, 3, 1).contiguous()

        Difference_Feature_M = Feature_F - Feature_M
        Difference_Feature_L = Feature_F - Feature_L

        Feature_M1 = Feature_M + self.drop_path(self._mamba_bhwc(Feature_M, self.ln_1))
        Feature_L1 = Feature_L + self.drop_path(self._mamba_bhwc(Feature_L, self.ln_2))
        Difference_Feature_M1 = Difference_Feature_M + self.drop_path(self._mamba_bhwc(Difference_Feature_M, self.ln_3))
        Difference_Feature_L1 = Difference_Feature_M + self.drop_path(self._mamba_bhwc(Difference_Feature_L, self.ln_4))

        Hybrid_Feature_M = Feature_M1 + self.alpha * Difference_Feature_M1 + Feature_L
        Hybrid_Feature_L = Feature_L1 + self.beta * Difference_Feature_L1 + Feature_M

        final_output = self.linear(torch.cat([Hybrid_Feature_M, Hybrid_Feature_L], dim=-1))
        final_output = self._bhwc_to_tokens(final_output)
        Hybrid_Feature_M = self._bhwc_to_tokens(Hybrid_Feature_M)
        Hybrid_Feature_L = self._bhwc_to_tokens(Hybrid_Feature_L)

        return Hybrid_Feature_M, Hybrid_Feature_L, final_output
