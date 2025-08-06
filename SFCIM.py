from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
from SS2D import SS2D

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


    
class FIM(nn.Module):
    def __init__(self, channels):
        super(FIM, self).__init__()
        self.conv1 = nn.Conv2d(channels*2,channels,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(channels*2,channels,kernel_size=1,stride=1,padding=0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self, mid, long):

        _, _, H, W = mid.shape

        # Frequency domain part
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

        # Merge
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
        self.Mamba = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.alpha = nn.Parameter(torch.Tensor([1.0]).cuda(), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([1.0]).cuda(), requires_grad=True)
    def forward(self, Feature_M: torch.Tensor, Feature_L: torch.Tensor):
        # B N C --> B C H W
        B, N, C = Feature_M.shape
        H, W = int(N ** 0.5), int(N ** 0.5)
        assert H * W == N, "H and W must satisfy H * W = N"
        Feature_M = Feature_M.view(B, C, H, W)
        Feature_L = Feature_L.view(B, C, H, W)

        # Frequency domain fusion feature
        Feature_F = self.FIM(Feature_M, Feature_L)

        # Before Mamba module processing, change to B H W C
        Feature_M = Feature_M.permute(0, 2, 3, 1).contiguous()
        Feature_L = Feature_L.permute(0, 2, 3, 1).contiguous()
        Feature_F = Feature_F.permute(0, 2, 3, 1).contiguous()

        # Subtract from frequency domain feature to get difference feature
        Difference_Feature_M = Feature_F - Feature_M
        Difference_Feature_L = Feature_F - Feature_L

        # Feature_M and Feature_L are first processed by LN and then passed through the SS2D block to capture long-range dependencies
        Feature_M1 = Feature_M + self.drop_path(self.Mamba(self.ln_1(Feature_M)))
        Feature_L1 = Feature_L + self.drop_path(self.Mamba(self.ln_2(Feature_L)))
        Difference_Feature_M1 = Difference_Feature_M + self.drop_path(self.Mamba(self.ln_3(Difference_Feature_M)))
        Difference_Feature_L1 = Difference_Feature_M + self.drop_path(self.Mamba(self.ln_4(Difference_Feature_L)))

        # Get enhanced modal features with frequency domain features and original features of the other modality
        Hybrid_Feature_M = Feature_M1 + self.alpha * Difference_Feature_M1 + Feature_L
        Hybrid_Feature_L = Feature_L1 + self.beta * Difference_Feature_L1 + Feature_M

        # Concatenate + Linear Projection --> B C H W
        final_output = self.linear(torch.cat([Hybrid_Feature_M, Hybrid_Feature_L], dim=-1))
        final_output = final_output.permute(0, 3, 1, 2).contiguous()
        # B N C --> B C H W
        final_output = final_output.view(B, N, C)
        Hybrid_Feature_M = Hybrid_Feature_M.view(B, N, C)
        Hybrid_Feature_L = Hybrid_Feature_L.view(B, N, C)

        return Hybrid_Feature_M, Hybrid_Feature_L, final_output


if __name__ == '__main__':
    # x1 = torch.randn(8, 64, 128, 128).cuda()
    # x2 = torch.randn(8, 64, 128, 128).cuda()
    x1 = torch.randn(8, 16384, 60).cuda()
    x2 = torch.randn(8, 16384, 60).cuda()
    model = SFCIM(hidden_dim=x1.shape[2]).cuda()
