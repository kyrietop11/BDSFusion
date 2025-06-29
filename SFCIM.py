import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile, clever_format
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
# an alternative for mamba_ssm (in which causal_conv1d is needed)
# try:
#     from selective_scan import selective_scan_fn as selective_scan_fn_v1
#     from selective_scan import selective_scan_ref as selective_scan_ref_v1
# except:
#     pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
    
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
