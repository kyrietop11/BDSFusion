# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : CSAMamba.py
# @Software: VScode

import math
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
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


class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param=3, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU(),
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
        ], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
        ], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1),
        ], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = rearrange(x_offset, "b c h w n -> b c (h n) w")
        return self.conv(x_offset)

    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number, device=self.p_conv.weight.device),
            torch.arange(0, base_int, device=self.p_conv.weight.device), indexing="ij")
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1, device=self.p_conv.weight.device),
                torch.arange(0, mod_number, device=self.p_conv.weight.device), indexing="ij")
            p_n_x = torch.cat((torch.flatten(p_n_x), torch.flatten(mod_p_n_x)))
            p_n_y = torch.cat((torch.flatten(p_n_y), torch.flatten(mod_p_n_y)))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        return p_n.view(1, 2 * N, 1, 1).type(dtype)

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride, device=self.p_conv.weight.device),
            torch.arange(0, w * self.stride, self.stride, device=self.p_conv.weight.device), indexing="ij")
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        return torch.cat([p_0_x, p_0_y], 1).type(dtype)

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        return p_0 + p_n + offset

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        return x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)


class Conv_Mamba(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 24,
            num_ldconv: int = 2,
            ldconv_num_param: int = 3,
            **kwargs,
    ):
        super().__init__()
        half = hidden_dim // 2
        self.ln_1 = norm_layer(half)
        self.Mamba = SS2D(d_model=half, d_state=d_state, d_conv=3, expand=2, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path)
        layers = []
        for _ in range(num_ldconv):
            layers.extend([LDConv(half, half, num_param=ldconv_num_param), nn.BatchNorm2d(half), nn.ReLU()])
        self.LDConvLayer = nn.Sequential(*layers)
        self.shuffle = DifferentiableChannelShuffle(2)

    def forward(self, input: torch.Tensor):
        input = input.permute(0, 2, 3, 1).contiguous()
        input_left, input_right = input.chunk(2, dim=-1)
        output_right = input_right + self.drop_path(self.Mamba(self.ln_1(input_right)))
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        output_left = self.LDConvLayer(input_left)
        output_right = output_right.permute(0, 3, 1, 2).contiguous()
        output = torch.cat((output_left, output_right), dim=1)
        output = self.shuffle(output).permute(0, 2, 3, 1).contiguous()
        final_output = output + input
        return final_output.permute(0, 3, 1, 2).contiguous()


class Channel_Self_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int = 4,
            window_size: int = 7,
            qkv_bias: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'Sigmoid',
    ):
        super(Channel_Self_Attention, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size
        self.down_sample_mode = down_sample_mode

        self.conv_d = nn.Identity()
        self.LayerNorm = nn.LayerNorm(dim, eps=1e-06)
        self.q = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.k = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.v = nn.Linear(in_features=dim, out_features=dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        if gate_layer == 'SiLU':
            self.ca_gate = nn.SiLU()
        elif gate_layer == 'Softmax':
            self.ca_gate = nn.Softmax(dim=1)
        else:
            self.ca_gate = nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.down_func(input)
        y = self.conv_d(y)
        B, C, H, W = y.size()

        y = y.permute(0, 2, 3, 1)
        y = self.LayerNorm(y)
        q = self.q(y).permute(0, 3, 1, 2).contiguous()
        k = self.k(y).permute(0, 3, 1, 2).contiguous()
        v = self.v(y).permute(0, 3, 1, 2).contiguous()
        q = rearrange(q, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        k = rearrange(k, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)
        v = rearrange(v, 'b (hn hd) h w -> b hn hd (h w)', hn=self.head_num, hd=self.head_dim)

        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v
        attn = rearrange(attn, 'b hn hd (h w) -> b (hn hd) h w', h=H, w=W)

        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * input + input


class CSAMamba(nn.Module):
    def __init__(
            self,
            in_channels: int,
            head_num: int = 4,
            window_size: int = 7,
            drop_path: float = 0.0,
            attn_drop_rate: float = 0.0,
            d_state: int = 24,
            num_ldconv: int = 2,
            ldconv_num_param: int = 3,
    ):
        super(CSAMamba, self).__init__()
        self.Conv_Mamba = Conv_Mamba(
            in_channels, drop_path=drop_path, attn_drop_rate=attn_drop_rate,
            d_state=d_state, num_ldconv=num_ldconv, ldconv_num_param=ldconv_num_param)
        self.Channel_Self_Attention = Channel_Self_Attention(
            in_channels, head_num=head_num, window_size=window_size,
            attn_drop_ratio=attn_drop_rate)

    def forward(self, x, spatial_size: Optional[Tuple[int, int]] = None):
        B, N, C = x.shape
        H, W = _resolve_spatial_size(N, spatial_size)
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)
        x = self.Conv_Mamba(x)
        x = self.Channel_Self_Attention(x)
        return x.flatten(2).transpose(1, 2).contiguous()
