import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

from SS2D import SS2D

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"



# Light-weight Deformable Convolution
class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param=3, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv2d(
                inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias
            ),
            nn.BatchNorm2d(outc),
            nn.SiLU(),
        )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(
            inc, 2 * num_param, kernel_size=3, padding=1, stride=stride
        )
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number), torch.arange(0, base_int)
        )
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1), torch.arange(0, mod_number)
            )

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride),
        )

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, "b c h w n -> b c (h n) w")
        return x_offset


# Convolutional Mamba Block

class Conv_Mamba(nn.Module):
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
        self.ln_1 = norm_layer(hidden_dim // 2)
        self.Mamba = SS2D(d_model=hidden_dim // 2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        #  Conv + BN + Relu
        self.LDConvLayer = nn.Sequential(
            LDConv(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            LDConv(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            LDConv(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self, input: torch.Tensor):
        # Before Mamba, change to B H W C
        input = input.permute(0, 2, 3, 1).contiguous()
        input_left, input_right = input.chunk(2, dim=-1)
        output_right = input_right + self.drop_path(self.Mamba(self.ln_1(input_right)))
        # Before convolution operation, change to B C H W
        input_left = input_left.permute(0, 3, 1, 2).contiguous()
        output_left = self.LDConvLayer(input_left)
        output_right = output_right.permute(0, 3, 1, 2).contiguous()
        output = torch.cat((output_left, output_right), dim=1)
        output = self.shuffle(output).permute(0, 2, 3, 1).contiguous()
        # Residual connection
        final_output = output + input
        final_output = final_output.permute(0, 3, 1, 2).contiguous()
        return final_output


# Channel Self-Attention Block
class Channel_Self_Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int = 4,
            window_size: int = 7,
            group_kernel_sizes: list = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: dict = dict(type='BN'),
            act_cfg: dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'Sigmoid',
    ):
        super(Channel_Self_Attention, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.conv_d = nn.Identity()
        self.LayerNorm = nn.LayerNorm(dim,  eps = 1e-06)
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
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        y = self.down_func(input)
        y = self.conv_d(y)
        B, C, H, W = y.size()

        # reshape -> (B, H, W, C) -> (B, C, H * W) to generate q, k, v
        y = y.permute(0, 2, 3, 1) 
        y = self.LayerNorm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        q = q.permute(0, 3, 1, 2).contiguous()
        k = k.permute(0, 3, 1, 2).contiguous()
        v = v.permute(0, 3, 1, 2).contiguous()
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(H), w=int(W))
        
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        
        # 
        output = attn*input + input
        return output


# Convolutional Mamba with Channel Self-Attention

class CSAMamba(nn.Module):
    def __init__(self, in_channels):
        super(CSAMamba, self).__init__()
        self.Conv_Mamba = Conv_Mamba(in_channels)
        self.Channel_Self_Attention = Channel_Self_Attention(in_channels)
    def forward(self, x):
        B, N, C = x.shape
        H, W = int(N**0.5), int(N**0.5)
        assert H * W == N, "H and W must satisfy H * W = N"
        x = x.view(B, C, H, W)
        CM_out = self.Conv_Mamba(x)
        CSAMamba_out = self.Channel_Self_Attention(CM_out)
        CSAMamba_out = CSAMamba_out.view(CSAMamba_out.shape[0],CSAMamba_out.shape[2]*CSAMamba_out.shape[3],CSAMamba_out.shape[1])
        return CSAMamba_out

if __name__ == '__main__':
    x = torch.randn(8, 16384, 60).cuda()
    # model = SS2D(d_model = 64, d_state = 16, d_conv=3, expand=2).cuda()
    model = CSAMamba(x.shape[2]).cuda()


