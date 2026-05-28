# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : saliency.py
# @Software: VScode

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def admd_response(image: torch.Tensor) -> torch.Tensor:
    """Differentiable ADMD-style directional response used for BR loss/metric."""

    gray = image.mean(dim=1, keepdim=True)
    kernels = []
    base = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]], device=image.device, dtype=image.dtype)
    kernels.extend([base, base.t()])
    kernels.extend(
        [
            torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -1.0]], device=image.device, dtype=image.dtype),
            torch.tensor([[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]], device=image.device, dtype=image.dtype),
        ]
    )
    weight = torch.stack(kernels).unsqueeze(1) / 3.0
    response = F.conv2d(gray, weight, padding=1).abs()
    return response.max(dim=1, keepdim=True).values


def admd_residual_ratio(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable ADMD residual ratio with straight-through sigmoid surrogate."""

    response = admd_response(image)
    threshold = response.mean(dim=(-2, -1), keepdim=True) + response.std(dim=(-2, -1), keepdim=True)
    scale = response.std(dim=(-2, -1), keepdim=True).detach().clamp_min(eps)
    soft_detection = torch.sigmoid((response - threshold) / scale)
    hard_detection = (response > threshold).to(dtype=image.dtype)
    detection = hard_detection.detach() + soft_detection - soft_detection.detach()
    return detection.mean()


def haar_smooth_background(image: torch.Tensor) -> torch.Tensor:
    """Haar low-frequency background reconstruction."""

    h, w = image.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    ll = F.avg_pool2d(image, kernel_size=2, stride=2)
    smooth = ll.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return smooth[..., :h, :w]


class L_Intensity(nn.Module):
    def forward(self, image_A: torch.Tensor, image_B: torch.Tensor, image_fused: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(image_fused, torch.maximum(image_A, image_B))


class L_Background_Residue(nn.Module):
    def __init__(self, target: float = 0.0) -> None:
        super().__init__()
        self.target = float(target)

    def forward(self, _image_A: torch.Tensor, _image_B: torch.Tensor, image_fused: torch.Tensor) -> torch.Tensor:
        br = admd_residual_ratio(image_fused)
        return torch.abs(br - self.target)


class L_MaskFusionL1loss(nn.Module):
    def forward(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        image_fused: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_t = (mask > 0).to(dtype=image_fused.dtype)
        mask_b = 1.0 - mask_t
        target_m = image_A[:, :1] * mask_t
        background_l = image_B[:, :1] * mask_b
        expected_img = target_m + haar_smooth_background(background_l)
        return F.l1_loss(image_fused, expected_img, reduction="mean")


class DetectionDrivenSaliencyLoss(nn.Module):
    """Detection-driven saliency loss: intensity + background residual + Haar wavelet mask."""

    def __init__(
        self,
        lambda_intensity: float = 20.0,
        lambda_background: float = 500.0,
        lambda_haar: float = 10.0,
        background_target: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_intensity = float(lambda_intensity)
        self.lambda_background = float(lambda_background)
        self.lambda_haar = float(lambda_haar)
        self.L_Inten = L_Intensity()
        self.L_Background_Residue = L_Background_Residue(target=background_target)
        self.L_MaskFusionL1loss = L_MaskFusionL1loss()

    def forward(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
        image_fused: torch.Tensor,
        image_M: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_l1 = self.lambda_intensity * self.L_Inten(image_A, image_B, image_fused)
        loss_br = self.lambda_background * self.L_Background_Residue(image_A, image_B, image_fused)
        loss_mf = self.lambda_haar * self.L_MaskFusionL1loss(image_A, image_B, image_fused, image_M)
        return loss_l1 + loss_br + loss_mf, loss_l1, loss_br, loss_mf


class BDSFusionLoss(DetectionDrivenSaliencyLoss):
    """Training-friendly wrapper returning named loss components."""

    def forward(
        self,
        mwir: torch.Tensor,
        lwir: torch.Tensor,
        fused: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total, loss_l1, loss_br, loss_mf = super().forward(mwir, lwir, fused, target_mask)
        return total, {
            "intensity": (loss_l1 / self.lambda_intensity).detach(),
            "background": (loss_br / self.lambda_background).detach(),
            "haar_mask": (loss_mf / self.lambda_haar).detach(),
            "total": total.detach(),
        }
