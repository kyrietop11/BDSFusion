# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : fusion.py
# @Software: VScode

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.ndimage as filters
import torch
import torch.nn.functional as F

from losses import admd_residual_ratio


def _admd_subfubc(img: np.ndarray, length: int, threshold: float) -> np.ndarray:
    weights = np.ones((length, length)) / (length * length)
    cf_img = filters.convolve(img, weights, mode="mirror")
    lx = ly = length // 2
    op = np.zeros((length * 3, length * 3))
    op[ly, lx] = 1
    op[ly, length + lx] = 1
    op[ly, length * 2 + lx] = 1
    op[length + ly, lx] = 1
    op[length + ly, length * 2 + lx] = 1
    op[length * 2 + ly, lx] = 1
    op[length * 2 + ly, length + lx] = 1
    op[length * 2 + ly, length * 2 + lx] = 1
    mf_img = filters.maximum_filter(cf_img, footprint=op)
    diff = cf_img - mf_img
    out = (diff ** 2) * (diff > 0)
    return (out >= threshold).astype(np.float32)


@torch.no_grad()
def admd_traditional(image: torch.Tensor, threshold: float = 200.0) -> torch.Tensor:
    """Traditional multi-scale morphological ADMD background-residual ratio.

    Operates on each batch element independently with images normalised to [0, 255].
    Returns a 0-D tensor with the mean BR across the batch.
    """
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[:, 0]
    elif arr.ndim == 3:
        arr = arr[None, 0]
    ratios = []
    for single in arr:
        s_min, s_max = float(single.min()), float(single.max())
        scale = (s_max - s_min) if (s_max - s_min) > 1e-8 else 1.0
        normed = (single - s_min) / scale * 255.0
        outs = [_admd_subfubc(normed.astype(float), L, threshold) for L in (3, 5, 7, 9)]
        merged = np.max(np.stack(outs, axis=2), axis=2)
        ratios.append(float(merged.sum()) / float(merged.shape[0] * merged.shape[1]))
    return torch.tensor(np.mean(ratios), dtype=image.dtype)


@torch.no_grad()
def background_residual(fused: torch.Tensor) -> torch.Tensor:
    return admd_residual_ratio(fused).detach()


@torch.no_grad()
def background_residual_eval(image: torch.Tensor, threshold: float = 200.0) -> torch.Tensor:
    """Paper-style BR for evaluation (multi-scale morphological ADMD)."""
    return admd_traditional(image, threshold=threshold)


@torch.no_grad()
def signal_clutter_ratio(image: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = (mask > 0).to(dtype=image.dtype)
    background = 1.0 - mask
    target_count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    bg_count = background.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    target_mean = (image * mask).sum(dim=(-2, -1), keepdim=True) / target_count
    bg_values = image * background
    bg_mean = bg_values.sum(dim=(-2, -1), keepdim=True) / bg_count
    bg_var = (((image - bg_mean) * background) ** 2).sum(dim=(-2, -1), keepdim=True) / bg_count
    return ((target_mean - bg_mean).abs() / (bg_var.sqrt() + eps)).mean()


@torch.no_grad()
def fusion_metrics(
    mwir: torch.Tensor,
    lwir: torch.Tensor,
    fused: torch.Tensor,
    mask: torch.Tensor | None = None,
    eval_mode: bool = False,
) -> Dict[str, float]:
    br_fn = background_residual_eval if eval_mode else background_residual
    metrics = {
        "mae_to_max": F.l1_loss(fused, torch.maximum(mwir, lwir)).item(),
        "background_residual": br_fn(fused).item(),
    }
    if mask is not None:
        metrics["scr"] = signal_clutter_ratio(fused, mask).item()
    return metrics
