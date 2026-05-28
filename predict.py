# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : predict.py
# @Software: VScode

from __future__ import annotations

import torch


@torch.no_grad()
def infer_tensor(
    model: torch.nn.Module,
    mwir: torch.Tensor,
    lwir: torch.Tensor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    model = model.to(device)
    was_training = model.training
    model.eval()
    fused = model(mwir.to(device), lwir.to(device)).detach().cpu()
    if was_training:
        model.train()
    return fused.clamp(0.0, 1.0)
