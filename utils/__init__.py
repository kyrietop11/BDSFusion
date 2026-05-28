from __future__ import annotations

import torch

from .config import build_model_config, load_config, merge_dict


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


__all__ = ["build_model_config", "load_config", "merge_dict", "resolve_device"]
