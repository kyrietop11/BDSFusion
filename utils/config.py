# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : config.py
# @Software: VScode

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from models.bdsfusion import BDSFusionConfig


def merge_dict(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_dict(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if overrides:
        cfg = merge_dict(cfg, overrides)
    return cfg


def build_model_config(config: Mapping[str, Any]) -> BDSFusionConfig:
    model_cfg = dict(config.get("model", {}))
    if "depths" in model_cfg and isinstance(model_cfg["depths"], list):
        model_cfg["depths"] = tuple(model_cfg["depths"])
    return BDSFusionConfig(**model_cfg)
