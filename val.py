from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from datasets.fusion_dataset import build_fusion_dataset
from losses import BDSFusionLoss
from metrics.fusion import fusion_metrics
from models.bdsfusion import BDSFusion
from utils import build_model_config, load_config, resolve_device


@torch.no_grad()
def validate(
    model: BDSFusion,
    loader: DataLoader,
    criterion: BDSFusionLoss,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {"loss": 0.0, "background_residual": 0.0, "scr": 0.0, "mae_to_max": 0.0}
    count = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        mwir = batch["mwir"].to(device)
        lwir = batch["lwir"].to(device)
        mask = batch["mask"].to(device)
        fused = model(mwir, lwir)
        loss, _ = criterion(mwir, lwir, fused, mask)
        values = fusion_metrics(mwir, lwir, fused, mask)
        totals["loss"] += float(loss.cpu())
        totals["background_residual"] += values["background_residual"]
        totals["scr"] += values.get("scr", 0.0)
        totals["mae_to_max"] += values["mae_to_max"]
        count += 1

    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BDSFusion on the configured test split.")
    parser.add_argument("--config", default="configs/bdsfusion.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    model = BDSFusion(build_model_config(config)).to(device)
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    criterion = BDSFusionLoss(**config.get("loss", {})).to(device)
    dataset = build_fusion_dataset(config, split=args.split)
    batch_size = int(config.get("train", {}).get("batch_size", 2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metrics = validate(model, loader, criterion, device, max_steps=args.steps)
    print(
        f"loss={metrics['loss']:.6f} "
        f"mae={metrics['mae_to_max']:.6f} "
        f"br={metrics['background_residual']:.6f} "
        f"scr={metrics['scr']:.6f}"
    )


if __name__ == "__main__":
    main()
