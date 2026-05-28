# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : test.py
# @Software: VScode

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from datasets.fusion_dataset import BDSFusionDataset
from metrics.fusion import fusion_metrics
from models import BDSFusion
from utils import build_model_config, load_config, resolve_device


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = tensor.shape
    h_pad = (multiple - h % multiple) % multiple
    w_pad = (multiple - w % multiple) % multiple
    if h_pad == 0 and w_pad == 0:
        return tensor, (h, w)
    padded = F.pad(tensor, (0, w_pad, 0, h_pad), mode="reflect")
    return padded, (h, w)


def save_fused_image(fused: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    array = fused.clamp(0.0, 1.0).squeeze().cpu().numpy()
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BDSFusion inference.")
    parser.add_argument("--config", type=str, default="configs/bdsfusion_train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data.root in config (test split is used).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save-dir", type=str, default="results/fused")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patch-multiple", type=int, default=None,
                        help="Pad H/W to a multiple of this value (defaults to model patch_size).")
    parser.add_argument("--measure-fps", action="store_true",
                        help="Report mean inference time and FPS.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_config = build_model_config(config)
    model = BDSFusion(model_config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded checkpoint with {param_count:.4f}M parameters from {args.checkpoint}")

    multiple = args.patch_multiple or int(model_config.patch_size) * 2

    data_root = args.data_root or config.get("data", {}).get("root")
    if data_root is None:
        raise ValueError("data.root not provided. Pass --data-root or set it in the config.")

    dataset = BDSFusionDataset(
        root=data_root,
        split=args.split,
        crop_size=None,
        augment=False,
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    metric_totals: Dict[str, float] = {}
    metric_count = 0
    timings = []

    with torch.no_grad():
        for sample in loader:
            mwir = sample["mwir"].to(device)
            lwir = sample["lwir"].to(device)
            mask = sample["mask"].to(device)
            stem = sample["stem"][0]

            mwir_p, (h, w) = pad_to_multiple(mwir, multiple)
            lwir_p, _ = pad_to_multiple(lwir, multiple)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            fused = model(mwir_p, lwir_p)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.time() - t0)

            fused = fused[..., :h, :w].clamp(0.0, 1.0)

            save_fused_image(fused[0], save_dir / f"{stem}.png")

            values = fusion_metrics(mwir, lwir, fused, mask if mask.sum() > 0 else None, eval_mode=True)
            for key, value in values.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + float(value)
            metric_count += 1

    print(f"Saved {metric_count} fused images to {save_dir}")
    if metric_count:
        for key, value in metric_totals.items():
            print(f"  mean_{key}={value / metric_count:.6f}")
    if args.measure_fps and timings:
        warm = timings[1:] if len(timings) > 1 else timings
        mean_t = sum(warm) / len(warm)
        print(f"  mean_inference_time={mean_t * 1000:.2f} ms | FPS={1.0 / max(mean_t, 1e-9):.4f}")


if __name__ == "__main__":
    main()
