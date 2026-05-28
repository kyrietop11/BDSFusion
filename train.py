# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : train.py
# @Software: VScode

from __future__ import annotations

import argparse
import logging
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.fusion_dataset import build_fusion_dataset
from losses import BDSFusionLoss
from metrics.fusion import fusion_metrics
from models import BDSFusion, BDSFusionConfig
from utils import build_model_config, load_config, resolve_device


logger = logging.getLogger("BDSFusion")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 1e-4))
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    train_cfg = config.get("train", {})
    epochs = int(train_cfg.get("epochs", 100))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def train_one_epoch(
    model: BDSFusion,
    loader: DataLoader,
    criterion: BDSFusionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict,
) -> Dict[str, float]:
    model.train()
    train_cfg = config.get("train", {})
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_interval = int(train_cfg.get("log_interval", 50))

    running_loss = 0.0
    running_components: Dict[str, float] = {}
    count = 0

    for step, batch in enumerate(loader):
        mwir = batch["mwir"].to(device)
        lwir = batch["lwir"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        fused = model(mwir, lwir)
        loss, components = criterion(mwir, lwir, fused, mask)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        for k, v in components.items():
            running_components[k] = running_components.get(k, 0.0) + v.item()
        count += 1

        if log_interval > 0 and (step + 1) % log_interval == 0:
            avg = running_loss / count
            logger.info(f"  [Epoch {epoch} Step {step + 1}/{len(loader)}] loss={avg:.6f}")

    denom = max(count, 1)
    metrics = {"loss": running_loss / denom}
    for k, v in running_components.items():
        metrics[k] = v / denom
    return metrics


@torch.no_grad()
def validate(
    model: BDSFusion,
    loader: DataLoader,
    criterion: BDSFusionLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_components: Dict[str, float] = {}
    running_scr = 0.0
    running_br = 0.0
    count = 0

    for batch in loader:
        mwir = batch["mwir"].to(device)
        lwir = batch["lwir"].to(device)
        mask = batch["mask"].to(device)
        fused = model(mwir, lwir)
        loss, components = criterion(mwir, lwir, fused, mask)
        running_loss += loss.item()
        for k, v in components.items():
            running_components[k] = running_components.get(k, 0.0) + v.item()
        values = fusion_metrics(mwir, lwir, fused, mask if mask.sum() > 0 else None)
        running_scr += float(values.get("scr", 0.0))
        running_br += float(values.get("background_residual", 0.0))
        count += 1

    denom = max(count, 1)
    metrics = {"loss": running_loss / denom}
    for k, v in running_components.items():
        metrics[k] = v / denom
    metrics["scr"] = running_scr / denom
    metrics["background_residual"] = running_br / denom
    return metrics


def save_checkpoint(
    model: BDSFusion,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    save_path: Path,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDSFusion")
    parser.add_argument("--config", type=str, default="configs/bdsfusion.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Override work directory. Default: results/BDSFusion/<timestamp>")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = Path("results") / "BDSFusion" / timestamp
    work_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, work_dir / Path(args.config).name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(work_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)
    logger.info(f"Device: {device} | Seed: {seed}")

    model_config = build_model_config(config)
    model = BDSFusion(model_config).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {param_count:.4f}M")

    criterion = BDSFusionLoss(**config.get("loss", {})).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    start_epoch = 0
    best_loss = float("inf")
    best_scr = 0.0
    best_br = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("metrics", {}).get("loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    train_dataset = build_fusion_dataset(config, split="train")
    val_dataset = build_fusion_dataset(config, split="test")

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    epochs = int(train_cfg.get("epochs", 100))
    save_interval = int(train_cfg.get("save_interval", 10))
    logger.info(f"Training for {epochs} epochs, batch_size={batch_size}, lr={train_cfg.get('lr', 1e-4)}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) lr={lr_now:.2e} | "
            f"train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}"
        )

        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, work_dir / "last.pth")

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, work_dir / "best_loss.pth")
            logger.info(f"  New best (loss) saved: val_loss={best_loss:.6f}")

        if val_metrics.get("scr", 0.0) > best_scr:
            best_scr = val_metrics["scr"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, work_dir / "best_scr.pth")
            logger.info(f"  New best (scr) saved: val_scr={best_scr:.6f}")

        if val_metrics.get("background_residual", float("inf")) < best_br:
            best_br = val_metrics["background_residual"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, work_dir / "best_br.pth")
            logger.info(f"  New best (br) saved: val_br={best_br:.6f}")

        if epoch % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, work_dir / f"epoch_{epoch}.pth")

    logger.info(
        f"Training complete. Best val_loss={best_loss:.6f} | "
        f"best val_scr={best_scr:.6f} | best val_br={best_br:.6f}"
    )


if __name__ == "__main__":
    main()
