#!/usr/bin/env python3
"""
Train Seismic-Well CLIP model with InfoNCE contrastive loss.

Usage:
    python train.py --n-wells 200 --epochs 100 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/yxjiang/datatest")

from clip_seismic_well.config import CLIPConfig
from clip_seismic_well.models import SeismicWellCLIP
from clip_seismic_well.data.dataset import SeismicWellDataset, collate_pairs


def parse_args():
    p = argparse.ArgumentParser(description="Train Seismic-Well CLIP")
    p.add_argument("--dataset", type=str,
                   default="./openseisml_dataset_large.h5")
    p.add_argument("--n-wells", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="./clip_checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", action="store_true", default=True)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_one_epoch(model, dataloader, optimizer, scaler,
                    epoch: int, args, device):
    """Train one epoch with optional mixed precision."""
    model.train()
    total_loss = 0.0
    total_acc_s2w = 0.0
    total_acc_w2s = 0.0
    n_batches = len(dataloader)

    t0 = time.time()
    for bidx, batch in enumerate(dataloader):
        seismic = batch["seismic"].to(device, non_blocking=True)
        well_logs = batch["well_logs"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if args.mixed_precision:
            with autocast():
                loss, acc_s2w, acc_w2s, _ = model(seismic, well_logs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, acc_s2w, acc_w2s, _ = model(seismic, well_logs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc_s2w += acc_s2w.item()
        total_acc_w2s += acc_w2s.item()

        if bidx % 10 == 0:
            print(f"  Epoch {epoch:3d} | Batch {bidx:4d}/{n_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc s→w: {acc_s2w.item():.3f} | "
                  f"Acc w→s: {acc_w2s.item():.3f} | "
                  f"τ: {model.logit_scale.exp().item():.4f}")

    dt = time.time() - t0
    avg_loss = total_loss / n_batches
    avg_acc_s2w = total_acc_s2w / n_batches
    avg_acc_w2s = total_acc_w2s / n_batches

    return {
        "loss": avg_loss,
        "acc_s2w": avg_acc_s2w,
        "acc_w2s": avg_acc_w2s,
        "temperature": model.logit_scale.exp().item(),
        "time": dt,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_acc_s2w = 0.0
    total_acc_w2s = 0.0
    n_batches = len(dataloader)

    for batch in dataloader:
        seismic = batch["seismic"].to(device, non_blocking=True)
        well_logs = batch["well_logs"].to(device, non_blocking=True)

        loss, acc_s2w, acc_w2s, _ = model(seismic, well_logs)

        total_loss += loss.item()
        total_acc_s2w += acc_s2w.item()
        total_acc_w2s += acc_w2s.item()

    return {
        "loss": total_loss / n_batches,
        "acc_s2w": total_acc_s2w / n_batches,
        "acc_w2s": total_acc_w2s / n_batches,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Config
    cfg = CLIPConfig()
    cfg.dataset_path = args.dataset
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.save_dir = args.save_dir
    print(f"Config: batch={cfg.batch_size}, epochs={cfg.num_epochs}, lr={cfg.learning_rate}")

    # Model
    model = SeismicWellCLIP(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    print(f"  Seismic encoder: {sum(p.numel() for p in model.seismic_encoder.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"  Well encoder:    {sum(p.numel() for p in model.well_encoder.parameters() if p.requires_grad) / 1e6:.2f}M")

    # Data
    train_ds = SeismicWellDataset(
        hdf5_path=cfg.dataset_path,
        augment=True,
        normalize=True,
        seismic_size=cfg.seismic_img_size,
        well_len=cfg.well_seq_len,
    )
    val_ds = SeismicWellDataset(
        hdf5_path=cfg.dataset_path,
        augment=False,
        normalize=True,
        seismic_size=cfg.seismic_img_size,
        well_len=cfg.well_seq_len,
    )

    # Split: 80/20
    n_total = len(train_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    # Validation uses the non-augmenting dataset on val indices
    val_subset.dataset = val_ds

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_pairs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_pairs,
    )
    print(f"Data: {n_train} train / {n_val} val")

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.warmup_epochs])
    scaler = GradScaler() if args.mixed_precision else None

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n{'='*50}\nEpoch {epoch}/{cfg.num_epochs}\n{'='*50}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        scaler, epoch, args, device)
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        lr = scheduler.get_last_lr()[0]
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | "
              f"Acc s→w: {train_metrics['acc_s2w']:.4f} | "
              f"Acc w→s: {train_metrics['acc_w2s']:.4f} | "
              f"τ: {train_metrics['temperature']:.4f} | "
              f"Time: {train_metrics['time']:.1f}s")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
              f"Acc s→w: {val_metrics['acc_s2w']:.4f} | "
              f"Acc w→s: {val_metrics['acc_w2s']:.4f} | "
              f"LR: {lr:.2e}")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append((train_metrics["acc_s2w"] + train_metrics["acc_w2s"]) / 2)
        history["val_acc"].append((val_metrics["acc_s2w"] + val_metrics["acc_w2s"]) / 2)

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "history": history,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        # Save latest
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "history": history,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last_model.pt"))

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return model, history


if __name__ == "__main__":
    main()
