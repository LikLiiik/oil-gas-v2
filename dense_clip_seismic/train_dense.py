#!/usr/bin/env python3
"""
Train Dense Seismic-Well CLIP for pixel-level geological feature detection.

Stage 1: Pre-train with dense contrastive loss (no labels needed)
Stage 2: Fine-tune with task-specific labels (fault, horizon, facies, fracture)

The key insight: well logs provide "ground truth" petrophysical constraints
at the well intersection column. Contrastive learning propagates these
constraints laterally, enabling accurate dense prediction.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/data/yxjiang/datatest")

from dense_clip_seismic.config import DenseCLIPConfig
from dense_clip_seismic.models.dense_clip import DenseSeismicWellCLIP
from dense_clip_seismic.data.dataset import DenseSeismicWellDataset, collate_dense


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, scaler,
                    epoch: int, device, stage: str = "contrastive"):
    """Train one epoch. stage: 'contrastive' (Stage 1) or 'joint' (Stage 2)."""
    model.train()
    metrics = {
        "total_loss": 0.0, "cont_loss": 0.0,
        "acc_s2w": 0.0, "acc_w2s": 0.0,
    }
    n_batches = len(dataloader)

    for bidx, batch in enumerate(dataloader):
        seismic = batch["seismic"].to(device, non_blocking=True)
        well_logs = batch["well_logs"].to(device, non_blocking=True)
        well_x = batch["well_x"].to(device, non_blocking=True)
        labels = batch.get("labels", None)

        if labels is not None and stage == "joint":
            labels = {k: v.to(device) if v is not None else None
                      for k, v in labels.items()}

        optimizer.zero_grad()

        with autocast():
            outputs = model(seismic, well_logs, well_x,
                            task_labels=labels if stage == "joint" else None)

        if stage == "contrastive":
            loss = outputs["contrastive_loss"]
        else:
            loss = outputs["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        metrics["total_loss"] += loss.item()
        metrics["cont_loss"] += outputs["contrastive_loss"].item()
        metrics["acc_s2w"] += outputs["acc_s2w"].item()
        metrics["acc_w2s"] += outputs["acc_w2s"].item()

        if bidx % 20 == 0:
            print(f"  Epoch {epoch:3d} | [{bidx:3d}/{n_batches}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Cont: {outputs['contrastive_loss'].item():.4f} | "
                  f"Acc s→w: {outputs['acc_s2w'].item():.3f} | "
                  f"τ: {model.logit_scale.exp().item():.4f}")

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


@torch.no_grad()
def validate(model, dataloader, device, stage: str = "contrastive"):
    """Validation."""
    model.eval()
    metrics = {
        "total_loss": 0.0, "cont_loss": 0.0,
        "acc_s2w": 0.0, "acc_w2s": 0.0,
    }
    n_batches = len(dataloader)

    for batch in dataloader:
        seismic = batch["seismic"].to(device)
        well_logs = batch["well_logs"].to(device)
        well_x = batch["well_x"].to(device)
        labels = batch.get("labels", None)
        if labels is not None and stage == "joint":
            labels = {k: v.to(device) if v is not None else None
                      for k, v in labels.items()}

        outputs = model(seismic, well_logs, well_x,
                        task_labels=labels if stage == "joint" else None)
        loss = outputs["contrastive_loss"] if stage == "contrastive" \
               else outputs.get("total_loss", outputs["contrastive_loss"])

        metrics["total_loss"] += loss.item()
        metrics["cont_loss"] += outputs["contrastive_loss"].item()
        metrics["acc_s2w"] += outputs["acc_s2w"].item()
        metrics["acc_w2s"] += outputs["acc_w2s"].item()

    for k in metrics:
        metrics[k] /= n_batches
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="./openseisml_dataset_labeled.h5")
    parser.add_argument("--epochs-s1", type=int, default=60,
                        help="Stage 1 (contrastive) epochs")
    parser.add_argument("--epochs-s2", type=int, default=40,
                        help="Stage 2 (joint) epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="./dense_clip_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = DenseCLIPConfig()
    cfg.dataset_path = args.dataset
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.save_dir = args.save_dir

    print(f"Device: {device} | PyTorch: {torch.__version__}")
    print(f"Config: batch={cfg.batch_size}, epochs S1={args.epochs_s1}, "
          f"S2={args.epochs_s2}, lr={cfg.learning_rate}")

    # ── Model ─────────────────────────────────────────────────
    model = DenseSeismicWellCLIP(cfg).to(device)

    # Optionally load Stage 1 checkpoint
    s1_ckpt = os.path.join(args.save_dir, "best_s1_contrastive.pt")
    if os.path.exists(s1_ckpt):
        print(f"Loading Stage 1 checkpoint: {s1_ckpt}")
        ckpt = torch.load(s1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded epoch {ckpt['epoch']}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M params")
    print(f"  Seismic UNet: {sum(p.numel() for p in model.seismic_encoder.parameters()) / 1e6:.2f}M")
    print(f"  Well encoder: {sum(p.numel() for p in model.well_encoder.parameters()) / 1e6:.2f}M")
    print(f"  Task heads:   {sum(p.numel() for p in model.task_heads.parameters()) / 1e6:.2f}M")

    # ── Data ──────────────────────────────────────────────────
    has_labels = os.path.exists(args.dataset) and \
        ("labeled" in args.dataset or "geological" in args.dataset
         or "realistic" in args.dataset or "consistent" in args.dataset)
    dataset = DenseSeismicWellDataset(
        hdf5_path=args.dataset if os.path.exists(args.dataset)
                  else "/data/yxjiang/datatest/openseisml_dataset.h5",
        augment=True, normalize=True,
        has_labels=has_labels,
    )
    print(f"Dataset: {len(dataset)} pairs, labels={'yes' if has_labels else 'no'}")

    # Split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_dense,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_dense,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate,
                      weight_decay=cfg.weight_decay)
    scaler = GradScaler()
    # Stage 1: Contrastive Pre-training
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("STAGE 1: Dense Contrastive Pre-training")
    print(f"{'='*60}")

    warmup = LinearLR(optimizer, start_factor=0.1,
                      total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=args.epochs_s1 - cfg.warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.warmup_epochs])
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs_s1 + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, scaler,
                                  epoch, device, stage="contrastive")
        scheduler.step()
        val_m = validate(model, val_loader, device, stage="contrastive")

        print(f"  [S1 E{epoch:3d}] Train: loss={train_m['cont_loss']:.4f}, "
              f"acc_s2w={train_m['acc_s2w']:.4f} | "
              f"Val: loss={val_m['cont_loss']:.4f}, "
              f"acc_s2w={val_m['acc_s2w']:.4f}")

        if val_m["cont_loss"] < best_val_loss:
            best_val_loss = val_m["cont_loss"]
            torch.save({
                "epoch": epoch, "stage": "contrastive",
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, os.path.join(args.save_dir, "best_s1_contrastive.pt"))

    # ═══════════════════════════════════════════════════════════
    # Stage 2: Joint Training with Task Supervision
    # ═══════════════════════════════════════════════════════════
    if has_labels:
        print(f"\n{'='*60}")
        print("STAGE 2: Joint Contrastive + Task Supervision")
        print(f"{'='*60}")

        warmup2 = LinearLR(optimizer, start_factor=0.5,
                           total_iters=5)
        cosine2 = CosineAnnealingLR(optimizer,
                                    T_max=args.epochs_s2 - 5)
        scheduler2 = SequentialLR(optimizer, schedulers=[warmup2, cosine2],
                                  milestones=[5])
        best_task_loss = float("inf")

        for epoch in range(1, args.epochs_s2 + 1):
            train_m = train_one_epoch(model, train_loader, optimizer, scaler,
                                      epoch, device, stage="joint")
            scheduler2.step()
            val_m = validate(model, val_loader, device, stage="joint")

            print(f"  [S2 E{epoch:3d}] Train: loss={train_m['total_loss']:.4f}, "
                  f"cont={train_m['cont_loss']:.4f} | "
                  f"Val: loss={val_m['total_loss']:.4f}")

            if val_m["total_loss"] < best_task_loss:
                best_task_loss = val_m["total_loss"]
                torch.save({
                    "epoch": epoch, "stage": "joint",
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                }, os.path.join(args.save_dir, "best_s2_joint.pt"))

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, os.path.join(args.save_dir, "final_model.pt"))

    print(f"\nTraining complete!")
    print(f"  Best Stage 1 val loss: {best_val_loss:.4f}")
    if has_labels:
        print(f"  Best Stage 2 val loss: {best_task_loss:.4f}")


if __name__ == "__main__":
    main()
