"""
Dense Seismic-Well CLIP Model
==============================
Pixel-level cross-modal contrastive learning framework.

Core idea:
  - Seismic U-Net outputs per-pixel features F_seis[H, W, C]
  - Well encoder outputs per-depth features F_well[L, C]
  - At well intersection point (wx, z): F_seis[z, wx, :] ↔ F_well[z, :]
  - Dense InfoNCE loss aligns seismic and well features at each depth

This enables:
  1. Well logs to supervise seismic features at the intersection column
  2. The aligned feature space to propagate petrophysical constraints
     laterally across the entire seismic section
  3. Downstream dense prediction with cross-modal feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List

from .seismic_unet import SeismicUNet
from .well_encoder import WellLogEncoder1D
from .task_heads import MultiTaskHeads


class ProjectionHead(nn.Module):
    """Project features to contrastive embedding space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead1D(nn.Module):
    """Project 1D features to contrastive embedding space."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseSeismicWellCLIP(nn.Module):
    """
    Dense contrastive learning for seismic images and well logs.

    Args:
        config: DenseCLIPConfig instance

    Forward:
        seismic: (B, 1, 256, 512)
        well_logs: (B, 6, 256)
        well_x: (B,) — lateral position of well in each seismic section

    Returns:
        loss_dict:   contrastive loss + task losses
        predictions: per-pixel geological feature predictions
        features:    aligned seismic/well features (for analysis)
    """

    def __init__(self, config):
        super().__init__()
        cfg = config

        # ── Encoders ──────────────────────────────────────────
        self.seismic_encoder = SeismicUNet(
            in_channels=cfg.seismic_channels,
            base_dim=cfg.s_base_dim,
            depths=cfg.s_depths,
            dims=cfg.s_dims,
            feature_dim=cfg.feature_dim,
        )

        self.well_encoder = WellLogEncoder1D(
            in_channels=cfg.well_channels,
            base_dim=cfg.w_base_dim,
            dilations=cfg.w_dilations,
            feature_dim=cfg.feature_dim,
        )

        # ── Projection to contrastive space ────────────────────
        self.s_proj = ProjectionHead(cfg.feature_dim,
                                     cfg.feature_dim, cfg.proj_dim)
        self.w_proj = ProjectionHead1D(cfg.feature_dim,
                                       cfg.feature_dim, cfg.proj_dim)

        # ── Learnable temperature ─────────────────────────────
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / cfg.temperature)
        )

        # ── Task heads ────────────────────────────────────────
        self.task_heads = MultiTaskHeads(
            in_channels=cfg.feature_dim,
            n_litho=3,
        )

    def encode_seismic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) seismic image
        Returns:
            features:    (B, C, H, W) per-pixel features
            proj_feats:  (B, D, H, W) projected features (for contrastive)
        """
        features = self.seismic_encoder(x)
        proj_feats = self.s_proj(features)
        return features, proj_feats

    def encode_well(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 6, L) well log curves
        Returns:
            features:    (B, C, L) per-depth features
            proj_feats:  (B, D, L) projected features (for contrastive)
        """
        features = self.well_encoder(x)
        proj_feats = self.w_proj(features)
        return features, proj_feats

    def dense_contrastive_loss(self, s_feats_proj: torch.Tensor,
                                w_feats_proj: torch.Tensor,
                                well_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Dense InfoNCE at well intersection points.

        At each depth z, the seismic feature at the well position (z, well_x)
        should match the well log feature at that same depth.

        Args:
            s_feats_proj: (B, D, H, W) projected seismic features
            w_feats_proj: (B, D, L) projected well features
            well_x:       (B,) lateral positions of wells [0, W-1]

        Returns:
            losses: {contrastive_loss, acc_s2w, acc_w2s}
        """
        B, D, H, W = s_feats_proj.shape
        _, _, L = w_feats_proj.shape

        # Extract seismic features along the well column
        well_x_clamped = well_x.clamp(0, W - 1).long()
        s_at_well = []
        for b in range(B):
            s_at_well.append(s_feats_proj[b, :, :, well_x_clamped[b]])
        s_at_well = torch.stack(s_at_well, dim=0)  # (B, D, H)

        # Ensure H == L (both should be 256)
        assert H == L, f"Depth mismatch: seismic H={H}, well L={L}"

        # Permute to (B, N, D) where N = H = L
        s_vecs = s_at_well.permute(0, 2, 1)  # (B, N, D)
        w_vecs = w_feats_proj.permute(0, 2, 1)  # (B, N, D)

        # L2 normalize
        s_vecs = F.normalize(s_vecs, dim=-1)
        w_vecs = F.normalize(w_vecs, dim=-1)

        # Compute dense similarity matrix
        # For each depth point, compute similarity across all depths
        logit_scale = self.logit_scale.exp()

        total_loss = 0.0
        total_acc_s2w = 0.0
        total_acc_w2s = 0.0
        n_valid = 0

        for b in range(B):
            # (N, D) @ (N, D)^T = (N, N)
            sim = logit_scale * (s_vecs[b] @ w_vecs[b].T)

            labels = torch.arange(L, device=sim.device)

            loss_s2w = F.cross_entropy(sim, labels)
            loss_w2s = F.cross_entropy(sim.T, labels)
            total_loss += (loss_s2w + loss_w2s) / 2.0

            acc_s2w = (sim.argmax(dim=-1) == labels).float().mean()
            acc_w2s = (sim.T.argmax(dim=-1) == labels).float().mean()
            total_acc_s2w += acc_s2w
            total_acc_w2s += acc_w2s
            n_valid += 1

        return {
            "contrastive_loss": total_loss / n_valid,
            "acc_s2w": total_acc_s2w / n_valid,
            "acc_w2s": total_acc_w2s / n_valid,
        }

    def fuse_features(self, s_feats: torch.Tensor,
                      w_feats: torch.Tensor,
                      well_x: torch.Tensor) -> torch.Tensor:
        """
        Fuse well log features into seismic features at the well column.
        This injects petrophysical constraints into the seismic feature map.

        Args:
            s_feats: (B, C, H, W) seismic features
            w_feats: (B, C, L) well features
            well_x:  (B,) well lateral positions
        Returns:
            fused: (B, C, H, W) feature map with well info injected
        """
        B, C, H, W = s_feats.shape
        well_x_clamped = well_x.clamp(0, W - 1).long()

        # Start with seismic features
        fused = s_feats.clone()

        # At the well column, replace (or add) well features
        # Use a learned combination: s_feats + alpha * interpolated_w_feats
        for b in range(B):
            wx = well_x_clamped[b]
            # w_feats[b] is (C, L), interpolate to (C, H) if needed
            w_interp = F.interpolate(
                w_feats[b].unsqueeze(0), size=H, mode='linear',
                align_corners=False
            ).squeeze(0)  # (C, H)

            # Blend well features into the seismic column
            # Using a small window around the well for smooth blending
            half_window = 3
            for dx in range(-half_window, half_window + 1):
                x_pos = (wx + dx).clamp(0, W - 1)
                weight = 0.5 * (1.0 - abs(dx) / (half_window + 1))
                fused[b, :, :, x_pos] = (fused[b, :, :, x_pos] +
                                         weight * w_interp)

        return fused

    def forward(self, seismic: torch.Tensor,
                well_logs: torch.Tensor,
                well_x: torch.Tensor,
                task_labels: Optional[Dict] = None) -> Dict:
        """
        Full forward pass.

        Args:
            seismic:    (B, 1, H, W)
            well_logs:  (B, 6, L)
            well_x:     (B,) lateral position of each well
            task_labels: optional dict of ground truth labels
        Returns:
            outputs: dict with losses, predictions, features
        """
        # Encode both modalities
        s_feats, s_feats_proj = self.encode_seismic(seismic)
        w_feats, w_feats_proj = self.encode_well(well_logs)

        # Dense contrastive loss
        cont_losses = self.dense_contrastive_loss(
            s_feats_proj, w_feats_proj, well_x
        )

        # Task predictions on aligned seismic features
        # (contrastive loss implicitly injects well info via gradient)
        preds_seismic = self.task_heads(s_feats)
        # Fused variant for comparison: use same features (will evaluate separately)
        fused = self.fuse_features(s_feats, w_feats, well_x)
        preds_fused = self.task_heads(fused)

        outputs = {
            "contrastive_loss": cont_losses["contrastive_loss"],
            "acc_s2w": cont_losses["acc_s2w"],
            "acc_w2s": cont_losses["acc_w2s"],
            "preds_seismic": preds_seismic,
            "preds_fused": preds_fused,
            "s_feats": s_feats,
            "w_feats": w_feats,
            "fused_feats": fused,
        }

        # Task losses (if labels provided)
        if task_labels is not None:
            total_task_loss = 0.0
            TASK_KEYS = ["velocity", "porosity", "lithology", "density", "resistivity"]
            REGRESSION_TASKS = {"velocity", "porosity", "density", "resistivity"}
            CLASSIFICATION_TASKS = {"lithology"}

            # Per-task normalization constants (from data statistics)
            TASK_STATS = {
                "velocity":    (2600.0, 600.0),    # (mean, std)
                "porosity":    (0.22, 0.07),
                "density":     (2.35, 0.15),
                "resistivity": (5.0, 5.0),
            }

            for task in TASK_KEYS:
                if task in task_labels and task_labels[task] is not None:
                    # pred: (B, C, H, W) or (B, 1, H, W)
                    # label: (B, L) — 1D at well column
                    pred_col = []
                    for b in range(preds_seismic[task].shape[0]):
                        wx = well_x[b].clamp(0, preds_seismic[task].shape[3] - 1)
                        pred_col.append(preds_seismic[task][b, :, :, wx])
                    pred_col = torch.stack(pred_col, dim=0)  # (B, C, L)

                    gt = task_labels[task]  # (B, L)

                    if task in REGRESSION_TASKS:
                        # pred is z-scored (tanh*3), gt needs z-scoring
                        mean, std = TASK_STATS[task]
                        gt_norm = (gt - mean) / std
                        l = F.mse_loss(pred_col.squeeze(1), gt_norm)
                    elif task in CLASSIFICATION_TASKS:
                        l = F.cross_entropy(pred_col, gt.long())
                    else:
                        continue

                    total_task_loss = total_task_loss + l * 0.2  # weight per task
                    outputs[f"{task}_loss"] = l
            outputs["task_loss"] = total_task_loss

            # Total loss
            outputs["total_loss"] = (
                outputs["contrastive_loss"] + outputs["task_loss"]
            )

        return outputs
