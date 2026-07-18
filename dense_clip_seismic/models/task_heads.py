"""
Geological segmentation heads for per-pixel prediction.

Four tasks, all operating on U-Net feature maps (128, 256, 512):

  Fault:    binary segmentation (Sigmoid) → Dice + BCE loss
  Horizon:  multi-class segmentation (N+1 classes, Softmax) → CE loss
  Facies:   multi-class segmentation (5 classes, Softmax) → CE loss
  Fracture: binary segmentation (Sigmoid) → Dice + BCE loss

Physical grounding (see plan references):
  - Faults are explicit displacement surfaces in the velocity model
  - Horizons are stratal surfaces (impedance contrast interfaces)
  - Facies are Vsh-based lithology classes (not velocity ranges alone)
  - Fractures cluster around fault damage zones + structural curvature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvHead(nn.Module):
    """Small 3-layer conv decoder for per-pixel predictions."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch // 2, 3, padding=1),
            nn.BatchNorm2d(mid_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 2, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation (imbalanced classes)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) logits
            target: (B, H, W) or (B, 1, H, W) binary
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target.float()).sum(dim=(1, 2, 3))
        union = pred_sig.sum(dim=(1, 2, 3)) + target.float().sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class GeologicalTaskHeads(nn.Module):
    """
    Segmentation heads for geological feature detection.

    All heads share a common trunk, then branch into task-specific layers.
    Produces per-pixel predictions at full (256, 512) resolution.
    """

    def __init__(self, in_channels: int = 128,
                 n_horizon_classes: int = 13,
                 n_facies_classes: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.n_horizon_classes = n_horizon_classes
        self.n_facies_classes = n_facies_classes

        # Shared feature refinement
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Task-specific heads
        self.fault_head = ConvHead(128, 1)       # → sigmoid for binary
        self.horizon_head = ConvHead(128, n_horizon_classes)
        self.facies_head = ConvHead(128, n_facies_classes)
        self.fracture_head = ConvHead(128, 1)     # → sigmoid for binary

        self.dice_loss = DiceLoss(smooth=1.0)

    def forward(self, x: torch.Tensor) -> dict:
        """Return raw logits for all four tasks."""
        shared = self.shared(x)
        return {
            "fault": self.fault_head(shared),
            "horizon": self.horizon_head(shared),
            "facies": self.facies_head(shared),
            "fracture": self.fracture_head(shared),
        }

    def compute_losses(self, predictions: dict,
                        labels: dict) -> dict:
        """
        Compute task-specific losses on full 2D predictions.

        Args:
            predictions: dict with keys 'fault', 'horizon', 'facies', 'fracture'
                         each a (B, C, H, W) tensor of logits
            labels:      dict with same keys, each (B, H, W) ground truth
        Returns:
            losses: dict with per-task loss values
        """
        losses = {}
        total = 0.0

        # Fault: Dice + BCE
        if "fault" in predictions and "fault" in labels:
            pred = predictions["fault"]  # (B, 1, H, W)
            gt = labels["fault"].float()  # (B, H, W)
            l_dice = self.dice_loss(pred, gt)
            l_bce = F.binary_cross_entropy_with_logits(
                pred.squeeze(1), gt
            )
            losses["fault"] = l_dice + l_bce
            total += losses["fault"]

        # Fracture: Dice + BCE
        if "fracture" in predictions and "fracture" in labels:
            pred = predictions["fracture"]
            gt = labels["fracture"].float()
            l_dice = self.dice_loss(pred, gt)
            l_bce = F.binary_cross_entropy_with_logits(
                pred.squeeze(1), gt
            )
            losses["fracture"] = l_dice + l_bce
            total += losses["fracture"]

        # Horizon: Cross-Entropy (multi-class)
        if "horizon" in predictions and "horizon" in labels:
            losses["horizon"] = F.cross_entropy(
                predictions["horizon"], labels["horizon"].long()
            )
            total += losses["horizon"]

        # Facies: Cross-Entropy (multi-class)
        if "facies" in predictions and "facies" in labels:
            losses["facies"] = F.cross_entropy(
                predictions["facies"], labels["facies"].long()
            )
            total += losses["facies"]

        losses["total"] = total
        return losses

    def predict(self, x: torch.Tensor) -> dict:
        """Return probability/prediction maps (for inference)."""
        logits = self(x)
        return {
            "fault": torch.sigmoid(logits["fault"]),
            "horizon": logits["horizon"].argmax(1),
            "facies": logits["facies"].argmax(1),
            "fracture": torch.sigmoid(logits["fracture"]),
            "fault_prob": torch.sigmoid(logits["fault"]),
            "fracture_prob": torch.sigmoid(logits["fracture"]),
        }
