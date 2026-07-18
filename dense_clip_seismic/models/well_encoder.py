"""
Well Log Encoder (1D, resolution-preserving)

Produces per-depth features at the same resolution as input,
enabling depth-point-level contrastive learning with seismic.

Uses dilated convolutions for large receptive field without
losing depth resolution (no pooling in depth dimension).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DilatedConvBlock(nn.Module):
    """Dilated 1D conv → BN → ReLU, with residual."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class WellLogEncoder1D(nn.Module):
    """
    Resolution-preserving 1D encoder for well log curves.

    Input:  (B, 6, 256) — 6 log types × 256 depth samples
    Output: (B, feature_dim, 256) — per-depth features

    Architecture:
        stem conv → dilated residual blocks → final projection

    No pooling/stride in depth dimension preserves the
    1:1 depth correspondence with the seismic vertical axis.
    """

    def __init__(self, in_channels: int = 6, base_dim: int = 64,
                 dilations: List[int] = None,
                 feature_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 1, 2, 4, 8]

        # Stem: project channels without changing length
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, kernel_size=7,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_dim),
            nn.ReLU(inplace=True),
        )

        # Dilated residual blocks (resolution-preserving)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(DilatedConvBlock(base_dim, dilation=d))

        # Final projection
        self.final = nn.Sequential(
            nn.Conv1d(base_dim, feature_dim * 2, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, L) well log curves
        Returns:
            features: (B, feature_dim, L) per-depth features
        """
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        x = self.dropout(x)
        return x  # (B, feature_dim, L)
