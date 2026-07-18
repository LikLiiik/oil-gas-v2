"""
Well Log Encoder: 1D ConvNeXt-style backbone
For multi-channel petrophysical log curves.

Input: 6 channels (GR, NPHI, RHOB, DT, RT, velocity) × 256 depth samples.
The encoder captures vertical variations in rock properties through
hierarchical 1D convolutions with depth-wise separable convs.
"""

import torch
import torch.nn as nn
import math


class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt-style 1D block:
    - Depthwise conv (large kernel) → LayerNorm → 1×1 conv → GELU → 1×1 conv
    - With residual connection and layer scale.
    """

    def __init__(self, dim: int, kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)  # layer scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (B, L, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # back to (B, C, L)
        x = x * self.gamma.view(1, -1, 1)
        return x + residual


class Downsample1D(nn.Module):
    """Downsample with LayerNorm + Conv1d stride-2."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.conv(x)


class WellLogEncoder(nn.Module):
    """
    Hierarchical 1D ConvNeXt encoder for well log curves.

    Input:  (B, 6, 256)  — 6 log types × 256 depth samples
    Output: (B, embed_dim) — feature vector for contrastive learning

    Architecture:
        stem → [ConvNeXt1DBlock × d₁] → downsample →
               [ConvNeXt1DBlock × d₂] → downsample →
               [ConvNeXt1DBlock × d₃] → downsample →
               [ConvNeXt1DBlock × d₄] → GAP → projection
    """

    def __init__(self, in_channels: int = 6, seq_len: int = 256,
                 base_dim: int = 64,
                 depths: list = None,
                 dims: list = None,
                 kernel_size: int = 7,
                 embed_dim: int = 384,
                 dropout: float = 0.1):
        super().__init__()
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [64, 128, 256, 512]

        # Stem convolution
        self.stem = nn.Conv1d(in_channels, base_dim, kernel_size=4,
                              stride=4, padding=0)

        # Compute sequence lengths at each stage
        cur_len = seq_len // 4

        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur_dim = base_dim

        for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
            # ConvNeXt blocks
            blocks = nn.ModuleList([
                ConvNeXt1DBlock(cur_dim, kernel_size, dropout)
                for _ in range(depth)
            ])
            self.stages.append(blocks)

            # Downsample (except last stage)
            if stage_idx < len(depths) - 1:
                self.downsamples.append(Downsample1D(cur_dim, dim))
                cur_dim = dim
                cur_len //= 2
            else:
                # Last stage: project to target dim
                self.downsamples.append(nn.Identity())

        self.final_len = cur_len
        self.gap_dim = cur_dim
        self.norm = nn.LayerNorm(cur_dim)

        # Final projection to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(cur_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 256)
        x = self.stem(x)  # (B, base_dim, 64)

        for stage_blocks, downsample in zip(self.stages, self.downsamples):
            for block in stage_blocks:
                x = block(x)
            x = downsample(x)

        # Global average pooling over length dimension
        x = x.mean(dim=-1)  # (B, dim)
        x = self.norm(x)
        x = self.proj(x)
        return x  # (B, embed_dim)

    def get_depth_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-pooling features for depth-wise analysis."""
        x = self.stem(x)
        for stage_blocks, downsample in zip(self.stages, self.downsamples):
            for block in stage_blocks:
                x = block(x)
            x = downsample(x)
        return x  # (B, final_dim, final_len)
