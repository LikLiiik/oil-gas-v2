"""
Seismic U-Net: Per-pixel feature extractor for seismic images.

Produces a feature map at the same spatial resolution as input,
enabling pixel-level contrastive learning with well logs and
downstream dense prediction (faults, horizons, facies, fractures).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU, with residual."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class EncoderStage(nn.Module):
    """n ConvBlocks, last one with optional stride."""

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, stride: int = 2):
        super().__init__()
        layers = [ConvBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(ConvBlock(out_ch, out_ch))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class DecoderBlock(nn.Module):
    """Upsample + skip connection + ConvBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x)


class SeismicUNet(nn.Module):
    """
    U-Net encoder-decoder for per-pixel seismic feature extraction.

    Input:  (B, 1, 256, 512)
    Output: (B, feature_dim, 256, 512) — per-pixel C-dim features
    """

    def __init__(self, in_channels: int = 1, base_dim: int = 32,
                 depths: List[int] = None, dims: List[int] = None,
                 feature_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if dims is None:
            dims = [32, 64, 128, 256]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoder_stages = nn.ModuleList()
        in_ch = base_dim
        self.enc_dims = []
        for d, dim in zip(depths, dims):
            stage = EncoderStage(in_ch, dim, d, stride=2)
            self.encoder_stages.append(stage)
            self.enc_dims.append(dim)
            in_ch = dim

        # Bottleneck
        bottleneck_dim = dims[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBlock(in_ch, bottleneck_dim),
            ConvBlock(bottleneck_dim, bottleneck_dim),
        )

        # Decoder
        dec_dims = list(reversed(dims))
        self.decoder_blocks = nn.ModuleList()
        in_ch = bottleneck_dim
        for i, dim in enumerate(dec_dims):
            skip_dim = dims[len(dims) - 1 - i]
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_dim, dim))
            in_ch = dim

        # Final projection to feature_dim
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, feature_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) seismic image
        Returns:
            features: (B, feature_dim, H', W') per-pixel features
                      (H', W') ≈ (H, W) up to even size alignment
        """
        input_size = x.shape[-2:]  # save for final alignment

        # Stem
        f0 = self.stem(x)  # (B, base, H/2, W/2)

        # Encoder: collect skip features at each level
        skips = []
        cur = f0
        for stage in self.encoder_stages:
            feats = stage(cur)
            cur = feats[-1]           # continue encoding
            skips.append(feats[-1])   # save for skip: dims=[32, 64, 128, 256]

        # Bottleneck takes the deepest encoder output
        x = self.bottleneck(cur)  # cur = (B, 256, H/32, W/32) → (B, 512, ...)

        # Decoder: reversed skips = [256, 128, 64, 32], matching decoder blocks
        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder(x, skip)

        # Final projection
        x = self.final(x)
        x = self.dropout(x)

        # Align to exact input size
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear',
                              align_corners=False)

        return x  # (B, feature_dim, H, W)
