"""
Seismic Encoder: Vision Transformer (ViT-Small)
Adapted for single-channel seismic images (256×512).

Seismic sections contain geological structures (faults, horizons,
channels) that span large spatial extents — ViT's self-attention
captures these long-range dependencies better than CNNs.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class PatchEmbed(nn.Module):
    """Split seismic image into patches and embed."""

    def __init__(self, img_size: Tuple[int, int] = (256, 512),
                 patch_size: int = 16, in_channels: int = 1,
                 embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_h = img_size[0] // patch_size
        self.n_patches_w = img_size[1] // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # -> (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SeismicEncoder(nn.Module):
    """
    ViT-Small encoder for seismic images.

    Input:  (B, 1, 256, 512)
    Output: (B, embed_dim) — feature vector for contrastive learning
    """

    def __init__(self, img_size: Tuple[int, int] = (256, 512),
                 patch_size: int = 16, in_channels: int = 1,
                 embed_dim: int = 384, depth: int = 8,
                 num_heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add position encoding and CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed_with_cls()
        x = self.dropout(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return CLS token as image representation
        return x[:, 0, :]  # (B, embed_dim)

    def pos_embed_with_cls(self) -> torch.Tensor:
        """Position embedding with extra slot for CLS token."""
        cls_pe = torch.zeros(1, 1, self.pos_embed.shape[-1],
                             device=self.pos_embed.device)
        return torch.cat([cls_pe, self.pos_embed], dim=1)

    def get_intermediate_features(self, x: torch.Tensor,
                                  layers: list = None) -> dict:
        """Extract features from intermediate layers (for analysis)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed_with_cls()
        x = self.dropout(x)

        features = {"patch_embed": x[:, 1:, :].clone()}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if layers is None or i in layers:
                features[f"block_{i}"] = x.clone()

        x = self.norm(x)
        features["final"] = x.clone()
        return features
