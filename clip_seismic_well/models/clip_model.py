"""
Seismic-Well CLIP Model
=======================
Joint multi-modal contrastive learning framework for seismic images
and well log curves. Inspired by CLIP (Radford et al., 2021).

Architecture:
  f_seismic: ViT encoder  → projection → L2-normalized embedding
  f_well:    1D ConvNeXt  → projection → L2-normalized embedding

Loss: Symmetric InfoNCE with learnable temperature τ.

  L = ½·(L_{s→w} + L_{w→s})

  L_{s→w} = -1/B Σᵢ log[ exp(sim(sᵢ, wᵢ)/τ) / Σⱼ exp(sim(sᵢ, wⱼ)/τ) ]

where sim(a, b) = a^T b  (cosine similarity, since a,b are L2-normed).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

from .seismic_encoder import SeismicEncoder
from .well_encoder import WellLogEncoder


class ProjectionHead(nn.Module):
    """Non-linear projection head with BatchNorm (CLIP-style)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeismicWellCLIP(nn.Module):
    """
    CLIP-style dual-encoder for seismic images and well logs.

    Usage:
        model = SeismicWellCLIP(cfg)
        loss, acc_s2w, acc_w2s = model(seismic_imgs, well_logs)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ── Encoders ───────────────────────────────────────────
        self.seismic_encoder = SeismicEncoder(
            img_size=config.seismic_img_size,
            patch_size=config.seismic_patch_size,
            in_channels=config.seismic_in_channels,
            embed_dim=config.seismic_embed_dim,
            depth=config.seismic_depth,
            num_heads=config.seismic_num_heads,
            mlp_ratio=config.seismic_mlp_ratio,
        )

        self.well_encoder = WellLogEncoder(
            in_channels=config.well_in_channels,
            seq_len=config.well_seq_len,
            base_dim=config.well_base_dim,
            depths=config.well_depths,
            dims=config.well_dims,
            embed_dim=config.seismic_embed_dim,
        )

        # ── Projection heads ──────────────────────────────────
        self.seismic_proj = ProjectionHead(
            config.seismic_embed_dim,
            config.projection_dim * 2,
            config.embed_dim,
        )
        self.well_proj = ProjectionHead(
            config.seismic_embed_dim,
            config.projection_dim * 2,
            config.embed_dim,
        )

        # ── Learnable temperature ─────────────────────────────
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / config.temperature)
        )

        # ── Optional: modality-specific decoders (for reconstruction) ──
        self._init_weights()

    def _init_weights(self):
        for m in [self.seismic_proj, self.well_proj]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def encode_seismic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode seismic image → normalized embedding."""
        features = self.seismic_encoder(x)
        embedding = self.seismic_proj(features)
        return F.normalize(embedding, dim=-1)

    def encode_well(self, x: torch.Tensor) -> torch.Tensor:
        """Encode well logs → normalized embedding."""
        features = self.well_encoder(x)
        embedding = self.well_proj(features)
        return F.normalize(embedding, dim=-1)

    def forward(self, seismic: torch.Tensor,
                well_logs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with contrastive loss.

        Args:
            seismic:  (B, 1, 256, 512) seismic images
            well_logs: (B, 6, 256) well log curves

        Returns:
            loss:       scalar InfoNCE loss
            acc_s2w:    seismic→well retrieval accuracy (top-1)
            acc_w2s:    well→seismic retrieval accuracy (top-1)
            logits:     (B, B) similarity matrix (for analysis)
        """
        # Encode both modalities
        s_emb = self.encode_seismic(seismic)   # (B, D)
        w_emb = self.encode_well(well_logs)      # (B, D)

        # Cosine similarity with temperature scaling
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * s_emb @ w_emb.T  # (B, B)

        # Symmetric InfoNCE
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_s2w = F.cross_entropy(logits, labels)
        loss_w2s = F.cross_entropy(logits.T, labels)
        loss = (loss_s2w + loss_w2s) / 2.0

        # Accuracy
        acc_s2w = (logits.argmax(dim=-1) == labels).float().mean()
        acc_w2s = (logits.T.argmax(dim=-1) == labels).float().mean()

        return loss, acc_s2w, acc_w2s, logits

    @torch.no_grad()
    def compute_similarity_matrix(self, seismic: torch.Tensor,
                                   well_logs: torch.Tensor) -> np.ndarray:
        """Compute full similarity matrix between seismic and well batches."""
        s_emb = self.encode_seismic(seismic)
        w_emb = self.encode_well(well_logs)
        sim = (s_emb @ w_emb.T).cpu().numpy()
        return sim

    @torch.no_grad()
    def retrieve_well_for_seismic(self, seismic: torch.Tensor,
                                   well_logs: torch.Tensor,
                                   top_k: int = 5) -> Tuple:
        """For each seismic, retrieve top-k matching wells."""
        s_emb = self.encode_seismic(seismic)
        w_emb = self.encode_well(well_logs)
        logits = s_emb @ w_emb.T
        scores, indices = logits.topk(top_k, dim=-1)
        return indices.cpu().numpy(), scores.cpu().numpy()

    @torch.no_grad()
    def retrieve_seismic_for_well(self, well_logs: torch.Tensor,
                                   seismic: torch.Tensor,
                                   top_k: int = 5) -> Tuple:
        """For each well, retrieve top-k matching seismic sections."""
        w_emb = self.encode_well(well_logs)
        s_emb = self.encode_seismic(seismic)
        logits = w_emb @ s_emb.T
        scores, indices = logits.topk(top_k, dim=-1)
        return indices.cpu().numpy(), scores.cpu().numpy()
