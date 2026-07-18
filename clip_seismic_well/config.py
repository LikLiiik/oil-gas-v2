"""Configuration for Seismic-Well CLIP training."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CLIPConfig:
    # ── Model architecture ───────────────────────────────────────
    # Seismic encoder (ViT-Small)
    seismic_img_size: Tuple[int, int] = (256, 512)  # (H, W)
    seismic_patch_size: int = 16
    seismic_in_channels: int = 1
    seismic_embed_dim: int = 384
    seismic_depth: int = 8
    seismic_num_heads: int = 6
    seismic_mlp_ratio: float = 4.0

    # Well log encoder (1D ConvNeXt-style)
    well_in_channels: int = 6  # GR, NPHI, RHOB, DT, RT, velocity
    well_seq_len: int = 256
    well_base_dim: int = 64
    well_depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    well_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Shared projection
    projection_dim: int = 256
    embed_dim: int = 512  # final joint embedding dimension

    # ── Training ─────────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    temperature: float = 0.07  # initial τ for InfoNCE

    # ── Data ─────────────────────────────────────────────────────
    dataset_path: str = "./openseisml_dataset.h5"
    num_workers: int = 4
    seismic_aug_scale: Tuple[float, float] = (0.7, 1.0)
    seismic_aug_noise_std: float = 0.02
    well_aug_mask_ratio: float = 0.1
    well_aug_noise_std: float = 0.05

    # ── Hardware ─────────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: bool = True
    num_gpus: int = 1

    # ── Logging ──────────────────────────────────────────────────
    log_interval: int = 10
    save_dir: str = "./clip_checkpoints"
