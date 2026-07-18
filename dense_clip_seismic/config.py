"""Configuration for Dense Seismic-Well CLIP."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DenseCLIPConfig:
    # ── Input dimensions ──────────────────────────────────────
    seismic_shape: Tuple[int, int] = (256, 512)  # (H, W)
    seismic_channels: int = 1
    well_channels: int = 6  # GR, NPHI, RHOB, DT, RT, velocity
    well_length: int = 256

    # ── Seismic encoder (U-Net style) ─────────────────────────
    s_base_dim: int = 32
    s_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    s_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])

    # ── Well encoder (1D ConvNet, resolution-preserving) ──────
    w_base_dim: int = 64
    w_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 1, 2, 4, 8])

    # ── Shared feature dimension ──────────────────────────────
    feature_dim: int = 128
    proj_dim: int = 64  # dimension for contrastive loss

    # ── Contrastive learning ─────────────────────────────────
    temperature: float = 0.07
    n_negative_samples: int = 1024  # queue size for MoCo-style negative mining

    # ── Task heads ────────────────────────────────────────────
    n_facies_classes: int = 5  # shale, silt, sand, carbonate, basement

    # ── Training ──────────────────────────────────────────────
    batch_size: int = 8      # smaller batch due to per-pixel features
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    contrastive_weight: float = 1.0
    task_weight: float = 0.5

    # ── Data ──────────────────────────────────────────────────
    dataset_path: str = "./openseisml_dataset_labeled.h5"
    num_workers: int = 2

    # ── Hardware ──────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: bool = True

    # ── Logging ────────────────────────────────────────────────
    log_interval: int = 10
    save_dir: str = "./dense_clip_checkpoints"
