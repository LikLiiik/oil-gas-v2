"""
Data pipeline for Seismic-Well CLIP training.

Loads HDF5 dataset, pairs seismic sections with well logs,
applies modality-specific augmentations.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


class SeismicAugmentation:
    """
    Seismic-specific augmentations — preserving geological validity.

    Allowed: horizontal flip (structures can be mirrored), noise,
             brightness/contrast, random crop+resize
    Forbidden: vertical flip (geological layering is directional)
    """

    def __init__(self, target_size: Tuple[int, int] = (256, 512),
                 crop_scale: Tuple[float, float] = (0.7, 1.0),
                 noise_std: float = 0.02,
                 hflip_prob: float = 0.5,
                 brightness_range: float = 0.1,
                 contrast_range: float = 0.1):
        self.target_h, self.target_w = target_size
        self.crop_scale = crop_scale
        self.noise_std = noise_std
        self.hflip_prob = hflip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: (H, W) or (1, H, W) float32 seismic image
        Returns:
            augmented image of shape (target_h, target_w)
        """
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # (1, H, W)

        C, H, W = img.shape

        # 1. Random crop + resize (simulates different acquisition geometries)
        scale = np.random.uniform(*self.crop_scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top = np.random.randint(0, H - crop_h + 1) if crop_h < H else 0
        left = np.random.randint(0, W - crop_w + 1) if crop_w < W else 0
        img = img[:, top:top + crop_h, left:left + crop_w]

        # Resize back to target
        img_tensor = torch.from_numpy(img).float()
        img_tensor = F.interpolate(img_tensor.unsqueeze(0),
                                   size=(self.target_h, self.target_w),
                                   mode='bilinear', align_corners=False)
        img = img_tensor.squeeze(0).numpy()

        # 2. Horizontal flip (geologically valid)
        if np.random.random() < self.hflip_prob:
            img = img[:, :, ::-1].copy()

        # 3. Brightness jitter
        if self.brightness_range > 0:
            delta = np.random.uniform(-self.brightness_range,
                                       self.brightness_range)
            img = img + delta

        # 4. Contrast jitter
        if self.contrast_range > 0:
            factor = np.random.uniform(1 - self.contrast_range,
                                        1 + self.contrast_range)
            mean_val = img.mean(axis=(1, 2), keepdims=True)
            img = (img - mean_val) * factor + mean_val

        # 5. Additive Gaussian noise
        if self.noise_std > 0:
            noise = np.random.randn(*img.shape) * self.noise_std * np.std(img)
            img = img + noise

        return img.astype(np.float32)


class WellLogAugmentation:
    """
    Well log augmentations — preserving petrophysical consistency.

    Allowed: random depth window, channel dropout, Gaussian noise,
             small random shifts (simulates depth calibration errors)
    Forbidden: arbitrary permutation (depth ordering is physical)
    """

    def __init__(self, target_len: int = 256,
                 mask_ratio: float = 0.1,
                 noise_std: float = 0.05,
                 shift_max: int = 5):
        self.target_len = target_len
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.shift_max = shift_max

    def __call__(self, logs: np.ndarray) -> np.ndarray:
        """
        Args:
            logs: (C, L) float32 well log curves
        Returns:
            augmented logs of shape (C, target_len)
        """
        C, L = logs.shape

        # 1. Random depth window extraction
        if L > self.target_len:
            start = np.random.randint(0, L - self.target_len + 1)
            logs = logs[:, start:start + self.target_len]
        elif L < self.target_len:
            # Pad if needed (shouldn't happen normally)
            pad_len = self.target_len - L
            logs = np.pad(logs, ((0, 0), (0, pad_len)), mode='edge')

        # 2. Small depth shift (simulates calibration offset)
        if self.shift_max > 0 and self.shift_max < self.target_len:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            if shift != 0:
                logs = np.roll(logs, shift, axis=1)
                if shift > 0:
                    logs[:, :shift] = logs[:, shift:shift + 1]  # fill left edge
                elif shift < 0:
                    logs[:, shift:] = logs[:, [shift]]  # fill right edge

        # 3. Channel dropout (randomly zero out one log type)
        if np.random.random() < 0.15:
            c = np.random.randint(0, C)
            logs[c, :] = 0

        # 4. Random mask (blocks of zeros, simulates missing data)
        if self.mask_ratio > 0:
            mask = np.random.random(self.target_len) < self.mask_ratio
            logs[:, mask] = 0

        # 5. Gaussian noise
        if self.noise_std > 0:
            std_per_channel = np.std(logs, axis=1, keepdims=True)
            noise = np.random.randn(C, self.target_len) * self.noise_std * std_per_channel
            logs = logs + noise

        return logs.astype(np.float32)


class SeismicWellDataset(Dataset):
    """
    PyTorch Dataset for paired seismic sections and well logs.

    Each sample: (seismic_image, well_logs) pair.
    The pairing is natural — each seismic section passes through one well.

    When augment=True, each epoch sees different augmented views.
    """

    LOG_TYPES = ["GR", "NPHI", "RHOB", "DT", "RT", "velocity"]

    def __init__(self, hdf5_path: str,
                 augment: bool = True,
                 normalize: bool = True,
                 seismic_size: Tuple[int, int] = (256, 512),
                 well_len: int = 256):
        """
        Args:
            hdf5_path:  path to OpenSeisML HDF5 dataset
            augment:    apply data augmentations (disable for eval)
            normalize:  apply per-sample normalization
        """
        self.hdf5_path = hdf5_path
        self.augment = augment
        self.normalize = normalize
        self.seismic_size = seismic_size
        self.well_len = well_len

        if augment:
            self.seismic_aug = SeismicAugmentation(target_size=seismic_size)
            self.well_aug = WellLogAugmentation(target_len=well_len)

        # Index dataset entries
        with h5py.File(hdf5_path, 'r') as f:
            self.n_sections = len(f["seismic"])
            self.well_ids = sorted(f["wells"].keys())

    def __len__(self) -> int:
        return self.n_sections

    def _load_seismic(self, idx: int, f: h5py.File) -> np.ndarray:
        """Load a single seismic section."""
        key = f"seismic/section_{idx:04d}"
        img = f[key][:]  # (256, 512) float32
        return img

    def _load_well(self, idx: int, f: h5py.File) -> np.ndarray:
        """Load well logs for the well matching this section."""
        well_id = f"WELL_{(idx + 1):04d}"
        if well_id not in f["wells"]:
            # Fallback: use well associated with this section
            key = f"seismic/section_{idx:04d}"
            well_id = f[key].attrs.get("well_id", f"WELL_{(idx + 1):04d}")

        logs = []
        for lt in self.LOG_TYPES:
            if lt in f[f"wells/{well_id}"]:
                data = f[f"wells/{well_id}/{lt}"][:].astype(np.float32)
            else:
                data = np.zeros(self.well_len, dtype=np.float32)
            logs.append(data)

        logs = np.stack(logs, axis=0)  # (6, L)
        return logs

    def _normalize_seismic(self, img: np.ndarray) -> np.ndarray:
        """Per-sample standardization."""
        mean, std = img.mean(), img.std()
        if std < 1e-8:
            std = 1.0
        return (img - mean) / std

    def _normalize_well(self, logs: np.ndarray) -> np.ndarray:
        """Per-channel standardization."""
        mean = logs.mean(axis=1, keepdims=True)
        std = logs.std(axis=1, keepdims=True)
        std = np.clip(std, 1e-8, None)
        return (logs - mean) / std

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import sys
        # Re-open file each call (HDF5 is not thread-safe for multiple workers)
        with h5py.File(self.hdf5_path, 'r') as f:
            seismic = self._load_seismic(idx, f)
            well_logs = self._load_well(idx, f)

        # Augment
        if self.augment:
            seismic = self.seismic_aug(seismic)
            well_logs = self.well_aug(well_logs)
        else:
            if seismic.ndim == 2:
                seismic = seismic[np.newaxis, :, :]

        # Normalize
        if self.normalize:
            seismic = self._normalize_seismic(seismic)
            well_logs = self._normalize_well(well_logs)

        # Ensure correct shapes
        seismic = np.ascontiguousarray(seismic).squeeze(0) if seismic.shape[0] == 1 else seismic
        seismic = np.expand_dims(seismic, 0)  # (1, H, W)

        return {
            "seismic": torch.from_numpy(seismic).float(),
            "well_logs": torch.from_numpy(well_logs).float(),
            "index": idx,
        }


class SeismicWellDataModule:
    """Lightning-style data module for CLIP training."""

    def __init__(self, config, augment_train: bool = True):
        self.config = config
        self.augment_train = augment_train

    def train_dataloader(self) -> DataLoader:
        dataset = SeismicWellDataset(
            hdf5_path=self.config.dataset_path,
            augment=self.augment_train,
            normalize=True,
            seismic_size=self.config.seismic_img_size,
            well_len=self.config.well_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = SeismicWellDataset(
            hdf5_path=self.config.dataset_path,
            augment=False,
            normalize=True,
            seismic_size=self.config.seismic_img_size,
            well_len=self.config.well_seq_len,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def collate_pairs(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for paired data."""
    seismic = torch.stack([b["seismic"] for b in batch], dim=0)
    well_logs = torch.stack([b["well_logs"] for b in batch], dim=0)
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return {"seismic": seismic, "well_logs": well_logs, "index": indices}
