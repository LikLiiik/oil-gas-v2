"""
PyTorch Dataset for dense seismic-well CLIP training.

Petrophysical property prediction version:
  - Labels are 1D (well column only) → model propagates laterally
  - Tasks: velocity, porosity, lithology, density, resistivity
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from typing import Dict, Tuple, Optional, List


class SeismicAugmentation:
    """Seismic-specific augmentations for dense prediction."""

    def __init__(self, hflip_prob: float = 0.5, noise_std: float = 0.02):
        self.hflip_prob = hflip_prob
        self.noise_std = noise_std

    def __call__(self, img: np.ndarray, well_x: int) -> Tuple[np.ndarray, int]:
        if img.ndim == 2:
            img = img[np.newaxis, :, :]

        _, H, W = img.shape

        # Horizontal flip
        if np.random.random() < self.hflip_prob:
            img = img[:, :, ::-1].copy()
            well_x = W - 1 - well_x

        # Additive noise
        if self.noise_std > 0:
            noise = np.random.randn(*img.shape) * self.noise_std * np.std(img)
            img = img + noise

        return img.astype(np.float32), well_x


class DenseSeismicWellDataset(Dataset):
    """
    Dataset for dense contrastive learning + petrophysical prediction.

    Each item:
      - seismic:     (1, 256, 512)
      - well_logs:   (6, 256)
      - well_x:      lateral position of well in section
      - labels:      dict with 1D arrays at well column (velocity, porosity, etc.)
    """

    LOG_TYPES = ["GR", "NPHI", "RHOB", "DT", "RT", "velocity"]
    TASK_KEYS = ["velocity", "porosity", "lithology", "density", "resistivity"]

    def __init__(self, hdf5_path: str, augment: bool = True,
                 normalize: bool = True, has_labels: bool = True):
        self.hdf5_path = hdf5_path
        self.augment = augment
        self.normalize = normalize
        self.has_labels = has_labels
        self.aug_fn = SeismicAugmentation() if augment else None

        with h5py.File(hdf5_path, 'r') as f:
            self.n_sections = len(f["seismic"])
            self.has_label_group = "labels" in f

    def __len__(self) -> int:
        return self.n_sections

    def __getitem__(self, idx: int) -> Dict:
        with h5py.File(self.hdf5_path, 'r') as f:
            sec_key = f"seismic/section_{idx:04d}"

            # ── Seismic ──────────────────────────────────────
            seismic = f[sec_key][:].astype(np.float32)
            well_x = int(f[sec_key].attrs.get("well_x",
                         f[sec_key].attrs.get("well_index", 0)))

            # ── Well logs ────────────────────────────────────
            well_id = f[sec_key].attrs.get("well_id", f"WELL_{(idx + 1):04d}")
            logs = []
            well_path = f"wells/{well_id}"
            if well_path in f:
                for lt in self.LOG_TYPES:
                    if lt in f[well_path]:
                        logs.append(f[f"{well_path}/{lt}"][:].astype(np.float32))
                    else:
                        logs.append(np.zeros(256, dtype=np.float32))
            else:
                logs = [np.zeros(256, dtype=np.float32) for _ in self.LOG_TYPES]
            well_logs = np.stack(logs, axis=0)

            # ── Labels (1D at well column) ───────────────────
            labels = None
            if self.has_labels and self.has_label_group:
                lbl_key = f"labels/section_{idx:04d}"
                if lbl_key in f:
                    labels = {}
                    for task in self.TASK_KEYS:
                        if task in f[lbl_key]:
                            labels[task] = f[lbl_key][task][:].copy()

        # Augment
        if self.aug_fn is not None:
            seismic, well_x = self.aug_fn(seismic, well_x)

        # Normalize
        if self.normalize:
            s_mean, s_std = seismic.mean(), seismic.std() + 1e-8
            seismic = (seismic - s_mean) / s_std
            w_mean = well_logs.mean(axis=1, keepdims=True)
            w_std = well_logs.std(axis=1, keepdims=True) + 1e-8
            well_logs = (well_logs - w_mean) / w_std

        if seismic.ndim == 2:
            seismic = seismic[np.newaxis, :, :]

        result = {
            "seismic": torch.from_numpy(seismic).float(),
            "well_logs": torch.from_numpy(well_logs).float(),
            "well_x": well_x,
            "index": idx,
        }

        if labels is not None:
            lbl_tensors = {}
            for k, v in labels.items():
                if v.dtype == np.int64 or v.dtype == np.int32:
                    lbl_tensors[k] = torch.from_numpy(v).long()
                else:
                    lbl_tensors[k] = torch.from_numpy(v).float()
            result["labels"] = lbl_tensors

        return result


def collate_dense(batch: List[Dict]) -> Dict:
    """Collate function for dense training."""
    seismic = torch.stack([b["seismic"] for b in batch], dim=0)
    well_logs = torch.stack([b["well_logs"] for b in batch], dim=0)
    well_x = torch.tensor([b["well_x"] for b in batch], dtype=torch.long)
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)

    result = {
        "seismic": seismic,
        "well_logs": well_logs,
        "well_x": well_x,
        "index": indices,
    }

    if "labels" in batch[0] and batch[0]["labels"] is not None:
        labels = {}
        for task in batch[0]["labels"].keys():
            try:
                labels[task] = torch.stack([b["labels"][task]
                                            for b in batch], dim=0)
            except (KeyError, RuntimeError):
                labels[task] = None
        result["labels"] = labels

    return result
