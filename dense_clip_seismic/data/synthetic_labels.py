"""
Synthetic geological label generation.

Extends SyntheticDataGenerator to produce pixel-level labels for:
  - Faults:        from velocity discontinuities
  - Horizons:      from layer boundaries
  - Facies:        from velocity/density ranges (sand/shale/etc.)
  - Fractures:     from small-scale heterogeneity patterns

These labels are used to train/evaluate the task-specific heads
and to demonstrate that cross-modal fusion improves prediction.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, gaussian_gradient_magnitude
from scipy import signal as scipy_signal
from typing import Tuple, Dict, List
import h5py


class GeologicalLabelGenerator:
    """
    Generate synthetic geological labels from the velocity model
    and seismic data produced by SyntheticDataGenerator.
    """

    FACIES_NAMES = ["shale", "silt", "sand", "carbonate", "basement"]
    FACIES_VELOCITY_RANGES = [
        (1500, 2200),   # shale
        (2200, 2800),   # silt
        (2800, 3600),   # sand
        (3600, 4800),   # carbonate
        (4800, 6500),   # basement
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_fault_labels(self, v_model: np.ndarray,
                              threshold: float = 0.03) -> np.ndarray:
        """
        Detect fault-like discontinuities in velocity model.

        Uses horizontal gradient magnitude — faults appear as
        sharp horizontal velocity changes.

        Args:
            v_model: (ni, nx, nz) velocity model
            threshold: gradient magnitude threshold
        Returns:
            fault_mask: (ni, nx, nz) binary fault probability
        """
        # Horizontal gradient magnitude (faults = lateral discontinuities)
        grad_x = sobel(v_model, axis=1)
        grad_y = sobel(v_model, axis=0)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize per depth slice
        grad_norm = np.zeros_like(grad_mag)
        for z in range(grad_mag.shape[2]):
            s = grad_mag[:, :, z]
            sm = s.mean()
            ss = s.std() + 1e-8
            grad_norm[:, :, z] = (s - sm) / ss

        # Binary mask with soft edges
        fault_raw = (grad_norm > 2.5).astype(np.float32)
        fault_mask = gaussian_filter(fault_raw, sigma=(1, 1, 0.5))

        return fault_mask

    def generate_horizon_labels(self, v_model: np.ndarray,
                                n_horizons: int = 12) -> np.ndarray:
        """
        Detect layer boundaries (horizons) in velocity model.

        Uses vertical gradient — horizons are interfaces where
        velocity changes significantly in depth.

        Args:
            v_model: (ni, nx, nz) velocity model
            n_horizons: number of horizon classes
        Returns:
            horizon_labels: (ni, nx, nz) integer labels (0=bg, 1..N=horizons)
        """
        ni, nx, nz = v_model.shape

        # Vertical gradient
        grad_z = np.abs(np.diff(v_model, axis=2))
        grad_z = np.pad(grad_z, ((0, 0), (0, 0), (0, 1)), mode='edge')

        # Smooth and normalize
        grad_z_smooth = gaussian_filter(grad_z, sigma=(1, 1, 2))

        # Find local maxima in depth (horizon positions)
        labels = np.zeros((ni, nx, nz), dtype=np.int64)

        for i in range(ni):
            for j in range(nx):
                profile = grad_z_smooth[i, j, :]
                # Find peaks
                peaks = scipy_signal.find_peaks(
                    profile, distance=nz // (n_horizons + 1),
                    prominence=profile.std() * 0.5
                )[0]

                # Label horizons
                for h_idx, peak_pos in enumerate(peaks[:n_horizons]):
                    # Give a small thickness to each horizon
                    half_w = max(2, nz // 200)
                    z_min = max(0, peak_pos - half_w)
                    z_max = min(nz, peak_pos + half_w + 1)
                    labels[i, j, z_min:z_max] = h_idx + 1

        return labels.astype(np.int64)

    def generate_facies_labels(self, v_model: np.ndarray) -> np.ndarray:
        """
        Assign facies classes based on velocity ranges.

        This is a simplified sand/shale discrimination based on
        compaction-corrected velocity.

        Args:
            v_model: (ni, nx, nz) velocity model
        Returns:
            facies_labels: (ni, nx, nz) integer labels
        """
        ni, nx, nz = v_model.shape
        labels = np.zeros((ni, nx, nz), dtype=np.int64)

        # Add some spatial noise to make it realistic
        noise = self.rng.normal(0, 100, v_model.shape)
        v_noisy = v_model + noise

        for i, (v_min, v_max) in enumerate(self.FACIES_VELOCITY_RANGES):
            labels[(v_noisy >= v_min) & (v_noisy < v_max)] = i
        labels[v_noisy >= self.FACIES_VELOCITY_RANGES[-1][1]] = len(
            self.FACIES_VELOCITY_RANGES) - 1

        # Smooth boundaries
        labels = gaussian_filter(labels.astype(np.float32), sigma=(0.5, 0.5, 1))
        return np.clip(np.round(labels), 0,
                       len(self.FACIES_VELOCITY_RANGES) - 1).astype(np.int64)

    def generate_fracture_labels(self, v_model: np.ndarray) -> np.ndarray:
        """
        Detect fracture-like features from small-scale heterogeneity.

        Fractures appear as localized high-frequency perturbations
        in the velocity field.

        Args:
            v_model: (ni, nx, nz) velocity model
        Returns:
            fracture_mask: (ni, nx, nz) binary fracture probability
        """
        # Extract high-frequency component
        v_smooth = gaussian_filter(v_model, sigma=(2, 2, 1))
        v_high = v_model - v_smooth

        # Fractures are extreme high-frequency deviations
        v_high_std = np.std(v_high)
        fracture_raw = (np.abs(v_high) > 2.0 * v_high_std).astype(np.float32)

        # Small spatial extent and random orientation
        fracture_mask = gaussian_filter(fracture_raw, sigma=(0.3, 0.3, 0.3))

        return fracture_mask


class LabeledDatasetBuilder:
    """
    Build HDF5 dataset with geological labels for dense prediction tasks.

    Extends the OpenSeisML pipeline output format with additional
    label arrays for faults, horizons, facies, and fractures.
    """

    def __init__(self, label_gen: GeologicalLabelGenerator = None):
        self.label_gen = label_gen or GeologicalLabelGenerator()

    def build_from_pipeline(self, v_model: np.ndarray,
                            seismic_depth: np.ndarray,
                            sections: List[Dict],
                            wells: List[Dict],
                            well_locs: np.ndarray,
                            output_path: str):
        """
        Generate labels and save labeled dataset.

        Args:
            v_model: (ni, nx, nz) true velocity model
            seismic_depth: (ni, nx, nz) depth-domain seismic
            sections: list of 2D section dicts
            wells: list of well dicts
            well_locs: (n_wells, 2) well positions
            output_path: HDF5 output path
        """
        print("Generating geological labels...")

        # Generate 3D labels
        fault_3d = self.label_gen.generate_fault_labels(v_model)
        horizon_3d = self.label_gen.generate_horizon_labels(v_model)
        facies_3d = self.label_gen.generate_facies_labels(v_model)
        fracture_3d = self.label_gen.generate_fracture_labels(v_model)

        print(f"  Fault:     non-zero={fault_3d.mean():.4f}")
        print(f"  Horizon:   n_classes={horizon_3d.max()}")
        print(f"  Facies:    n_classes={facies_3d.max() + 1}")
        print(f"  Fracture:  non-zero={fracture_3d.mean():.6f}")

        with h5py.File(output_path, 'w') as f:
            # ── Seismic sections ──────────────────────────────
            sg = f.create_group("seismic")
            for i, sec in enumerate(sections):
                ds = sg.create_dataset(f"section_{i:04d}", data=sec["section"],
                                       compression="gzip", compression_opts=4)
                ds.attrs["well_id"] = sec["well_id"]
                ds.attrs["well_index"] = sec["well_index"]
                ds.attrs["well_x"] = sec["well_index"]
                ds.attrs["section_type"] = sec["section_type"]

            # ── Wells ─────────────────────────────────────────
            wg = f.create_group("wells")
            for well in wells:
                wgrp = wg.create_group(well["well_id"])
                for k, v in well.items():
                    if isinstance(v, np.ndarray):
                        wgrp.create_dataset(k, data=v.astype(np.float32))

            # ── Labels ────────────────────────────────────────
            lg = f.create_group("labels")
            for i, sec in enumerate(sections):
                sec_idx = sec.get("inline", i) if sec["section_type"] == "inline" else i
                xl_idx = sec.get("xline", i) if sec["section_type"] == "xline" else i

                lgrp = lg.create_group(f"section_{i:04d}")

                # Extract 2D labels from 3D volumes
                # For simplicity: extract at the well intersection column
                # (full 2D labeling requires more work)
                ix = int(sec.get("inline", i)) if "inline" in sec else i
                iy = int(sec.get("xline", i)) if "xline" in sec else i

                # Map section type to 2D extraction
                ni, nx, nz = v_model.shape
                ix_c = min(ix, ni - 1)
                iy_c = min(iy, nx - 1)

                lgrp.create_dataset("fault",
                                    data=fault_3d[ix_c, :, :nz].T.astype(np.float32))
                lgrp.create_dataset("horizon",
                                    data=horizon_3d[ix_c, :, :nz].T.astype(np.int64))
                lgrp.create_dataset("facies",
                                    data=facies_3d[ix_c, :, :nz].T.astype(np.int64))
                lgrp.create_dataset("fracture",
                                    data=fracture_3d[ix_c, :, :nz].T.astype(np.float32))

            # ── Metadata ──────────────────────────────────────
            meta = f.create_group("metadata")
            meta.attrs["n_sections"] = len(sections)
            meta.attrs["n_wells"] = len(wells)
            meta.attrs["n_facies"] = len(GeologicalLabelGenerator.FACIES_NAMES)
            meta.attrs["facies_names"] = ", ".join(
                GeologicalLabelGenerator.FACIES_NAMES)

        print(f"✓ Labeled dataset saved to: {output_path}")
        print(f"  Sections: {len(sections)} with 4 label types each")
