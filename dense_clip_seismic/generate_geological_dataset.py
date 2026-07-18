#!/usr/bin/env python3
"""
Generate geological segmentation dataset.

Each sample = seismic image (256x512) + well logs (6x256)
            + full 2D geological labels (256x512 each):
              - fault:     binary, from explicit fault plane mask
              - horizon:   N+1 classes, from layer boundary surfaces
              - facies:    5 classes, from rock-physics facies classification
              - fracture:  binary, fault-zone damage + high-curvature zones

Physical principles (see plan for references):
  - Fault labels come from EXPLICIT fault simulation (not gradient heuristics)
  - Horizon labels follow stratal surfaces (impedance contrast interfaces)
  - Facies labels use Vsh-based classification (velocity alone is ambiguous)
  - Fracture labels are derived from fault damage zones (not random noise)
"""
import sys, os
sys.path.insert(0, "/data/yxjiang/datatest")

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, zoom, gaussian_filter1d
from typing import Tuple, Dict, List
import h5py, argparse

from openseisml_pipeline import (
    SeismicConfig, WellLogConfig, SyntheticDataGenerator,
    GridExtractor, RBFVelocityBuilder, TimeDepthConverter,
    Quasi2DExtractor, FFTResampler, DatasetAssembler
)


class GeologicalLabelGenerator:
    """
    Generate geological labels with physically-grounded methods.

    Labels are derived from KNOWN structural positions (not heuristics):
      - fault: from explicit fault mask
      - horizon: from layer boundary surfaces (velocity gradient maxima)
      - facies: from Vsh-based classification (not velocity ranges)
      - fracture: from fault damage zones (not random heterogeneity)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_fault_labels(self, fault_mask_3d: np.ndarray) -> np.ndarray:
        """
        Fault labels from explicit fault simulation.

        The fault_mask_3d is a binary float32 mask produced during
        velocity model generation. It marks actual fault plane positions
        with displacement, NOT gradient-derived heuristics.

        Returns: (ni, nx, nz) float32 soft binary
        """
        return fault_mask_3d.astype(np.float32)

    def generate_horizon_labels(self, v_model: np.ndarray,
                                 n_horizons: int = 12) -> np.ndarray:
        """
        Horizon labels from stratal surfaces (layer boundaries).

        Horizons are stratal surfaces = impedance contrast interfaces.
        In the velocity model, each layer perturbation creates such
        an interface. We detect them via vertical gradient peaks.

        Each horizon gets a thin label (1-2 pixel thickness after
        smoothing) — matching real seismic where a horizon is a
        single reflection event, not a thick zone.

        Returns: (ni, nx, nz) int64 (0=bg, 1..N_horizons)
        """
        ni, nx, nz = v_model.shape

        # Vertical gradient = interface indicator
        grad_z = np.abs(np.diff(v_model, axis=2))
        grad_z = np.pad(grad_z, ((0, 0), (0, 0), (0, 1)), mode='edge')
        grad_z_smooth = gaussian_filter(grad_z, sigma=(1.0, 1.0, 2.0))

        labels = np.zeros((ni, nx, nz), dtype=np.int64)

        for i in range(ni):
            for j in range(nx):
                profile = grad_z_smooth[i, j, :]
                # Find prominent peaks (layer boundaries)
                from scipy import signal as scipy_signal
                prominence = profile.std() * 0.4
                peaks = scipy_signal.find_peaks(
                    profile,
                    distance=max(8, nz // (n_horizons + 1)),
                    prominence=prominence,
                )[0]

                # Assign horizon IDs, thin width
                for h_idx, peak_pos in enumerate(peaks[:n_horizons]):
                    hw = max(1, nz // 300)  # thin: 1-2 pixels after resampling
                    lo = max(0, peak_pos - hw)
                    hi = min(nz, peak_pos + hw + 1)
                    labels[i, j, lo:hi] = h_idx + 1

        return labels.astype(np.int64)

    def generate_facies_labels(self, v_model: np.ndarray,
                                ni: int, nx: int, nz: int) -> np.ndarray:
        """
        Facies labels from Vsh-based classification.

        Real facies discrimination:
          - Shale:  high Vsh, low-moderate velocity
          - Silt:   moderate Vsh, moderate velocity
          - Sand:   low Vsh, variable velocity (clean quartz)
          - Carbonate: very low Vsh, HIGH velocity
          - Basement: very high velocity, very low Vsh

        Vsh is derived from velocity deviation from compaction trend,
        NOT from velocity alone. Two rocks with same velocity but
        different Vsh → different facies.

        Returns: (ni, nx, nz) int64 (0=shale, 1=silt, 2=sand, 3=carbonate, 4=basement)
        """
        # Compaction trend
        z_arr = np.linspace(0, 3000, nz).reshape(1, 1, nz)
        v_cmp = 1500.0 + 0.8 * z_arr
        delta_v = v_model - v_cmp
        delta_v_smooth = gaussian_filter(delta_v.astype(np.float64),
                                          sigma=(1, 1, 3))

        # Vsh model: negative delta_v (slower than trend) = less compacted = more shale-prone
        # Positive delta_v (faster) = more cemented = clean sand or carbonate
        vsh = 0.5 - 0.3 * np.tanh(delta_v_smooth / 300.0)  # ~[0.2, 0.8]
        vsh += 0.1 * self.rng.normal(0, 1, vsh.shape)
        vsh = np.clip(vsh, 0.0, 1.0)

        # Classification rules
        facies = np.zeros((ni, nx, nz), dtype=np.int64)  # default: shale

        # Sand: low Vsh, moderate velocity anomaly
        facies[(vsh < 0.45) & (delta_v_smooth > -200) & (delta_v_smooth < 500)] = 2
        # Silt: moderate Vsh
        facies[(vsh >= 0.35) & (vsh < 0.55)] = 1
        # Carbonate: low Vsh, HIGH positive velocity
        facies[(vsh < 0.35) & (delta_v_smooth > 400)] = 3
        # Basement: extreme velocity
        facies[(delta_v_smooth > 700) & (vsh < 0.25)] = 4

        return facies.astype(np.int64)

    def generate_fracture_labels(self, v_model: np.ndarray,
                                  fault_mask_3d: np.ndarray) -> np.ndarray:
        """
        Fracture labels from fault damage zones.

        Real fractures cluster around faults — the damage zone is
        where the rock is most fractured. This is well documented in
        structural geology (Torabi et al., 2024).

        Additional fracture-prone zones: high-curvature areas
        (fold hinges, channel edges).

        Returns: (ni, nx, nz) float32 soft binary
        """
        ni, nx, nz = v_model.shape

        # 1. Fault damage zones (dilated fault mask)
        from scipy.ndimage import binary_dilation
        fault_binary = fault_mask_3d > 0.3
        # Dilate fault mask to represent damage zone (wider than fault slip surface)
        structure = np.ones((3, 3, 3))
        damage_zone = binary_dilation(fault_binary, structure=structure, iterations=2)
        damage_zone = damage_zone.astype(np.float32)

        # 2. High-curvature zones (fold hinges, channel edges)
        # Curvature = second derivative of structure
        grad_x = sobel(v_model, axis=1)
        grad_y = sobel(v_model, axis=0)
        curv = sobel(grad_x, axis=1) + sobel(grad_y, axis=0)  # simplified
        curv_abs = np.abs(curv)
        curv_std = curv_abs.std()
        high_curv = (curv_abs > 2.5 * curv_std).astype(np.float32)

        # Combine: damage zones + high curvature
        fracture = np.maximum(damage_zone, high_curv * 0.3)

        # Localize: fractures are narrow features
        fracture = gaussian_filter(fracture, sigma=(0.3, 0.3, 0.3))

        # Only near faults or structural deformation (NOT random noise)
        # Suppress isolated noise pixels
        fracture[fracture < 0.15] = 0.0

        return fracture.astype(np.float32)


def extract_2d_slice(data_3d: np.ndarray, section: dict,
                      target_shape: Tuple[int, int]) -> np.ndarray:
    """Extract 2D slice from 3D volume matching section geometry."""
    ni, nx, nz_data = data_3d.shape
    H, W = target_shape
    stype = section.get("section_type", "inline")
    ix = min(int(section.get("inline", 0)), ni - 1)
    iy = min(int(section.get("xline", 0)), nx - 1)

    if stype == "inline":
        # Inline slice: fixed inline, vary xline + depth
        slice_2d = data_3d[ix, :, :].T  # (nz, nx) → transpose for (depth, lateral)
    else:
        # Xline slice: fixed xline, vary inline + depth
        slice_2d = data_3d[:, iy, :].T

    # Resample to target size
    h_ratio = H / slice_2d.shape[0]
    w_ratio = W / slice_2d.shape[1]

    if data_3d.dtype in (np.int64, np.int32):
        slice_2d = zoom(slice_2d, (h_ratio, w_ratio), order=0)
    else:
        slice_2d = zoom(slice_2d, (h_ratio, w_ratio), order=1)

    return slice_2d[:H, :W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wells", type=int, default=80)
    parser.add_argument("--output", type=str,
                        default="./openseisml_geological.h5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sc = SeismicConfig()
    sc.n_wells = args.n_wells
    sc.seed = args.seed
    wc = WellLogConfig()

    print(f"Generating {args.n_wells}-well geological segmentation dataset...")

    # ── Generate base data ───────────────────────────────────
    gen = SyntheticDataGenerator(sc, wc)
    seismic_time, v_model, fault_mask_3d = gen.generate_seismic()
    well_locs = gen.generate_well_locations()
    wells = gen.generate_well_logs(well_locs, v_model)
    checkshots = gen.generate_checkshots(wells)

    # ── Pipeline steps ───────────────────────────────────────
    ex = GridExtractor(sc)
    seismic_rect, wl_shifted = ex.extract(seismic_time, well_locs, wells)

    rbf = RBFVelocityBuilder(sc)
    v_avg = rbf.build(checkshots, seismic_rect.shape)

    tdc = TimeDepthConverter(sc)
    seismic_depth, depth_axis = tdc.convert(seismic_rect, v_avg)

    q2d = Quasi2DExtractor(sc)
    sections = q2d.extract(seismic_depth, wl_shifted, wells)

    rs = FFTResampler(sc)
    for sec in sections:
        sec["section"] = rs.resample(sec["section"])

    # Harmonize wells
    asm = DatasetAssembler(sc, wc)
    wells_h = asm.harmonize_wells(
        wells, sc.output_height, (depth_axis[0], depth_axis[-1])
    )

    # ── Generate geological labels ───────────────────────────
    print("\nGenerating geological labels...")
    label_gen = GeologicalLabelGenerator(seed=args.seed)

    fault_3d = label_gen.generate_fault_labels(fault_mask_3d)
    horizon_3d = label_gen.generate_horizon_labels(v_model)
    facies_3d = label_gen.generate_facies_labels(
        v_model, v_model.shape[0], v_model.shape[1], v_model.shape[2]
    )
    fracture_3d = label_gen.generate_fracture_labels(v_model, fault_mask_3d)

    print(f"  Fault:     non-zero={fault_3d.mean():.4f}")
    print(f"  Horizon:   n_classes={horizon_3d.max()}")
    print(f"  Facies:    classes={np.unique(facies_3d)}")
    print(f"  Fracture:  non-zero={fracture_3d.mean():.4f}")

    # ── Save HDF5 ────────────────────────────────────────────
    print(f"\nSaving to: {args.output}")

    with h5py.File(args.output, 'w') as f:
        # Seismic
        sg = f.create_group("seismic")
        for i, sec in enumerate(sections):
            ds = sg.create_dataset(f"section_{i:04d}", data=sec["section"],
                                   compression="gzip", compression_opts=4)
            ds.attrs["well_id"] = sec["well_id"]
            ds.attrs["section_type"] = sec.get("section_type", "inline")
            ds.attrs["well_index"] = sec["well_index"]
            ds.attrs["well_x"] = sec["well_index"]

        # Wells
        wg = f.create_group("wells")
        for well_h in wells_h:
            wgrp = wg.create_group(well_h["well_id"])
            for k, v in well_h.items():
                if isinstance(v, np.ndarray):
                    wgrp.create_dataset(k, data=v.astype(np.float32))

        # Full 2D geological labels
        lg = f.create_group("labels")
        for i, sec in enumerate(sections):
            lgrp = lg.create_group(f"section_{i:04d}")

            H, W = sec["section"].shape

            fault_2d = extract_2d_slice(fault_3d, sec, (H, W))
            horizon_2d = extract_2d_slice(horizon_3d, sec, (H, W))
            facies_2d = extract_2d_slice(facies_3d, sec, (H, W))
            fracture_2d = extract_2d_slice(fracture_3d, sec, (H, W))

            lgrp.create_dataset("fault", data=fault_2d.astype(np.float32))
            lgrp.create_dataset("horizon", data=horizon_2d.astype(np.int64))
            lgrp.create_dataset("facies", data=facies_2d.astype(np.int64))
            lgrp.create_dataset("fracture", data=fracture_2d.astype(np.float32))
            lgrp.attrs["well_x"] = sec["well_index"]

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["n_sections"] = len(sections)
        meta.attrs["n_wells"] = len(wells_h)
        meta.attrs["n_horizon_classes"] = int(horizon_3d.max())
        meta.attrs["n_facies_classes"] = 5
        meta.attrs["label_type"] = "geological_segmentation"
        meta.attrs["fault_method"] = "explicit fault planes with throw/displacement"
        meta.attrs["horizon_method"] = "stratal surface boundaries"
        meta.attrs["facies_method"] = "Vsh-based classification"
        meta.attrs["fracture_method"] = "fault damage zones + structural curvature"

    print(f"✓ Saved: {args.output}")
    print(f"  {len(sections)} sections × ({H}×{W})")
    print(f"  4 label types: fault (binary), horizon ({horizon_3d.max()}+1 classes),")
    print(f"                 facies (5 classes), fracture (binary)")


if __name__ == "__main__":
    main()
