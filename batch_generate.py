#!/usr/bin/env python3
"""
Batch generate large-scale geological segmentation dataset.

Uses RealisticModelGenerator (v2) with North Sea physical parameters
and improved rock physics well log generation.

Output: openseisml_geological_large.h5 with N samples,
each having seismic (256x512), well logs (6x256),
and 4 geological labels (fault, horizon, facies, fracture) at full 2D.
"""
import sys, os
sys.path.insert(0, "/data/yxjiang/datatest")

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, zoom, gaussian_filter1d
from scipy import signal as scipy_signal
import h5py, argparse, time
from typing import Tuple

from openseisml_pipeline_v2 import RealisticModelGenerator, seismic_from_velocity
from openseisml_pipeline import (
    SeismicConfig, WellLogConfig, SyntheticDataGenerator,
    RBFVelocityBuilder, TimeDepthConverter, FFTResampler, DatasetAssembler
)


def extract_section_2d(data_3d: np.ndarray, section_type: str,
                        ix: int, iy: int, target_shape: Tuple[int, int],
                        is_class: bool = False) -> np.ndarray:
    """Extract 2D section from 3D volume."""
    H, W = target_shape
    if section_type == "inline":
        s = data_3d[ix % data_3d.shape[0], :, :].T  # (nz, nx)
    else:
        s = data_3d[:, iy % data_3d.shape[1], :].T

    order = 0 if is_class else 1
    s = zoom(s, (H / s.shape[0], W / s.shape[1]), order=order)
    return s[:H, :W]


def generate_facies_2d(Vsh_3d: np.ndarray, v_model: np.ndarray,
                        section_type: str, ix: int, iy: int,
                        target_shape: Tuple[int, int]) -> np.ndarray:
    """5-class facies from Vsh + velocity (rock-physics-based)."""
    z_arr = np.linspace(0, 3000, v_model.shape[2]).reshape(1, 1, -1)
    v_cmp = 1500.0 + 0.8 * z_arr
    delta_v = v_model - v_cmp
    delta_v_s = gaussian_filter(delta_v.astype(np.float64), sigma=(1, 1, 3))

    facies_3d = np.zeros(v_model.shape, dtype=np.int64)

    # Sand: low Vsh, moderate velocity anomaly
    mask_sand = (Vsh_3d < 0.45) & (delta_v_s > -200) & (delta_v_s < 500)
    facies_3d[mask_sand] = 2
    # Silt: moderate Vsh
    mask_silt = (Vsh_3d >= 0.35) & (Vsh_3d < 0.55) & (facies_3d == 0)
    facies_3d[mask_silt] = 1
    # Carbonate: low Vsh, high velocity
    mask_carb = (Vsh_3d < 0.35) & (delta_v_s > 400)
    facies_3d[mask_carb] = 3
    # Basement: extreme velocity
    mask_base = (delta_v_s > 700) & (Vsh_3d < 0.25)
    facies_3d[mask_base] = 4

    return extract_section_2d(facies_3d, section_type, ix, iy, target_shape, is_class=True)


def generate_fracture_2d(v_model: np.ndarray, fault_mask_3d: np.ndarray,
                          section_type: str, ix: int, iy: int,
                          target_shape: Tuple[int, int]) -> np.ndarray:
    """Fracture from fault damage zones + curvature."""
    from scipy.ndimage import binary_dilation

    # Dilate fault mask for damage zone
    fault_bin = fault_mask_3d > 0.3
    structure = np.ones((3, 3, 3))
    damage = binary_dilation(fault_bin, structure=structure, iterations=2).astype(np.float32)

    # Curvature
    curv = np.abs(sobel(gaussian_filter(v_model, sigma=(0.5, 0.5, 0.5)), axis=1) +
                  sobel(gaussian_filter(v_model, sigma=(0.5, 0.5, 0.5)), axis=0))
    curv_norm = curv / (curv.std() + 1e-10)
    high_curv = (curv_norm > 3.0).astype(np.float32)

    fracture_3d = np.maximum(damage, high_curv * 0.3)
    fracture_3d = gaussian_filter(fracture_3d, sigma=(0.3, 0.3, 0.3))
    fracture_3d[fracture_3d < 0.12] = 0.0

    return extract_section_2d(fracture_3d, section_type, ix, iy, target_shape)


def generate_one_sample(seed: int, sc: SeismicConfig, wc: WellLogConfig
                         ) -> dict:
    """Generate one complete sample (seismic + well + labels)."""
    rng = np.random.RandomState(seed)

    # 1. Velocity model + geological features
    gen = RealisticModelGenerator(
        ni=sc.n_inline, nx=sc.n_xline, nz=sc.n_samples,
        z_max=3000.0, seed=seed
    )
    v_model, fault_3d, horizon_3d, Vsh_3d = gen.generate_full_model()

    # 2. Seismic
    seismic_3d = seismic_from_velocity(v_model)

    # 3. Well logs using improved rock physics
    syn_gen = SyntheticDataGenerator(sc, wc)
    syn_gen.rng = rng
    well_locs = syn_gen.generate_well_locations()
    wells = syn_gen.generate_well_logs(well_locs, v_model)
    checkshots = syn_gen.generate_checkshots(wells)

    # 4. Grid extraction (bounding box around wells)
    from openseisml_pipeline import GridExtractor
    ex = GridExtractor(sc)
    seismic_rect, wl_shifted = ex.extract(seismic_3d, well_locs, wells)

    # Re-extract fault/horizon/Vsh on same grid
    mx0 = max(0, int(np.floor(well_locs[:, 0].min())) - 5)
    mx1 = min(v_model.shape[0], int(np.ceil(well_locs[:, 0].max())) + 5)
    my0 = max(0, int(np.floor(well_locs[:, 1].min())) - 5)
    my1 = min(v_model.shape[1], int(np.ceil(well_locs[:, 1].max())) + 5)
    fault_rect = fault_3d[mx0:mx1, my0:my1, :]
    horizon_rect = horizon_3d[mx0:mx1, my0:my1, :]
    Vsh_rect = Vsh_3d[mx0:mx1, my0:my1, :]
    v_rect = v_model[mx0:mx1, my0:my1, :]

    # 5. RBF velocity + time-depth conversion
    rbf = RBFVelocityBuilder(sc)
    v_avg = rbf.build(checkshots, seismic_rect.shape)

    tdc = TimeDepthConverter(sc)
    seismic_depth, depth_axis = tdc.convert(seismic_rect, v_avg)

    # 6. Quasi-2D extraction
    from openseisml_pipeline import Quasi2DExtractor
    q2d = Quasi2DExtractor(sc)
    sections = q2d.extract(seismic_depth, wl_shifted, wells)

    # 7. FFT resample
    rs = FFTResampler(sc)
    for sec in sections:
        sec["section"] = rs.resample(sec["section"])

    # 8. Harmonize wells
    asm = DatasetAssembler(sc, wc)
    wells_h = asm.harmonize_wells(
        wells, sc.output_height, (depth_axis[0], depth_axis[-1])
    )

    # 9. Extract 2D labels
    results = {
        "sections": sections,
        "wells": wells_h,
        "fault_rect": fault_rect,
        "horizon_rect": horizon_rect,
        "v_rect": v_rect,
        "Vsh_rect": Vsh_rect,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--output", type=str,
                        default="./openseisml_geological_large.h5")
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    sc = SeismicConfig()
    sc.n_wells = 1  # one well per sample (we generate N independent samples)
    sc.seed = args.start_seed
    wc = WellLogConfig()

    print(f"Generating {args.n_samples} geological samples...")

    with h5py.File(args.output, 'w') as f:
        sg = f.create_group("seismic")
        wg = f.create_group("wells")
        lg = f.create_group("labels")

        t0 = time.time()
        for idx in range(args.n_samples):
            seed = args.start_seed + idx
            sample = generate_one_sample(seed, sc, wc)

            # Save seismic section
            sec = sample["sections"][0]
            ds = sg.create_dataset(f"section_{idx:05d}", data=sec["section"],
                                   compression="gzip", compression_opts=4)
            ds.attrs["well_id"] = f"WELL_{idx+1:05d}"
            ds.attrs["section_type"] = sec.get("section_type", "inline")
            ds.attrs["well_x"] = sec["well_index"]
            ds.attrs["well_index"] = sec["well_index"]
            ds.attrs["sampling_interval_m"] = 12.5

            # Save well
            well = sample["wells"][0]
            wgrp = wg.create_group(f"WELL_{idx+1:05d}")
            for k, v in well.items():
                if isinstance(v, np.ndarray):
                    wgrp.create_dataset(k, data=v.astype(np.float32))
            wgrp.attrs["well_x"] = sec["well_index"]

            # Save 2D labels
            H, W = sec["section"].shape
            stype = sec.get("section_type", "inline")
            ix = min(int(sec.get("inline", 0)), sample["fault_rect"].shape[0] - 1)
            iy = min(int(sec.get("xline", 0)), sample["fault_rect"].shape[1] - 1)

            lgrp = lg.create_group(f"section_{idx:05d}")
            lgrp.create_dataset("fault",
                data=extract_section_2d(sample["fault_rect"], stype, ix, iy, (H, W)))
            lgrp.create_dataset("horizon",
                data=extract_section_2d(sample["horizon_rect"], stype, ix, iy, (H, W), is_class=True))
            lgrp.create_dataset("facies",
                data=generate_facies_2d(sample["Vsh_rect"], sample["v_rect"], stype, ix, iy, (H, W)))
            lgrp.create_dataset("fracture",
                data=generate_fracture_2d(sample["v_rect"], sample["fault_rect"], stype, ix, iy, (H, W)))
            lgrp.attrs["well_x"] = sec["well_index"]

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (args.n_samples - idx - 1) / rate
                print(f"  [{idx+1:4d}/{args.n_samples}]  "
                      f"{elapsed:.0f}s elapsed, {rate:.1f} samples/min, ETA {eta:.0f}s")

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["n_sections"] = args.n_samples
        meta.attrs["n_wells"] = args.n_samples
        meta.attrs["n_horizon_classes"] = 13
        meta.attrs["n_facies_classes"] = 5
        meta.attrs["label_type"] = "geological_segmentation"
        meta.attrs["generator_version"] = "v2_realistic"
        meta.attrs["physics"] = "Faust/Athy compaction, listric faults (RGM-style), "
        meta.attrs["physics"] += "Wyllie porosity, Gardner density, Archie resistivity"
        meta.attrs["sampling_interval_m"] = 12.5
        meta.attrs["section_shape"] = "256x512"

    total_t = time.time() - t0
    print(f"\nDone! {args.n_samples} samples in {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
