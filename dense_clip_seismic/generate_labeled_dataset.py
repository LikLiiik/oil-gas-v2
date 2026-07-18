#!/usr/bin/env python3
"""
Generate labeled dataset for petrophysical property prediction.

Core competition concept:
  - Well logs provide ground truth petrophysical measurements at ONE column
  - The model must use seismic context to propagate these laterally
  - Contrastive learning aligns the modalities
  - Task heads predict velocity, porosity, lithology, density, resistivity

Labels:
  - velocity:    from well log (m/s)
  - porosity:    NPHI from well log (v/v)
  - lithology:   derived from GR (0=shale, 1=silt, 2=sand)
  - density:     RHOB from well log (g/cm³)
  - resistivity: RT from well log (ohm·m)

Each label is a 2D map (256×512) but ONLY valid at the well column.
During training, loss is computed at the well column only.
"""
import sys
sys.path.insert(0, "/data/yxjiang/datatest")

import numpy as np
from openseisml_pipeline import (
    SeismicConfig, WellLogConfig, SyntheticDataGenerator,
    GridExtractor, RBFVelocityBuilder, TimeDepthConverter,
    Quasi2DExtractor, FFTResampler, DatasetAssembler
)
import h5py, argparse
from scipy.ndimage import gaussian_filter1d


def gr_to_lithology(gr: np.ndarray) -> np.ndarray:
    """Convert GR to 3-class lithology: 0=shale(high GR), 1=silt(mid), 2=sand(low)."""
    gr_smooth = gaussian_filter1d(gr.astype(np.float64), sigma=3)
    p33, p67 = np.percentile(gr_smooth, [33, 67])
    litho = np.zeros_like(gr, dtype=np.int64)
    litho[gr_smooth < p33] = 2    # sand (low GR)
    litho[(gr_smooth >= p33) & (gr_smooth < p67)] = 1  # silt
    litho[gr_smooth >= p67] = 0   # shale (high GR)
    return litho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wells", type=int, default=80)
    parser.add_argument("--output", type=str,
                        default="./openseisml_dataset_labeled.h5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sc = SeismicConfig()
    sc.n_wells = args.n_wells
    sc.seed = args.seed
    wc = WellLogConfig()

    print(f"Generating {args.n_wells}-well petrophysical dataset...")

    # Generate base data
    gen = SyntheticDataGenerator(sc, wc)
    seismic_time, v_model = gen.generate_seismic()
    well_locs = gen.generate_well_locations()
    wells = gen.generate_well_logs(well_locs, v_model)
    checkshots = gen.generate_checkshots(wells)

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

    # Harmonize wells to 256 samples
    print("Harmonizing well logs...")
    asm = DatasetAssembler(sc, wc)
    wells_h = asm.harmonize_wells(wells, sc.output_height,
                                  (depth_axis[0], depth_axis[-1]))

    # ── Build petrophysical labels ───────────────────────────
    print("Building petrophysical labels...")

    # 3D velocity model → per-section labels (interpolate to 2D section)
    ni, nx, nz = v_model.shape
    z_3d = np.linspace(0, 3000, nz)
    z_2d = np.linspace(0, depth_axis[-1], 256)

    label_keys = ["velocity", "porosity", "lithology", "density", "resistivity"]

    with h5py.File(args.output, 'w') as f:
        # ── Seismic sections ─────────────────────────────────
        sg = f.create_group("seismic")
        for i, sec in enumerate(sections):
            ds = sg.create_dataset(f"section_{i:04d}", data=sec["section"],
                                   compression="gzip", compression_opts=4)
            ds.attrs["well_id"] = sec["well_id"]
            ds.attrs["section_type"] = sec["section_type"]
            ds.attrs["well_index"] = sec["well_index"]
            ds.attrs["well_x"] = sec["well_index"]

        # ── Wells ────────────────────────────────────────────
        wg = f.create_group("wells")
        for well in wells_h:
            wgrp = wg.create_group(well["well_id"])
            for k, v in well.items():
                if isinstance(v, np.ndarray):
                    wgrp.create_dataset(k, data=v.astype(np.float32))

        # ── Petrophysical labels at well positions ────────────
        lg = f.create_group("labels")
        for i, well in enumerate(wells_h):
            lgrp = lg.create_group(f"section_{i:04d}")
            wx = sections[i]["well_index"]

            # Extract well log values (1D → 2D mask along well column)
            H, W = 256, 512

            for key in label_keys:
                # Map label key → well log key
                key_map = {
                    "velocity": "velocity",
                    "porosity": "NPHI",
                    "density": "RHOB",
                    "resistivity": "RT",
                    "lithology": "GR",  # special: derived from GR
                }
                well_log_key = key_map.get(key, key)

                if key == "lithology":
                    val_1d = gr_to_lithology(well["GR"]).astype(np.float32)
                elif well_log_key in well:
                    val_1d = well[well_log_key].astype(np.float32)  # (256,)
                else:
                    val_1d = np.zeros(256, dtype=np.float32)

                # Store as 1D column (the full 2D map is unnecessary;
                # training loss is computed at well column only)
                lgrp.create_dataset(key, data=val_1d)

            # Also store well_x for easy access
            lgrp.attrs["well_x"] = wx

        # ── Metadata ─────────────────────────────────────────
        meta = f.create_group("metadata")
        meta.attrs["n_sections"] = len(sections)
        meta.attrs["n_wells"] = len(wells_h)
        meta.attrs["label_tasks"] = ",".join(label_keys)
        meta.attrs["description"] = (
            "Petrophysical property prediction dataset. "
            "Labels valid at well column only; model must propagate "
            "laterally using seismic context via contrastive learning."
        )

    print(f"✓ Saved: {args.output}")
    print(f"  {len(sections)} sections, {len(wells_h)} wells")
    print(f"  Tasks: {label_keys}")
    # Stats
    print(f"\nLabel statistics (well column):")
    for key in label_keys:
        vals = []
        for i in range(min(10, len(wells_h))):
            vals.append(wells_h[i].get(key, wells_h[i].get("GR", np.zeros(1))).mean())
        print(f"  {key:15s}: mean={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
