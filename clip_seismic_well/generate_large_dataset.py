#!/usr/bin/env python3
"""
Generate larger dataset for CLIP training.
Extends the OpenSeisML pipeline to produce more wells/sections.
"""
import sys
sys.path.insert(0, "/data/yxjiang/datatest")
import numpy as np
from openseisml_pipeline import (
    SeismicConfig, WellLogConfig, SyntheticDataGenerator,
    GridExtractor, RBFVelocityBuilder, TimeDepthConverter,
    Quasi2DExtractor, FFTResampler, DatasetAssembler
)
from scipy.ndimage import gaussian_filter1d
import os, time, argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wells", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default="./openseisml_dataset_large.h5")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    sc = SeismicConfig()
    sc.n_wells = args.n_wells
    sc.seed = args.seed
    sc.n_inline = max(sc.n_inline, int(np.sqrt(args.n_wells) * 1.5))
    sc.n_xline = max(sc.n_xline, int(np.sqrt(args.n_wells) * 1.5))

    wc = WellLogConfig()

    print(f"Generating dataset with {args.n_wells} wells")
    print(f"Volume: {sc.n_inline}×{sc.n_xline}×{sc.n_samples}")

    # Step 1: Generate
    gen = SyntheticDataGenerator(sc, wc)
    seismic_time, v_model = gen.generate_seismic()
    well_locs = gen.generate_well_locations()
    wells = gen.generate_well_logs(well_locs, v_model)
    checkshots = gen.generate_checkshots(wells)

    # Step 2: Grid extraction
    ex = GridExtractor(sc)
    seismic_rect, wl_shifted = ex.extract(seismic_time, well_locs, wells)

    # Step 3: RBF velocity
    rbf = RBFVelocityBuilder(sc)
    v_avg = rbf.build(checkshots, seismic_rect.shape)

    # Step 4: Time-depth conversion
    tdc = TimeDepthConverter(sc)
    seismic_depth, depth_axis = tdc.convert(seismic_rect, v_avg)

    # Step 5: Quasi-2D extraction
    q2d = Quasi2DExtractor(sc)
    sections = q2d.extract(seismic_depth, wl_shifted, wells)

    # Step 6: FFT resample
    rs = FFTResampler(sc)
    for sec in sections:
        sec["section"] = rs.resample(sec["section"])

    # Step 7: Harmonize & save
    asm = DatasetAssembler(sc, wc)
    wells_h = asm.harmonize_wells(
        wells, sc.output_height,
        (depth_axis[0], depth_axis[-1])
    )
    asm.save(sections, wells_h, args.output,
             v_avg_3d=v_avg, depth_axis=depth_axis)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
