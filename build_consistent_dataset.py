#!/usr/bin/env python3
"""
Build physically-consistent geological dataset.

Key principle: downstream task labels MUST be consistent with
seismic images AND well log curves at every depth point.

Consistency fixes vs previous version:
  1. Facies labels use SAME classification as well log GR→facies
  2. Fault zones in well logs show real fault signatures (DT jump, density drop)
  3. Fracture labels in well logs show DT cycle-skipping, RT invasion
  4. Horizon boundaries align with well log lithology changes
"""
import sys, os, numpy as np, h5py, time
sys.path.insert(0, "/data/yxjiang/datatest")
from openseisml_pipeline_v2 import RealisticModelGenerator, seismic_from_velocity
from openseisml_pipeline import *
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel, zoom, binary_dilation

t0 = time.time()
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# 1. Generate large realistic 3D model
# ═══════════════════════════════════════════════════════════
print("Step 1: 3D geological model...")
gen = RealisticModelGenerator(ni=300, nx=400, nz=512, z_max=3000, seed=42)
v_model, fault_mask_3d, horizon_mask_3d, Vsh_3d = gen.generate_full_model()
seismic_3d = seismic_from_velocity(v_model)

# ═══════════════════════════════════════════════════════════
# 2. Facies: column-wise classification (SAME as well logs)
# ═══════════════════════════════════════════════════════════
print("Step 2: Consistent facies classification...")
ni, nx, nz = v_model.shape
z = np.linspace(0, 3000, nz)
v_cmp = 1500.0 + 0.8 * z.reshape(1, 1, nz)
delta_v = v_model - v_cmp
# Smooth only in depth (matching per-column gaussian_filter1d sigma=3)
delta_v_s = gaussian_filter(delta_v.astype(np.float64), sigma=(0, 0, 3))

facies_3d = np.zeros(v_model.shape, dtype=np.int64)  # default: shale=0
facies_3d[delta_v_s < -150] = 2                         # sand channel
facies_3d[(delta_v_s > 200) & (z.reshape(1, 1, nz) > 800)] = 3  # carbonate
facies_3d[(delta_v_s >= -150) & (delta_v_s <= 200) & (z.reshape(1, 1, nz) > 200)] = 1  # silt

print(f"  Facies: classes={np.unique(facies_3d)}, "
      f"shale={(facies_3d==0).mean():.1%}, silt={(facies_3d==1).mean():.1%}, "
      f"sand={(facies_3d==2).mean():.1%}, carb={(facies_3d==3).mean():.1%}")

# ═══════════════════════════════════════════════════════════
# 3. Fracture: fault damage zones + physically-meaningful additions
# ═══════════════════════════════════════════════════════════
print("Step 3: Fracture labels...")
fault_bin = fault_mask_3d > 0.2
# Damage zone around faults (wider than slip surface)
struct = np.ones((3, 3, 3)); struct[0] = 0; struct[2] = 0
damage_zone = binary_dilation(fault_bin, structure=struct, iterations=2).astype(np.float32)

# Add high-curvature zones (fold hinges, channel edges)
curv_v = gaussian_filter(v_model.astype(np.float64), sigma=(0.5, 0.5, 0.5))
curv_mag = np.abs(sobel(curv_v, axis=1)) + np.abs(sobel(curv_v, axis=0))
curv_mag_n = curv_mag / (curv_mag.std() + 1e-10)
high_curv = (curv_mag_n > 3.5).astype(np.float32)

fracture_3d = np.maximum(damage_zone * 0.7, high_curv * 0.3)
fracture_3d = gaussian_filter(fracture_3d, sigma=(0.3, 0.3, 0.3))
fracture_3d[fracture_3d < 0.12] = 0.0
print(f"  Fracture: non-zero={fracture_3d.mean():.4%}")

# ═══════════════════════════════════════════════════════════
# 4. Wells with fault-zone and fracture-zone signatures
# ═══════════════════════════════════════════════════════════
print("Step 4: Wells with fault/fracture signatures...")
sc = SeismicConfig(); sc.n_wells = 500; sc.seed = 42
sc.n_inline = 300; sc.n_xline = 400
wc = WellLogConfig()
syn_gen = SyntheticDataGenerator(sc, wc)
syn_gen.rng = np.random.RandomState(seed=42)
well_locs = syn_gen.generate_well_locations()

# Generate wells with standard rock physics (already has facies consistency)
wells = syn_gen.generate_well_logs(well_locs, v_model)

# ── Add fault-zone and fracture-zone signatures to well logs ──
for w_idx, well in enumerate(wells):
    ix = int(np.clip(well_locs[w_idx, 0], 0, ni - 1))
    iy = int(np.clip(well_locs[w_idx, 1], 0, nx - 1))

    # Check if well passes through fault zone
    fault_at_well = fault_mask_3d[ix, iy, :] > 0.3
    fracture_at_well = fracture_3d[ix, iy, :] > 0.15

    if fault_at_well.sum() > 3:
        # Fault signature: DT spikes, density drops, RT drops
        fault_idx = np.where(fault_at_well)[0]
        for idx in fault_idx:
            lo = max(0, idx - 2); hi = min(nz, idx + 3)
            well["DT"][lo:hi] *= 1.12   # 12% slower (fracturing)
            well["RHOB"][lo:hi] *= 0.93  # 7% density drop
            well["RT"][lo:hi] *= 0.35   # mud filtrate invasion

    if fracture_at_well.sum() > 5:
        # Fracture signature: subtle DT anomalies + RT drops
        frac_idx = np.where(fracture_at_well)[0]
        for idx in frac_idx[::2]:  # sparse: every other fracture point
            well["DT"][idx] *= 1.05
            well["RT"][idx] *= 0.6

checkshots = syn_gen.generate_checkshots(wells)

# ═══════════════════════════════════════════════════════════
# 5. Pipeline: extraction, RBF, time-depth, resample
# ═══════════════════════════════════════════════════════════
print("Step 5: Pipeline extraction...")
ex = GridExtractor(sc)
seismic_rect, wl_shifted = ex.extract(seismic_3d, well_locs, wells)
mx0 = max(0, int(np.floor(well_locs[:, 0].min())) - 5)
mx1 = min(ni, int(np.ceil(well_locs[:, 0].max())) + 5)
my0 = max(0, int(np.floor(well_locs[:, 1].min())) - 5)
my1 = min(nx, int(np.ceil(well_locs[:, 1].max())) + 5)
fault_rect = fault_mask_3d[mx0:mx1, my0:my1, :]
horizon_rect = horizon_mask_3d[mx0:mx1, my0:my1, :]
facies_rect = facies_3d[mx0:mx1, my0:my1, :]
fracture_rect = fracture_3d[mx0:mx1, my0:my1, :]

rbf = RBFVelocityBuilder(sc)
v_avg = rbf.build(checkshots, seismic_rect.shape)
tdc = TimeDepthConverter(sc)
seismic_depth, depth_axis = tdc.convert(seismic_rect, v_avg)
q2d = Quasi2DExtractor(sc)
sections = q2d.extract(seismic_depth, wl_shifted, wells)
rs = FFTResampler(sc)
for sec in sections:
    sec["section"] = rs.resample(sec["section"])
asm = DatasetAssembler(sc, wc)
wells_h = asm.harmonize_wells(wells, 256, (depth_axis[0], depth_axis[-1]))

# ═══════════════════════════════════════════════════════════
# 6. Extract 2D labels + save
# ═══════════════════════════════════════════════════════════
print("Step 6: Saving...")
H, W = 256, 512

def extract_2d(d3d, stype, ix, iy, is_class=False):
    s = d3d[ix % d3d.shape[0], :, :].T if stype == "inline" else d3d[:, iy % d3d.shape[1], :].T
    return zoom(s, (H/s.shape[0], W/s.shape[1]), order=0 if is_class else 1)[:H, :W]

output = "./openseisml_consistent.h5"
with h5py.File(output, 'w') as f:
    sg, wg, lg = f.create_group("seismic"), f.create_group("wells"), f.create_group("labels")
    for i, sec in enumerate(sections):
        ds = sg.create_dataset(f"section_{i:05d}", data=sec["section"], compression="gzip", compression_opts=4)
        ds.attrs["well_id"] = f"WELL_{i+1:05d}"
        ds.attrs["well_x"] = sec["well_index"]; ds.attrs["section_type"] = sec.get("section_type", "inline")

        wgrp = wg.create_group(f"WELL_{i+1:05d}")
        for k, v in wells_h[i].items():
            if isinstance(v, np.ndarray): wgrp.create_dataset(k, data=v.astype(np.float32))

        st = sec.get("section_type", "inline")
        ix_s = min(int(sec.get("inline", i)), fault_rect.shape[0] - 1)
        iy_s = min(int(sec.get("xline", i)), fault_rect.shape[1] - 1)

        grp = lg.create_group(f"section_{i:05d}")
        grp.create_dataset("fault", data=extract_2d(fault_rect, st, ix_s, iy_s))
        grp.create_dataset("horizon", data=extract_2d(horizon_rect, st, ix_s, iy_s, is_class=True))
        grp.create_dataset("facies", data=extract_2d(facies_rect, st, ix_s, iy_s, is_class=True))
        grp.create_dataset("fracture", data=extract_2d(fracture_rect, st, ix_s, iy_s))
        grp.attrs["well_x"] = sec["well_index"]
        if (i+1) % 100 == 0: print(f"  [{i+1}/500]")

    meta = f.create_group("metadata")
    for k, v in {
        "n_sections": 500, "n_wells": 500, "n_horizon_classes": 13, "n_facies_classes": 5,
        "label_type": "geological_segmentation",
        "physics": "facies=column_classification(consistent_with_wells), "
                   "fault=explicit_planes, fracture=damage_zones, "
                   "wells_have_fault_fracture_signatures"
    }.items():
        meta.attrs[k] = v

# ═══════════════════════════════════════════════════════════
# 7. Verify consistency
# ═══════════════════════════════════════════════════════════
print("\nStep 7: Verifying label consistency...")
f2 = h5py.File(output, 'r')
consistent = 0
for i in range(min(50, 500)):
    sec = f2[f'seismic/section_{i:05d}']
    wx = sec.attrs["well_x"]
    facies_2d = f2[f'labels/section_{i:05d}/facies'][:]
    well_gr = f2[f'wells/WELL_{i+1:05d}/GR'][:]

    # Check: at well column, does facies align with GR?
    # High GR (>50 API) → shale/silt (facies 0 or 1)
    # Low GR (<30 API) → sand (facies 2)
    facies_at_well = facies_2d[:, wx]
    high_gr_mask = well_gr > 50
    low_gr_mask = well_gr < 30

    # High GR should be shale/silt (0 or 1), not sand (2)
    high_gr_facies_ok = (facies_at_well[high_gr_mask] != 2).mean()
    # Low GR should be sand (2), not shale (0)
    low_gr_facies_ok = (facies_at_well[low_gr_mask] == 2).mean()

    if not np.isnan(high_gr_facies_ok) and not np.isnan(low_gr_facies_ok):
        consistent += (high_gr_facies_ok + low_gr_facies_ok) / 2

print(f"  Facies-GR consistency: {consistent/max(1,min(50,500)):.2%}")
print(f"  (1.0 = perfect, facies label matches well log GR at well position)")

f2.close()
print(f"\n✓ Done in {time.time()-t0:.0f}s → {output}")
