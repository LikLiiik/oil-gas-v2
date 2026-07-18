#!/usr/bin/env python3
"""
OpenSeisML Data Curation Pipeline
==================================
Implements the seismic + well-log joint dataset construction pipeline
as described in:
  "OpenSeisML: Open Large-Scale Real Seismic and well-log Dataset for Generative AI"
  Bhar et al., arXiv:2605.20539v1, 2026

Pipeline steps:
  1. Data generation (synthetic SEG-Y-like seismic + LAS well logs + checkshots)
  2. CRS alignment & regular grid extraction
  3. Checkshot-based RBF velocity volume construction
  4. Time-to-depth conversion
  5. Quasi-2D line extraction through wells
  6. FFT resampling to 256×512 with cosine taper low-pass filter
  7. Well log harmonization & HDF5 storage
"""

import numpy as np
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom
from scipy import signal
import h5py
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SeismicConfig:
    """Configuration matching the OpenSeisML paper specifications."""
    # 3D volume dimensions (time/depth domain)
    n_inline: int = 150        # number of inlines
    n_xline: int = 200         # number of crosslines
    n_samples: int = 512       # time/depth samples

    # Grid spacing
    inline_spacing: float = 12.5   # meters
    xline_spacing: float = 12.5    # meters
    time_sample_rate: float = 0.004  # seconds (4ms)

    # Output 2D section dimensions (paper: 256×512)
    output_height: int = 256
    output_width: int = 512

    # Number of wells (paper used 40 initially, targeting 1000+)
    n_wells: int = 40

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class WellLogConfig:
    """Well log types as specified in the paper (Table in Figure 1)."""
    log_types: List[str] = field(default_factory=lambda: [
        "GR",    # Gamma Ray (API unit)
        "NPHI",  # Neutron Porosity (v/v)
        "RHOB",  # Bulk Density (g/cm³)
        "DT",    # Sonic travel time (μs/ft)
        "RT",    # Resistivity (ohm·m)
    ])
    log_units: Dict[str, str] = field(default_factory=lambda: {
        "GR": "API",
        "NPHI": "v/v",
        "RHOB": "g/cm3",
        "DT": "us/ft",
        "RT": "ohm.m",
    })
    well_sample_rate: float = 0.1524  # meters (~0.5 ft)


# ============================================================================
# Step 1: Synthetic Data Generation
# ============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic-but-realistic 3D seismic data, well logs, and
    checkshot measurements mimicking real North Sea marine geology.

    Stand-in for UK NDR data. The pipeline logic is identical whether
    data is synthetic or real.
    """

    def __init__(self, sc: SeismicConfig, wc: WellLogConfig):
        self.sc = sc
        self.wc = wc
        self.rng = np.random.RandomState(sc.seed)

    def _generate_velocity_model_3d(self) -> np.ndarray:
        """Generate 3D interval velocity model with realistic geology."""
        ni, nx, nz = self.sc.n_inline, self.sc.n_xline, self.sc.n_samples
        z = np.linspace(0, 3000, nz)

        # --- Compaction trend: v(z) = v0 + k*z ---
        v0, k = 1500.0, 0.8
        v_model = np.tile((v0 + k * z).reshape(1, 1, nz), (ni, nx, 1))

        # --- Layered structure ---
        n_layers = 12
        for i in range(n_layers):
            z_center = (i + 1) * 3000 / n_layers
            z_idx = int(z_center / 3000 * (nz - 1))
            dv = self.rng.uniform(-250, 350)
            sigma_z = self.rng.uniform(3, 8)
            for iz in range(nz):
                dist = (iz - z_idx) ** 2 / (2 * sigma_z ** 2)
                if dist < 5:
                    v_model[:, :, iz] += dv * np.exp(-dist)

        # --- Channel structures ---
        for _ in range(5):
            cx = self.rng.randint(10, ni - 10)
            amp = self.rng.choice([-350, 350])
            w = self.rng.uniform(2, 8)
            for iz in range(nz // 3, 2 * nz // 3):
                cy = nx // 2 + int(15 * np.sin(iz * 0.05 + self.rng.uniform(0, np.pi)))
                for di in range(-int(2 * w), int(2 * w) + 1):
                    for dj in range(-int(2 * w), int(2 * w) + 1):
                        ii, jj = cx + di, cy + dj
                        if 0 <= ii < ni and 0 <= jj < nx:
                            v_model[ii, jj, iz] += amp * np.exp(-(di**2 + dj**2) / w**2)

        # --- Small-scale heterogeneity ---
        v_model += self.rng.normal(0, 25, (ni, nx, nz))
        v_model = gaussian_filter(v_model, sigma=(1.0, 1.0, 0.5))
        return v_model

    def _velocity_to_reflectivity(self, v_model: np.ndarray) -> np.ndarray:
        """Gardner → Acoustic Impedance → Normal-incidence reflectivity."""
        rho = 0.31 * (v_model ** 0.25)
        ai = v_model * rho
        refl = np.diff(ai, axis=2) / (ai[:, :, :-1] + ai[:, :, 1:] + 1e-10)
        return np.pad(refl, ((0, 0), (0, 0), (0, 1)), mode='constant')

    def _convolve_with_wavelet(self, refl: np.ndarray) -> np.ndarray:
        """Ricker wavelet convolution → post-stack seismic."""
        f_peak, dt = 30.0, self.sc.time_sample_rate
        t = np.arange(-0.1, 0.1 + dt, dt)
        wvlt = (1 - 2 * (np.pi * f_peak * t)**2) * np.exp(-(np.pi * f_peak * t)**2)
        ni, nx, nz = refl.shape
        seismic = np.empty_like(refl)
        for i in range(ni):
            for j in range(nx):
                seismic[i, j, :] = np.convolve(refl[i, j, :], wvlt, mode='same')
        seismic += 0.02 * np.std(seismic) * self.rng.normal(0, 1, seismic.shape)
        return seismic.astype(np.float32)

    def generate_seismic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (seismic_time, v_model_true)."""
        print("[Step 1a] Generating 3D velocity model ...")
        v_model = self._generate_velocity_model_3d()
        print("[Step 1b] Computing reflectivity ...")
        refl = self._velocity_to_reflectivity(v_model)
        print("[Step 1c] Convolving with Ricker wavelet ...")
        seismic = self._convolve_with_wavelet(refl)
        return seismic, v_model

    def generate_well_locations(self) -> np.ndarray:
        """Random well positions within survey boundary."""
        m = 15
        wx = self.rng.uniform(m, self.sc.n_inline - m, self.sc.n_wells)
        wy = self.rng.uniform(m, self.sc.n_xline - m, self.sc.n_wells)
        return np.column_stack([wx, wy])

    def generate_well_logs(self, well_locs: np.ndarray,
                           v_model: np.ndarray) -> List[Dict]:
        """Generate petrophysical logs + velocity at each well."""
        nz = self.sc.n_samples
        depth = np.linspace(0, 3000, nz)
        wells = []
        for w in range(self.sc.n_wells):
            ix = int(np.clip(well_locs[w, 0], 0, self.sc.n_inline - 1))
            iy = int(np.clip(well_locs[w, 1], 0, self.sc.n_xline - 1))
            v_well = v_model[ix, iy, :]
            base = gaussian_filter1d(self.rng.normal(0, 1, nz), sigma=3)
            v_norm = (v_well - np.mean(v_well)) / (np.std(v_well) + 1e-10)
            gr = np.clip(45 + 25 * v_norm + 8 * base, 0, 200)
            nphi = np.clip(0.22 - 0.08 * v_norm + 0.03 * base, 0.0, 0.45)
            rhob = np.clip(2.35 + 0.15 * v_norm + 0.03 * base, 1.8, 2.9)
            dt_us_ft = 1e6 / np.clip(v_well, 1500, 6000) * 0.3048
            dt = dt_us_ft + 2 * self.rng.normal(0, 1, nz)
            rt = np.clip(20 * np.exp(-depth / 1200) + 5 * base, 0.5, 100)
            wells.append(dict(well_id=f"WELL_{w+1:04d}",
                              x_inline=well_locs[w, 0], y_xline=well_locs[w, 1],
                              depth=depth, GR=gr, NPHI=nphi, RHOB=rhob, DT=dt, RT=rt,
                              velocity=v_well))
        return wells

    def generate_checkshots(self, wells: List[Dict]) -> List[Dict]:
        """Sparse checkshot measurements (depth↔TWT pairs)."""
        nz = self.sc.n_samples
        depth = np.linspace(0, 3000, nz)
        dz = depth[1] - depth[0]
        step = max(1, int(100 / dz))  # ~every 100 m
        css = []
        for w, well in enumerate(wells):
            v = np.clip(well["velocity"], 100, 10000)
            twt = 2 * np.cumsum(dz / v)
            idx = np.arange(0, nz, step)
            css.append(dict(well_id=well["well_id"],
                            x_inline=well["x_inline"], y_xline=well["y_xline"],
                            depth=depth[idx], twt=twt[idx], velocity=v[idx]))
        return css


# ============================================================================
# Step 2: CRS Alignment & Regular Grid Extraction
# ============================================================================

class GridExtractor:
    """
    Extract largest contiguous rectangular region from survey boundary.
    Paper: "estimate survey boundaries using a concave hull ...
           extract the largest contiguous rectangular region with regular grids"
    """
    def __init__(self, sc: SeismicConfig):
        self.sc = sc

    def extract(self, seismic: np.ndarray, well_locs: np.ndarray,
                wells: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        ni, nx, nz = seismic.shape
        mx0 = max(0, int(np.floor(well_locs[:, 0].min())) - 5)
        mx1 = min(ni, int(np.ceil(well_locs[:, 0].max())) + 5)
        my0 = max(0, int(np.floor(well_locs[:, 1].min())) - 5)
        my1 = min(nx, int(np.ceil(well_locs[:, 1].max())) + 5)
        seismic_rect = seismic[mx0:mx1, my0:my1, :]
        wl_shifted = well_locs.copy()
        wl_shifted[:, 0] -= mx0
        wl_shifted[:, 1] -= my0
        print(f"  Extracted: {seismic_rect.shape}  |  wells inside: {len(wl_shifted)}")
        return seismic_rect, wl_shifted


# ============================================================================
# Step 3: RBF Velocity Volume Construction  (paper Eq. 1)
# ============================================================================

class RBFVelocityBuilder:
    """
    Construct 3-D average velocity volume from checkshot data via RBF.

    Paper Eq. (1):  f(x) = Σ λᵢ φ(‖x − xᵢ‖)

    Strategy (performance):
      - RBF on a *coarse* 3-D grid  (every 4th sample)
      - 3-D linear interpolation → full-resolution volume
    """
    def __init__(self, sc: SeismicConfig):
        self.sc = sc
        self.coarsen = 4  # coarse-grid stride

    def build(self, checkshots: List[Dict],
              seismic_shape: Tuple[int, int, int]) -> np.ndarray:
        ni, nx, nz = seismic_shape
        # collect all (x, y, z, v) points from checkshots
        pts, vals = [], []
        for cs in checkshots:
            x, y = cs["x_inline"], cs["y_xline"]
            for z, v in zip(cs["depth"], cs["velocity"]):
                pts.append([x, y, z])
                vals.append(v)
        pts_arr = np.array(pts)
        vals_arr = np.array(vals)
        print(f"  Checkshot points: {len(pts_arr)}")

        # --- RBF on coarse grid ---
        xi = np.arange(0, ni, self.coarsen, dtype=np.float64)
        yi = np.arange(0, nx, self.coarsen, dtype=np.float64)
        zi = np.linspace(0, 3000, nz // self.coarsen, dtype=np.float64)
        ci, cx, cz = len(xi), len(yi), len(zi)
        XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
        grid = np.column_stack([XI.ravel(), YI.ravel(), ZI.ravel()])

        print(f"  RBF on coarse grid {ci}×{cx}×{cz} = {len(grid):,} points ...", end='', flush=True)
        t0 = time.time()
        rbf = RBFInterpolator(pts_arr, vals_arr, kernel='multiquadric', epsilon=1.0)
        v_coarse = rbf(grid).reshape(ci, cx, cz)
        print(f"  done in {time.time() - t0:.1f}s")

        # --- Upsample to full resolution ---
        print("  Upsampling to full resolution ...", end='', flush=True)
        t0 = time.time()
        v_full = zoom(v_coarse,
                      (ni / ci, nx / cx, nz / cz),
                      order=1)  # bilinear
        v_full = np.clip(v_full, 1500.0, 6000.0).astype(np.float32)
        print(f"  done in {time.time() - t0:.1f}s")
        print(f"  Velocity range: [{v_full.min():.0f}, {v_full.max():.0f}] m/s")
        return v_full


# ============================================================================
# Step 4: Time-to-Depth Conversion
# ============================================================================

class TimeDepthConverter:
    """
    Convert time-migrated seismic → depth domain.

    Paper:  z(t) = ∫₀ᵗ v(τ)/2 dτ
            zₖ = Σⱼ (vⱼ / 2) × Δt
    """
    def __init__(self, sc: SeismicConfig):
        self.sc = sc

    def convert(self, seismic_time: np.ndarray,
                v_avg_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ni, nx, nt = seismic_time.shape
        dt = self.sc.time_sample_rate
        # cumulative depth at each time sample
        depth_cum = np.cumsum(v_avg_3d * dt / 2.0, axis=2)
        z_max = depth_cum[:, :, -1].max()
        nz_out = self.sc.n_samples
        z_target = np.linspace(0, z_max, nz_out)
        seismic_depth = np.zeros((ni, nx, nz_out), dtype=np.float32)
        for i in range(ni):
            for j in range(nx):
                seismic_depth[i, j, :] = np.interp(z_target, depth_cum[i, j, :],
                                                    seismic_time[i, j, :],
                                                    left=0, right=0)
        print(f"  Time → Depth:  {seismic_time.shape}  →  {seismic_depth.shape}")
        print(f"  Depth range: [0, {z_max:.0f}] m")
        return seismic_depth, z_target


# ============================================================================
# Step 5: Quasi-2D Line Extraction Through Wells
# ============================================================================

class Quasi2DExtractor:
    """
    Extract 2D sections along trajectories passing through wells.
    Paper Fig. 5: quasi-2D lines through well locations.
    """
    def __init__(self, sc: SeismicConfig):
        self.sc = sc

    def extract(self, seismic_depth: np.ndarray,
                well_locs: np.ndarray,
                wells: List[Dict]) -> List[Dict]:
        ni, nx, nz = seismic_depth.shape
        sections = []
        for w, (loc, well) in enumerate(zip(well_locs, wells)):
            ix = int(np.clip(np.round(loc[0]), 0, ni - 1))
            iy = int(np.clip(np.round(loc[1]), 0, nx - 1))
            if w % 2 == 0:
                sec = seismic_depth[ix, :, :].T.astype(np.float32)  # (nz, nx)
                stype, wpos = "inline", iy
            else:
                sec = seismic_depth[:, iy, :].T.astype(np.float32)  # (nz, ni)
                stype, wpos = "xline", ix
            sections.append(dict(well_id=well["well_id"], section_type=stype,
                                 section=sec, well_index=wpos,
                                 well_x=loc[0], well_y=loc[1],
                                 inline=ix, xline=iy))
        print(f"  Extracted {len(sections)} quasi-2D sections")
        return sections


# ============================================================================
# Step 6: FFT Resampling to 256×512 with Cosine Taper
# ============================================================================

class FFTResampler:
    """
    Resample 2-D sections to 256×512 via 2-D FFT + cosine-taper low-pass.

    Paper: "resampled to 256×512 using a 2D FFT, where a smooth low-pass
    filter with a cosine taper suppresses high-frequency components"
    """
    def __init__(self, sc: SeismicConfig):
        self.sc = sc
        self.H = sc.output_height  # 256
        self.W = sc.output_width   # 512

    def _cosine_taper(self, h: int, w: int, cutoff: float = 0.75) -> np.ndarray:
        fy = np.abs(np.fft.fftfreq(h)).reshape(-1, 1)
        fx = np.abs(np.fft.fftfreq(w)).reshape(1, -1)
        f_mag = np.sqrt(fx**2 + fy**2)
        f_norm = f_mag / np.max(f_mag) if np.max(f_mag) > 0 else f_mag
        taper = np.ones((h, w), dtype=np.float32)
        nyq = 0.5
        f_cut = cutoff * nyq
        mask = (f_norm > f_cut) & (f_norm < nyq)
        taper[mask] = 0.5 * (1 + np.cos(np.pi * (f_norm[mask] - f_cut) / (nyq - f_cut)))
        taper[f_norm >= nyq] = 0.0
        return taper

    def resample(self, section: np.ndarray) -> np.ndarray:
        H_in, W_in = section.shape
        # Forward FFT
        F = np.fft.fft2(section)
        # Resize FFT to target size (keep low freqs centred)
        F_out = np.zeros((self.H, self.W), dtype=complex)
        hc = min(H_in, self.H) // 2
        wc = min(W_in, self.W) // 2
        F_out[:hc, :wc] = F[:hc, :wc]
        F_out[:hc, -wc:] = F[:hc, -wc:]
        F_out[-hc:, :wc] = F[-hc:, :wc]
        F_out[-hc:, -wc:] = F[-hc:, -wc:]
        # Cosine taper low-pass
        filt = self._cosine_taper(self.H, self.W, 0.75)
        F_out *= filt
        # Inverse FFT
        out = np.fft.ifft2(F_out).real
        out *= (self.H * self.W) / (H_in * W_in)
        return out.astype(np.float32)


# ============================================================================
# Step 7: Well Log Harmonization & HDF5 Storage
# ============================================================================

class DatasetAssembler:
    """
    Harmonise well-log resolution to seismic, save entire dataset as HDF5.

    Paper: "Since well logs typically have finer vertical sampling than
    seismic traces, filtering and resampling are applied to wells to
    harmonize resolutions ... All curated datasets are stored in HDF5 format."
    """
    def __init__(self, sc: SeismicConfig, wc: WellLogConfig):
        self.sc = sc
        self.wc = wc

    def harmonize_wells(self, wells: List[Dict],
                        target_nz: int,
                        depth_range: Tuple[float, float]) -> List[Dict]:
        z_new = np.linspace(depth_range[0], depth_range[1], target_nz)
        hw = []
        for well in wells:
            wd = dict(well_id=well["well_id"],
                      x_inline=well["x_inline"], y_xline=well["y_xline"],
                      depth=z_new)
            for key in self.wc.log_types + ["velocity"]:
                if key in well:
                    wd[key] = np.interp(z_new, well["depth"],
                                        gaussian_filter1d(well[key], sigma=2.0))
            hw.append(wd)
        return hw

    def save(self, sections: List[Dict], wells_h: List[Dict],
             out_path: str, v_avg_3d=None, depth_axis=None):
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
        with h5py.File(out_path, 'w') as f:
            # --- seismic ---
            sg = f.create_group("seismic")
            for i, sec in enumerate(sections):
                ds = sg.create_dataset(f"section_{i:04d}", data=sec["section"],
                                       compression="gzip", compression_opts=4)
                ds.attrs["well_id"] = sec["well_id"]
                ds.attrs["section_type"] = sec["section_type"]
                ds.attrs["well_index"] = sec["well_index"]
                ds.attrs["inline"] = sec["inline"]
                ds.attrs["xline"] = sec["xline"]
                ds.attrs["sampling_interval_m"] = self.sc.inline_spacing
            # --- wells ---
            wg = f.create_group("wells")
            for well in wells_h:
                wgrp = wg.create_group(well["well_id"])
                wgrp.create_dataset("depth", data=well["depth"].astype(np.float32))
                for lt in self.wc.log_types + ["velocity"]:
                    if lt in well:
                        d = wgrp.create_dataset(lt, data=well[lt].astype(np.float32))
                        d.attrs["unit"] = self.wc.log_units.get(lt, "m/s" if lt == "velocity" else "")
                wgrp.attrs["x_inline"] = well["x_inline"]
                wgrp.attrs["y_xline"] = well["y_xline"]
            # --- 3D velocity volume ---
            if v_avg_3d is not None:
                f.create_dataset("velocity_volume_3d", data=v_avg_3d,
                                 compression="gzip", compression_opts=4)
            if depth_axis is not None:
                f.create_dataset("depth_axis", data=depth_axis.astype(np.float32))
            # --- metadata ---
            meta = f.create_group("metadata")
            meta.attrs["description"] = ("OpenSeisML-style curated seismic + well-log dataset. "
                                         "Bhar et al., arXiv:2605.20539v1 (2026).")
            meta.attrs["n_sections"] = len(sections)
            meta.attrs["n_wells"] = len(wells_h)
            meta.attrs["section_shape"] = f"{self.sc.output_height}x{self.sc.output_width}"
            meta.attrs["sampling_interval_m"] = self.sc.inline_spacing
            meta.attrs["log_types"] = ", ".join(self.wc.log_types)
        print(f"\n✓ Saved to: {out_path}")
        print(f"  Sections: {len(sections)}  |  Wells: {len(wells_h)}")
        print(f"  Shape: {self.sc.output_height}×{self.sc.output_width}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(output_path: str = "./openseisml_dataset.h5"):
    """
    Execute the complete OpenSeisML data curation pipeline.

    Follows Figure 1 flow diagram exactly:
      1. Data Collection (UK NDR → here: synthetic)
      2. Regular grid check → survey boundary extraction
      3. Checkshot velocity via RBF → time-depth conversion
      4. Training data ← quasi-2D lines through wells
    """
    print("=" * 65)
    print("  OpenSeisML Data Curation Pipeline")
    print("  Bhar et al., arXiv:2605.20539v1 (2026)")
    print("=" * 65)

    sc = SeismicConfig()
    wc = WellLogConfig()

    # ── Step 1: Data Collection ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 1  |  Data Collection")
    print("─" * 50)
    gen = SyntheticDataGenerator(sc, wc)
    seismic_time, v_model_true = gen.generate_seismic()
    well_locs = gen.generate_well_locations()
    wells = gen.generate_well_logs(well_locs, v_model_true)
    checkshots = gen.generate_checkshots(wells)
    print(f"  Seismic: {seismic_time.shape}  |  Wells: {len(wells)}")
    print(f"  Log types: {wc.log_types}")

    # ── Step 2: Preprocessing ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 2  |  CRS + Regular Grid Extraction")
    print("─" * 50)
    ex = GridExtractor(sc)
    seismic_rect, well_locs_shifted = ex.extract(seismic_time, well_locs, wells)

    # ── Step 3: RBF Velocity Volume ───────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 3  |  RBF Average Velocity Volume (Eq.1)")
    print("─" * 50)
    rbf_builder = RBFVelocityBuilder(sc)
    v_avg_3d = rbf_builder.build(checkshots, seismic_rect.shape)

    # QC: checkshot vs interpolated at well 0
    from scipy.interpolate import interp1d
    z_test = np.linspace(0, 3000, sc.n_samples)
    idx = (int(well_locs[0, 0]), int(well_locs[0, 1]))
    v_interp = v_avg_3d[min(idx[0], v_avg_3d.shape[0] - 1),
                         min(idx[1], v_avg_3d.shape[1] - 1), :]
    cs = checkshots[0]
    v_cs = interp1d(cs["depth"], cs["velocity"], kind='linear',
                    fill_value='extrapolate')(z_test)
    mae = np.mean(np.abs(v_interp - v_cs))
    print(f"  [QC] {wells[0]['well_id']} checkshot vs interp MAE: {mae:.1f} m/s")

    # ── Step 4: Time→Depth Conversion ─────────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 4  |  Time-to-Depth Conversion")
    print("─" * 50)
    tdc = TimeDepthConverter(sc)
    seismic_depth, depth_axis = tdc.convert(seismic_rect, v_avg_3d)

    # ── Step 5: Quasi-2D Lines through Wells ──────────────────────────
    print("\n" + "─" * 50)
    print("STEP 5  |  Quasi-2D Line Extraction (Fig.5)")
    print("─" * 50)
    q2d = Quasi2DExtractor(sc)
    sections = q2d.extract(seismic_depth, well_locs_shifted, wells)

    # ── Step 6: FFT Resample to 256×512 ───────────────────────────────
    print("\n" + "─" * 50)
    print("STEP 6  |  FFT Resample 256×512 + Cosine Taper")
    print("─" * 50)
    rs = FFTResampler(sc)
    for sec in sections:
        sec["section"] = rs.resample(sec["section"])
    print(f"  Output shape: {sections[0]['section'].shape}  (256×512)")

    # ── Step 7: Well Log Harmonization & Save ─────────────────────────
    print("\n" + "─" * 50)
    print("STEP 7  |  Well-Log Harmonization & HDF5")
    print("─" * 50)
    asm = DatasetAssembler(sc, wc)
    wells_h = asm.harmonize_wells(wells, sc.output_height,
                                  (depth_axis[0], depth_axis[-1]))
    asm.save(sections, wells_h, output_path, v_avg_3d=v_avg_3d, depth_axis=depth_axis)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Output:    {output_path}")
    print(f"  Sections:  {len(sections)}  ×  {sc.output_height}×{sc.output_width}")
    print(f"  Sampling:  {sc.inline_spacing} m")
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OpenSeisML Data Curation Pipeline")
    p.add_argument("--output", "-o", default="./openseisml_dataset.h5")
    p.add_argument("--n-wells", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    SeismicConfig.n_wells = args.n_wells
    SeismicConfig.seed = args.seed
    run_pipeline(output_path=args.output)
