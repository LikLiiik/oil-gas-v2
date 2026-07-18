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

    def _generate_velocity_model_3d(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D interval velocity model with realistic geology.

        Returns:
            v_model:  (ni, nx, nz) velocity model
            fault_mask_3d: (ni, nx, nz) binary fault mask (from explicit fault planes)
        """
        ni, nx, nz = self.sc.n_inline, self.sc.n_xline, self.sc.n_samples
        z = np.linspace(0, 3000, nz)

        # --- Compaction trend: v(z) = v0 + k*z ---
        v0, k = 1500.0, 0.8
        v_model = np.tile((v0 + k * z).reshape(1, 1, nz), (ni, nx, 1))

        # --- Layered structure ---
        n_layers = 12
        layer_dv = np.zeros(n_layers)       # store dv per layer
        layer_z_center = np.zeros(n_layers)  # store depth center
        for i in range(n_layers):
            z_center = (i + 1) * 3000 / n_layers
            z_idx = int(z_center / 3000 * (nz - 1))
            dv = self.rng.uniform(-250, 350)
            sigma_z = self.rng.uniform(3, 8)
            layer_z_center[i] = z_center
            layer_dv[i] = dv
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

        # ============================================================
        # Explicit Fault Modeling (Wu et al., 2019; RGM, Gao et al., 2025)
        # ============================================================
        # Fault = displacement discontinuity in the velocity field.
        # Each fault has: strike, dip, throw (with spatial decay),
        # and a damage zone with reduced velocity.
        n_faults = self.rng.randint(3, 6)
        fault_mask_3d = np.zeros((ni, nx, nz), dtype=np.float32)

        for _ in range(n_faults):
            # Fault geometry
            dip = self.rng.uniform(55, 80)              # degrees from horizontal
            strike = self.rng.uniform(0, 180)            # degrees from xline axis
            throw_max = self.rng.uniform(20, 80)         # max vertical throw (m)
            throw_samples = int(throw_max / 3000 * nz)   # throw in depth samples

            # Fault center position (inline, xline at mid-depth)
            fx_c = self.rng.uniform(10, ni - 10)
            fy_c = self.rng.uniform(10, nx - 10)
            fz_c = self.rng.uniform(500, 2500)            # depth center (m)

            # Strike vector in horizontal plane
            strike_rad = np.radians(strike)
            strike_dx = np.sin(strike_rad)   # inline component
            strike_dy = np.cos(strike_rad)   # xline component

            # Dip: as depth increases, fault plane position shifts
            dip_slope = 1.0 / np.tan(np.radians(dip))  # lateral shift per depth

            for iz in range(nz):
                dz = z[iz]
                # Fault plane position at this depth
                # (moves laterally with depth due to dip)
                offset = (dz - fz_c) * dip_slope
                # Position along normal direction
                fx = int(fx_c + offset * strike_dy * 0.012)  # scaled to grid
                fy = int(fy_c + offset * strike_dx * 0.012)

                # Throw with spatial decay (max at center, zero at tips)
                depth_frac = abs(dz - fz_c) / max(3000 - fz_c, fz_c)
                decay = max(0.0, 1.0 - depth_frac)  # linear decay from center
                if decay < 0.1:
                    continue
                throw_z = int(throw_samples * decay)

                if throw_z < 2:
                    continue

                # Damage zone: reduced velocity around fault plane
                # Width 1-3 grid cells, 10-15% velocity reduction
                damage_half_width = self.rng.randint(1, 4)
                damage_reduction = self.rng.uniform(0.08, 0.18)

                for di in range(-damage_half_width, damage_half_width + 1):
                    for dj in range(-damage_half_width, damage_half_width + 1):
                        ix_sample = fx + di
                        iy_sample = fy + dj
                        if 0 <= ix_sample < ni and 0 <= iy_sample < nx:
                            dist = np.sqrt(di**2 + dj**2) / max(damage_half_width, 1)
                            reduction = damage_reduction * (1.0 - dist)
                            v_model[ix_sample, iy_sample, iz] *= (1.0 - reduction)
                            # Mark as fault zone
                            if dist < 0.5:
                                # Mark fault trace ±1 pixel width
                                for dt in range(-1, 2):
                                    iz2 = iz + dt
                                    if 0 <= iz2 < nz:
                                        fault_mask_3d[ix_sample, iy_sample, iz2] = 1.0

                # Apply throw displacement along fault normal direction
                # Velocity on the hanging-wall side is shifted DOWN
                for di in range(-6, 7):
                    for dj in range(-6, 7):
                        ix_sample = fx + di
                        iy_sample = fy + dj
                        if not (0 <= ix_sample < ni and 0 <= iy_sample < nx):
                            continue

                        # Determine which side of fault (signed distance from plane)
                        # Normal vector in (ix, iy) plane at this depth
                        normal_dist = di * strike_dy - dj * strike_dx
                        if abs(normal_dist) < 2:
                            continue  # skip fault plane itself

                        is_hanging = normal_dist > 0  # one side drops

                        if is_hanging:
                            # Hanging wall: shift velocity DOWN by throw
                            src_idx = iz - throw_z
                            if 0 <= src_idx < nz:
                                v_model[ix_sample, iy_sample, iz] = (
                                    0.7 * v_model[ix_sample, iy_sample, iz] +
                                    0.3 * v_model[ix_sample, iy_sample, src_idx]
                                )
                        else:
                            # Footwall: shift velocity UP by throw
                            src_idx = iz + throw_z
                            if 0 <= src_idx < nz:
                                v_model[ix_sample, iy_sample, iz] = (
                                    0.7 * v_model[ix_sample, iy_sample, iz] +
                                    0.3 * v_model[ix_sample, iy_sample, src_idx]
                                )

        # --- Small-scale heterogeneity (NOT fractures — just natural variation) ---
        v_model += self.rng.normal(0, 25, (ni, nx, nz))
        v_model = gaussian_filter(v_model, sigma=(1.0, 1.0, 0.5))

        # Smooth fault mask slightly (faults are thin but not single-pixel)
        fault_mask_3d = gaussian_filter(fault_mask_3d, sigma=(0.5, 0.5, 0.5))

        return v_model, fault_mask_3d

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

    def generate_seismic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (seismic_time, v_model_true, fault_mask_3d)."""
        print("[Step 1a] Generating 3D velocity model ...")
        v_model, fault_mask_3d = self._generate_velocity_model_3d()
        print("[Step 1b] Computing reflectivity ...")
        refl = self._velocity_to_reflectivity(v_model)
        print("[Step 1c] Convolving with Ricker wavelet ...")
        seismic = self._convolve_with_wavelet(refl)
        return seismic, v_model, fault_mask_3d

    def generate_well_locations(self) -> np.ndarray:
        """Random well positions within survey boundary."""
        m = 15
        wx = self.rng.uniform(m, self.sc.n_inline - m, self.sc.n_wells)
        wy = self.rng.uniform(m, self.sc.n_xline - m, self.sc.n_wells)
        return np.column_stack([wx, wy])

    def _classify_facies(self, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Classify facies from velocity + depth context.

        Uses deviation from the regional compaction trend as the primary
        discriminator, then applies depth-dependent rules.

        Returns:
            facies: 0=shale, 1=silt, 2=sand, 3=carbonate
        """
        nz = len(v)
        # Compaction trend: v_cmp(z) = 1500 + 0.8*z (regional baseline)
        v_cmp = 1500.0 + 0.8 * depth
        delta_v = v - v_cmp  # deviation from trend

        # Smooth delta to remove high-freq noise
        delta_v_s = gaussian_filter1d(delta_v.astype(np.float64), sigma=3)

        facies = np.zeros(nz, dtype=np.int64)  # default: shale
        # Sand: significant negative velocity anomaly (channels, unconsolidated)
        facies[delta_v_s < -150] = 2
        # Carbonate: significant positive anomaly (cemented, tight)
        facies[(delta_v_s > 200) & (depth > 800)] = 3
        # Silt: moderate deviations
        facies[(delta_v_s >= -150) & (delta_v_s <= 200) & (depth > 200)] = 1

        return facies

    def _facies_rock_physics(self, facies: int) -> dict:
        """
        Rock physics parameters per facies type.

        V_ma:  matrix P-wave velocity (m/s)
        rho_ma: matrix grain density (g/cm³)
        GR_clean: clean (no shale) gamma ray (API)
        GR_shale: pure shale gamma ray (API)
        a, m, n: Archie parameters
        """
        params = {
            0: dict(V_ma=3300, rho_ma=2.72, GR_clean=80, GR_shale=150,
                    a=1.0, m=2.3, n=2.0, name="shale"),
            1: dict(V_ma=4500, rho_ma=2.68, GR_clean=30, GR_shale=130,
                    a=1.0, m=2.1, n=2.0, name="silt"),
            2: dict(V_ma=5500, rho_ma=2.65, GR_clean=15, GR_shale=120,
                    a=1.0, m=2.0, n=2.0, name="sand"),
            3: dict(V_ma=6400, rho_ma=2.71, GR_clean=10, GR_shale=100,
                    a=1.0, m=2.2, n=2.0, name="carbonate"),
        }
        return params.get(facies, params[0])

    def generate_well_logs(self, well_locs: np.ndarray,
                           v_model: np.ndarray) -> List[Dict]:
        """
        Generate petrophysical logs using physically-grounded formulas.

        Physics-based curves:
          DT   — inverse velocity (exact physical definition)
          NPHI — Wyllie time-average equation: φ = (1/V-1/V_ma)/(1/V_f-1/V_ma)
          RHOB — volumetric mixing: ρ = ρ_ma×(1-φ) + ρ_f×φ

        Non-physics-based (explained below):
          GR   — shale volume proxy. Measures natural radioactivity (K,Th,U).
                 NO physical link to velocity. Modeled from facies + Vsh.
          RT   — fluid saturation proxy. Governed by Archie's law:
                 R_t = a·R_w/(φ^m·S_w^n). NO physical link to velocity.
        """
        nz = self.sc.n_samples
        depth = np.linspace(0, 3000, nz)
        V_f = 1500.0     # pore fluid velocity (m/s), water
        rho_f = 1.0      # pore fluid density (g/cm³)
        R_w = 0.1        # formation water resistivity (ohm·m)

        wells = []
        for w in range(self.sc.n_wells):
            ix = int(np.clip(well_locs[w, 0], 0, self.sc.n_inline - 1))
            iy = int(np.clip(well_locs[w, 1], 0, self.sc.n_xline - 1))
            v_well = v_model[ix, iy, :]

            # ── Facies classification ────────────────────────
            facies = self._classify_facies(v_well, depth)

            # ── Independent random fields ────────────────────
            # Shale volume (Vsh): lognormal around facies-typical value
            # Represents clay content — geologically smooth, depth-correlated
            vsh_seed = self.rng.normal(0, 1, nz)
            vsh_noise = gaussian_filter1d(vsh_seed, sigma=8)

            # Water saturation (Sw): random but facies-dependent
            sw_seed = self.rng.normal(0, 1, nz)
            sw_noise = gaussian_filter1d(sw_seed, sigma=4)

            # Measurement noise (independent per curve)
            noise_gr = self.rng.normal(0, 1, nz) * 3    # GR noise ~3 API
            noise_rt = self.rng.normal(0, 1, nz) * 0.05  # log10(RT) noise

            # ── Per-sample rock physics ───────────────────────
            gr = np.zeros(nz)
            nphi = np.zeros(nz)
            rhob = np.zeros(nz)
            dt = np.zeros(nz)
            rt = np.zeros(nz)

            for z in range(nz):
                rp = self._facies_rock_physics(facies[z])

                # --- Porosity: Wyllie time-average equation ---
                # 1/V = φ/V_f + (1-φ)/V_ma → solve for φ
                inv_V = 1.0 / max(v_well[z], 1500)
                inv_Vf = 1.0 / V_f
                inv_Vma = 1.0 / rp["V_ma"]
                phi_wyllie = (inv_V - inv_Vma) / (inv_Vf - inv_Vma + 1e-10)
                # Add facies-dependent scatter (real rocks deviate from Wyllie)
                phi = phi_wyllie + 0.02 * self.rng.normal(0, 1)
                phi = np.clip(phi, 0.01, 0.45)

                # --- Density: volumetric mixing equation ---
                # ρ_bulk = ρ_ma×(1-φ) + ρ_fluid×φ
                rho = rp["rho_ma"] * (1 - phi) + rho_f * phi
                rho += 0.015 * self.rng.normal(0, 1)  # measurement noise
                rho = np.clip(rho, 1.8, 3.0)

                # --- Sonic: exact inverse velocity ---
                # DT(μs/ft) = 1e6 / V(m/s) × 0.3048 (m/ft)
                dt_val = 1e6 / max(v_well[z], 1500) * 0.3048
                dt_val += self.rng.normal(0, 1.5)  # ~1-2 μs/ft noise

                # --- Gamma Ray: shale volume proxy ---
                # GR = GR_clean + (GR_shale - GR_clean) × Vsh
                vsh = 0.5 + 0.4 * vsh_noise[z]  # Vsh ~0.5±0.4
                vsh = np.clip(vsh, 0.02, 0.98)
                if facies[z] == 2:  # sand: low Vsh
                    vsh = np.clip(0.1 + 0.2 * vsh_noise[z], 0.02, 0.40)
                elif facies[z] == 3:  # carbonate: very low Vsh
                    vsh = np.clip(0.05 + 0.1 * vsh_noise[z], 0.01, 0.20)
                elif facies[z] == 0:  # shale: high Vsh
                    vsh = np.clip(0.6 + 0.3 * vsh_noise[z], 0.40, 0.98)

                gr_val = rp["GR_clean"] + (rp["GR_shale"] - rp["GR_clean"]) * vsh
                gr_val += noise_gr[z]

                # --- Resistivity: Archie's law ---
                # R_t = a × R_w / (φ^m × Sw^n)
                sw = 0.3 + 0.5 * (0.5 + 0.5 * sw_noise[z])  # Sw ~0.3-0.8
                sw = np.clip(sw, 0.05, 1.0)
                if facies[z] == 3:  # carbonate: lower Sw (often hydrocarbon-bearing)
                    sw = np.clip(0.15 + 0.4 * (0.5 + 0.5 * sw_noise[z]), 0.03, 0.6)

                phi_archie = max(phi, 0.01)
                rt_val = rp["a"] * R_w / (phi_archie ** rp["m"] * sw ** rp["n"])
                rt_val *= 10 ** noise_rt[z]  # log-normal multiplicative noise

                # Store
                gr[z] = np.clip(gr_val, 0, 200)
                nphi[z] = np.clip(phi, 0.0, 0.45)
                rhob[z] = np.clip(rho, 1.8, 3.0)
                dt[z] = np.clip(dt_val, 40, 200)
                rt[z] = np.clip(rt_val, 0.2, 1000)

            wells.append(dict(well_id=f"WELL_{w+1:04d}",
                              x_inline=well_locs[w, 0], y_xline=well_locs[w, 1],
                              depth=depth, GR=gr, NPHI=nphi, RHOB=rhob, DT=dt, RT=rt,
                              velocity=v_well,
                              facies=facies))
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
    seismic_time, v_model_true, fault_mask_3d = gen.generate_seismic()
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
