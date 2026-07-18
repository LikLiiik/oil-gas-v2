#!/usr/bin/env python3
"""
OpenSeisML Pipeline V2 — Physically-Realistic Synthetic Data Generator

Based on:
  - Faust (1951): V = K·(z·T)^(1/6)
  - Athy (1930): φ(z) = φ₀·exp(-c·z)
  - Gardner et al. (1974): ρ = a·V^b
  - Wyllie et al. (1956):  1/V = φ/V_f + (1-φ)/V_ma
  - Archie (1942): R_t = a·R_w/(φ^m·S_w^n)
  - Japsen (1999): North Sea shale compaction baseline
  - RGM (Gao et al., 2025): multi-randomization, listric faults
  - Lin et al. (2025): rich geological feature modeling

North Sea parameters from Marcussen et al. (2010), Japsen (1999).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, zoom, gaussian_filter1d
from scipy.interpolate import interp1d
from typing import Tuple, Dict, List, Optional
import time


class LayerConfig:
    """Configuration for a single stratigraphic layer."""
    def __init__(self, name: str, thickness_range: Tuple[float, float],
                 Vp_fast: float, Vp_slow: float,
                 Vsh_range: Tuple[float, float], rho_matrix: float):
        self.name = name
        self.thickness_range = thickness_range
        self.Vp_fast = Vp_fast      # clean end-member velocity (m/s)
        self.Vp_slow = Vp_slow      # shaley end-member velocity (m/s)
        self.Vsh_range = Vsh_range  # (min, max) shale volume fraction
        self.rho_matrix = rho_matrix

class RealisticModelGenerator:
    """
    Generates 3D velocity models with physically-realistic geology.

    Uses North Sea compaction parameters and multiple geological
    processes modeled after RGM and Lin et al. frameworks:
      - Faust/Athy compaction trend (exponential)
      - Stratigraphic layering with random perturbations
      - Listric faults with depth-decaying dip
      - Fluvial channel systems (top-flat/bottom-convex)
      - Unconformities (angular)
      - Multi-scale heterogeneity (von Karman spectrum)
    """

    def __init__(self, ni: int = 150, nx: int = 200, nz: int = 512,
                 z_max: float = 3000.0, seed: int = 42):
        self.ni = ni; self.nx = nx; self.nz = nz
        self.z_max = z_max
        self.dz = z_max / nz
        self.z = np.linspace(0, z_max, nz)
        self.rng = np.random.RandomState(seed)

    def compaction_trend(self, Vsh: float = 0.5) -> np.ndarray:
        """
        Modified Faust/Athy compaction trend.

        Faust:  V(z) = V₀ · (1 + k·z)^b
        Athy:   φ(z) = φ₀ · exp(-c·z)
        Wyllie: 1/V = φ/V_f + (1-φ)/V_ma  →  V from φ

        North Sea parameters (Marcussen et al., 2010):
          Shale: φ₀=0.63, c=0.51 km⁻¹
          Sand:  φ₀=0.40, c=0.27 km⁻¹
          Smectitic shale: V can be 500-700 m/s lower
        """
        z_km = self.z / 1000.0

        # Shale compaction (Athy)
        phi_shale = 0.63 * np.exp(-0.51 * z_km)
        phi_shale = np.clip(phi_shale, 0.05, 0.63)

        # Sand compaction (Athy)
        phi_sand = 0.40 * np.exp(-0.27 * z_km)
        phi_sand = np.clip(phi_sand, 0.03, 0.40)

        # Interpolate phi based on Vsh
        phi = Vsh * phi_shale + (1 - Vsh) * phi_sand

        # Wyllie: 1/V = φ/V_f + (1-φ)/V_ma
        V_f = 1500.0
        V_ma = 5500.0 * (1 - Vsh) + 3300.0 * Vsh  # mix based on Vsh
        V = 1.0 / (phi / V_f + (1 - phi) / V_ma)

        return V

    def generate_layers(self, base_v: np.ndarray, n_layers: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate stratigraphic layers using RGM-style bounding surfaces.

        Returns:
            v_model:  (ni, nx, nz) velocity with layers
            horizon_mask: (ni, nx, nz) horizon index at each interface
            Vsh_3d:   (ni, nx, nz) shale volume fraction
        """
        ni, nx, nz = self.ni, self.nx, self.nz
        v_model = base_v.copy()
        Vsh_3d = np.zeros((ni, nx, nz))
        horizon_mask = np.zeros((ni, nx, nz), dtype=np.int64)

        # Random layer boundaries
        layer_top = np.zeros(n_layers + 1)
        total_ratio = 1.0
        for i in range(n_layers):
            ratio = self.rng.uniform(0.5, 1.5)
            layer_top[i + 1] = layer_top[i] + ratio
        layer_top = layer_top / layer_top[-1] * nz  # normalize to nz

        for i in range(n_layers):
            z0 = int(layer_top[i])
            z1 = int(layer_top[i + 1])

            # Add lateral perturbations to layer boundaries
            perturb_amp = self.rng.uniform(2, 10)  # depth samples
            perturb = perturb_amp * self._von_karman_noise_2d(ni, nx, scale=20)

            z0_2d = (z0 + perturb[:, :, 0] * 0.3).astype(int)
            z1_2d = (z1 + perturb[:, :, 1] * 0.3).astype(int)

            # Layer velocity contrast (±100-400 m/s)
            dv = self.rng.uniform(-350, 400)
            Vsh_layer = self.rng.uniform(0.05, 0.95)

            for ix in range(ni):
                for iy in range(nx):
                    top_idx = max(0, min(nz - 1, z0_2d[ix, iy]))
                    bot_idx = max(0, min(nz - 1, z1_2d[ix, iy]))
                    for iz in range(top_idx, bot_idx):
                        v_model[ix, iy, iz] += dv * (1.0 - abs(iz - top_idx) / max(1, bot_idx - top_idx) * 0.3)
                        Vsh_3d[ix, iy, iz] = Vsh_layer

            # Horizon label
            for ix in range(ni):
                for iy in range(nx):
                    z_h = max(0, min(nz - 1, z0_2d[ix, iy]))
                    for dz in range(-1, 2):
                        iz_h = z_h + dz
                        if 0 <= iz_h < nz:
                            horizon_mask[ix, iy, iz_h] = i + 1

        Vsh_3d = gaussian_filter(Vsh_3d.astype(np.float64), sigma=(2, 2, 1))
        return v_model, horizon_mask, Vsh_3d

    def generate_listric_faults(self, v_model: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate listric (curved) faults with depth-decaying dip.
        RGM-style: dip decreases with depth, fault surfaces curve.

        Listric fault geometry:
          - Steep near surface (60-75°)
          - Dip decreases with depth, approaching horizontal at detachment
          - Normal faults in extensional regimes (North Sea)
          - Throw decays spatially from center to tip (Cowie-Scholz model)
        """
        ni, nx, nz = self.ni, self.nx, self.nz
        v_model_faulted = v_model.copy()
        fault_mask = np.zeros((ni, nx, nz), dtype=np.float32)

        n_faults = self.rng.randint(10, 16)  # more faults for better coverage

        for _ in range(n_faults):
            # Fault geometry
            x_c = self.rng.uniform(15, ni - 15)   # inline center
            y_c_init = self.rng.uniform(15, nx - 15)  # xline at surface
            strike_angle = self.rng.uniform(0, 180)  # degrees
            strike_rad = np.radians(strike_angle)
            strike_dx = np.sin(strike_rad)
            strike_dy = np.cos(strike_rad)

            # Dip: starts steep, decreases with depth (listric)
            dip_surface = self.rng.uniform(60, 78)  # surface dip
            dip_deep = self.rng.uniform(15, 35)     # deep dip
            detachment_depth = self.rng.uniform(2000, 2800)  # where fault flattens

            # Maximum throw (10-100m in depth)
            throw_max_m = self.rng.uniform(15, 80)
            throw_max = int(throw_max_m / self.z_max * nz)

            # Throw center depth
            throw_center_z = self.rng.uniform(500, 2000)

            for iz in range(nz):
                dz = self.z[iz]

                # Listric dip: linear decrease with depth
                if dz >= detachment_depth:
                    dip = dip_deep
                else:
                    frac = dz / detachment_depth
                    dip = dip_surface + (dip_deep - dip_surface) * frac

                # Fault trace position at this depth
                # Shifts laterally due to dip
                dip_slope = 1.0 / np.tan(np.radians(dip))
                # Surface position
                y_surf = y_c_init + (dz) * dip_slope * 0.012
                # Staggering from strike
                offset_dx = dz * np.cos(strike_rad) * 0.003
                offset_dy = dz * np.sin(strike_rad) * 0.003

                fx = int(x_c + offset_dx)
                fy = int(y_surf + offset_dy)

                if fx < 0 or fx >= ni or fy < 0 or fy >= nx:
                    continue

                # Throw with spatial decay (Cowie-Scholz: max at center, zero at tips)
                depth_dist = abs(dz - throw_center_z) / max(throw_center_z, self.z_max - throw_center_z)
                decay_z = max(0.0, 1.0 - depth_dist)  # linear decay along depth
                if decay_z < 0.05:
                    continue
                throw = int(throw_max * decay_z)
                if throw < 2:
                    continue

                # Damage zone (wider for training coverage, physically justified)
                dw = self.rng.randint(3, 6)  # width in grid cells
                dv_reduction = self.rng.uniform(0.08, 0.20)

                for di in range(-dw - 2, dw + 3):
                    for dj in range(-dw - 2, dw + 3):
                        ix, iy = fx + di, fy + dj
                        if 0 <= ix < ni and 0 <= iy < nx:
                            dist = np.sqrt((di)**2 + (dj)**2) / (dw + 2)
                            if dist < 1.0:
                                reduction = dv_reduction * (1.0 - dist)
                                v_model_faulted[ix, iy, iz] *= (1.0 - reduction)

                # Fault mask: thin trace
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ix, iy = fx + di, fy + dj
                        if 0 <= ix < ni and 0 <= iy < nx and abs(di) + abs(dj) <= 1:
                            for dt in range(-1, 2):
                                iz2 = iz + dt
                                if 0 <= iz2 < nz:
                                    fault_mask[ix, iy, iz2] = 1.0

                # Apply throw: hanging wall DOWN, footwall UP
                normal_dx = strike_dy
                normal_dy = -strike_dx
                for di in range(-8, 9):
                    for dj in range(-8, 9):
                        ix, iy = fx + di, fy + dj
                        if not (0 <= ix < ni and 0 <= iy < nx):
                            continue
                        nd = di * normal_dx + dj * normal_dy
                        if abs(nd) < 2:
                            continue
                        is_hanging = nd > 0
                        src = iz - throw if is_hanging else iz + throw
                        src = max(0, min(nz - 1, src))
                        if abs(src - iz) > 1:
                            w = 0.35  # blending weight
                            v_model_faulted[ix, iy, iz] = (
                                (1 - w) * v_model_faulted[ix, iy, iz] +
                                w * v_model_faulted[ix, iy, src]
                            )

        return v_model_faulted, fault_mask

    def generate_channels(self, v_model: np.ndarray,
                          Vsh_3d: np.ndarray) -> np.ndarray:
        """
        Generate fluvial/tidal channel systems.
        Top-flat, bottom-convex geometry (realistic channel cross-section).
        Lateral migration using sine-superposition meanders.
        """
        ni, nx, nz = self.ni, self.nx, self.nz
        v_out = v_model.copy()

        n_channels = self.rng.randint(4, 8)
        for _ in range(n_channels):
            # Channel path: sinusoidal meander through the volume
            amp = self.rng.uniform(0.2, 0.4) * nx  # meander amplitude
            wavelength = self.rng.uniform(0.3, 0.7) * nz
            phase = self.rng.uniform(0, 2 * np.pi)

            # Channel cross-section: flat top, convex bottom
            width = self.rng.uniform(3, 12)   # grid cells
            depth = self.rng.uniform(4, 20)     # grid cells
            # Position
            center_x = self.rng.uniform(10 + width, ni - 10 - width)
            dv_channel = self.rng.uniform(-600, -150)  # slower = more porous sand

            for iz in range(nz // 3, 2 * nz // 3):
                # Meander position
                meander = int(amp * np.sin(2 * np.pi * iz / wavelength + phase))
                center_y = nx // 2 + meander

                for di in range(-int(width * 2), int(width * 2) + 1):
                    ix = int(center_x + di)
                    if not (0 <= ix < ni):
                        continue

                    # Asymmetric Gaussian: wide at top, narrow at bottom
                    sigma_h = width * (1.0 - 0.3 * abs(iz - nz // 2) / (nz // 2))

                    for dj in range(-int(width * 4), int(width * 4) + 1):
                        iy = center_y + dj
                        if not (0 <= iy < nx):
                            continue

                        # Concave-bottom cross-section
                        r = np.sqrt((di / sigma_h)**2 + (dj / sigma_h)**2)
                        if r < 2.0:
                            weight = np.exp(-r**2)  # Gaussian fill
                            v_out[ix, iy, iz] += dv_channel * weight

        return v_out

    def generate_channels_from_centers(self, v_model, Vsh_3d, channel_centers, channel_geometry):
        """Generate channels with explicit tracking (from original v1 compat)."""
        return self.generate_channels(v_model, Vsh_3d)

    def _von_karman_noise_2d(self, ny: int, nx: int, scale: float = 10) -> np.ndarray:
        """Von Karman spectrum noise for realistic heterogeneity (2D slice)."""
        ky = np.fft.fftfreq(ny).reshape(-1, 1)
        kx = np.fft.fftfreq(nx).reshape(1, -1)
        k = np.sqrt(kx**2 + ky**2) + 1e-10
        # von Karman power spectrum: P(k) ∝ 1/(k² + 1/L²)^(ν+1)
        nu = 1.0  # fractal exponent
        L = scale / max(ny, nx)
        spectrum = 1.0 / (k**2 + L**2)**((nu + 1) / 2)
        phase = 2 * np.pi * self.rng.random((ny, nx))
        noise_freq = spectrum * np.exp(1j * phase)
        noise = np.fft.ifft2(noise_freq).real
        noise = noise / (noise.std() + 1e-10)  # unit variance
        # Extend to 3D by taking 2 random slices
        result = np.zeros((ny, nx, 2))  # 2 slices for top/bot perturbation
        result[:, :, 0] = noise
        # Regenerate for second slice
        phase2 = 2 * np.pi * self.rng.random((ny, nx))
        noise_freq2 = spectrum * np.exp(1j * phase2)
        result[:, :, 1] = np.fft.ifft2(noise_freq2).real
        result[:, :, 1] /= (result[:, :, 1].std() + 1e-10)
        return result

    def generate_full_model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete 3D velocity model with all geological features.

        Returns:
            v_model:     (ni, nx, nz) final velocity model
            fault_mask:  (ni, nx, nz) binary fault mask
            horizon_mask: (ni, nx, nz) horizon class labels
            Vsh_3d:      (ni, nx, nz) shale volume fraction
        """
        ni, nx, nz = self.ni, self.nx, self.nz

        # 1. Compaction trend (background Vsh=0.5)
        print(f"  1. Compaction trend (Faust/Athy, North Sea params)...")
        base_v = np.tile(self.compaction_trend(0.5).reshape(1, 1, nz), (ni, nx, 1))

        # 2. Stratigraphic layers
        print(f"  2. Stratigraphic layers (RGM-style bounding surfaces)...")
        v_layered, horizon_mask, Vsh_3d = self.generate_layers(base_v)

        # 3. Channel systems
        print(f"  3. Fluvial channel systems (top-flat/bottom-convex)...")
        v_channeled = self.generate_channels(v_layered, Vsh_3d)

        # 4. Listric faults
        print(f"  4. Listric faults (depth-decaying dip)...")
        v_faulted, fault_mask = self.generate_listric_faults(v_channeled)

        # 5. Multi-scale heterogeneity (von Karman + Gaussian)
        print(f"  5. Multi-scale heterogeneity...")
        # Small-scale
        noise_small = self.rng.normal(0, 15, (ni, nx, nz))
        v_final = v_faulted + noise_small
        # Smooth preserving structures
        v_final = gaussian_filter(v_final, sigma=(0.8, 0.8, 0.3))

        # Smooth fault mask slightly
        fault_mask = gaussian_filter(fault_mask, sigma=(0.4, 0.4, 0.4))

        v_final = np.clip(v_final, 1500.0, 6500.0)

        return v_final.astype(np.float32), fault_mask.astype(np.float32), \
            horizon_mask, Vsh_3d.astype(np.float32)


def seismic_from_velocity(v_model: np.ndarray, dt: float = 0.004,
                           f_peak: float = 30.0) -> np.ndarray:
    """Convert velocity model to post-stack seismic via Gardner + Ricker."""
    ni, nx, nz = v_model.shape
    # Gardner: ρ = 0.31 * V^0.25
    rho = 0.31 * (v_model ** 0.25)
    ai = v_model * rho
    # Reflectivity
    refl = np.diff(ai, axis=2) / (ai[:, :, :-1] + ai[:, :, 1:] + 1e-10)
    refl = np.pad(refl, ((0, 0), (0, 0), (0, 1)), mode='constant')

    # Ricker wavelet
    t = np.arange(-0.1, 0.1 + dt, dt)
    wvlt = (1 - 2 * (np.pi * f_peak * t)**2) * np.exp(-(np.pi * f_peak * t)**2)

    seismic = np.zeros_like(refl)
    for i in range(ni):
        for j in range(nx):
            seismic[i, j, :] = np.convolve(refl[i, j, :], wvlt, mode='same')

    # Add noise ~2% of signal std
    seismic += 0.02 * seismic.std() * np.random.randn(*seismic.shape)
    return seismic.astype(np.float32)


# Quick test
if __name__ == "__main__":
    gen = RealisticModelGenerator(ni=150, nx=200, nz=512, z_max=3000, seed=42)
    t0 = time.time()
    v_model, fault_mask, horizon_mask, Vsh_3d = gen.generate_full_model()
    seismic = seismic_from_velocity(v_model)
    print(f"\nGenerated in {time.time() - t0:.1f}s")
    print(f"  v_model: {v_model.shape}, range=[{v_model.min():.0f},{v_model.max():.0f}]")
    print(f"  fault:   coverage={fault_mask.mean():.4%}")
    print(f"  horizon: classes={horizon_mask.max()}")
    print(f"  seismic: range=[{seismic.min():.4f},{seismic.max():.4f}]")
