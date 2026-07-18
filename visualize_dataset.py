#!/usr/bin/env python3
"""
Visualization script for the OpenSeisML dataset.
Shows: seismic sections, well logs, velocity model, and comparisons.
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ── Load dataset ──────────────────────────────────────────────────
PATH = "./openseisml_dataset.h5"
f = h5py.File(PATH, 'r')

# ── Figure 1: Seismic section with well overlay ──────────────────
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# (a) Seismic section 0 (inline)
ax1 = fig.add_subplot(gs[0, 0])
sec0 = f["seismic/section_0000"][:]
w0 = f["seismic/section_0000"].attrs["well_index"]
extent = [0, sec0.shape[1] * 12.5, 2754, 0]
im1 = ax1.imshow(sec0, cmap='seismic', aspect='auto', extent=extent,
                 vmin=-0.1, vmax=0.1)
ax1.axvline(x=w0 * 12.5, color='lime', linewidth=2, linestyle='--', label=f'Well 0001')
ax1.set_title(f'Seismic Section 0000 (Inline)\nWell at x={w0*12.5:.0f}m', fontsize=11)
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('Depth (m)')
ax1.legend()
plt.colorbar(im1, ax=ax1, shrink=0.8, label='Amplitude')

# (b) Seismic section 1 (xline)
ax2 = fig.add_subplot(gs[0, 1])
sec1 = f["seismic/section_0001"][:]
w1 = f["seismic/section_0001"].attrs["well_index"]
extent2 = [0, sec1.shape[1] * 12.5, 2754, 0]
im2 = ax2.imshow(sec1, cmap='seismic', aspect='auto', extent=extent2,
                 vmin=-0.1, vmax=0.1)
ax2.axvline(x=w1 * 12.5, color='lime', linewidth=2, linestyle='--', label=f'Well 0002')
ax2.set_title(f'Seismic Section 0001 (Xline)\nWell at x={w1*12.5:.0f}m', fontsize=11)
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Depth (m)')
ax2.legend()
plt.colorbar(im2, ax=ax2, shrink=0.8, label='Amplitude')

# (c) Another section
ax3 = fig.add_subplot(gs[0, 2])
sec5 = f["seismic/section_0005"][:]
w5 = f["seismic/section_0005"].attrs["well_index"]
im3 = ax3.imshow(sec5, cmap='seismic', aspect='auto', extent=extent,
                 vmin=-0.1, vmax=0.1)
ax3.axvline(x=w5 * 12.5, color='lime', linewidth=2, linestyle='--', label=f'Well 0006')
ax3.set_title(f'Seismic Section 0005\nWell at x={w5*12.5:.0f}m', fontsize=11)
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Depth (m)')
ax3.legend()
plt.colorbar(im3, ax=ax3, shrink=0.8, label='Amplitude')

# ── Well log panels ──────────────────────────────────────────────
well_names = ["WELL_0001", "WELL_0010", "WELL_0020"]
log_keys = ["GR", "DT", "RHOB", "NPHI", "RT"]
log_colors = ['green', 'blue', 'red', 'purple', 'brown']
log_labels = ['Gamma Ray\n(API)', 'Sonic DT\n(μs/ft)', 'Density\n(g/cm³)',
              'Neutron Porosity\n(v/v)', 'Resistivity\n(ohm·m)']

for wi, wname in enumerate(well_names):
    ax_log = fig.add_subplot(gs[1, wi])
    depth = f[f"wells/{wname}/depth"][:]

    for ki, (k, c, lbl) in enumerate(zip(log_keys, log_colors, log_labels)):
        data = f[f"wells/{wname}/{k}"][:]
        # Normalize each curve for display
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
        ax_log.plot(data_norm + ki * 0.15, depth, color=c, linewidth=0.8, label=lbl)

    ax_log.set_title(f'{wname} Logs', fontsize=10)
    ax_log.set_ylim(depth[-1], depth[0])
    ax_log.set_xlabel('Normalized logs (offset for clarity)')
    ax_log.set_ylabel('Depth (m)')
    ax_log.legend(loc='lower right', fontsize=7)

fig.suptitle('OpenSeisML Dataset — Seismic Sections & Well Logs\n'
             f'{len(f["seismic"])} sections, 256×512, 12.5m sampling',
             fontsize=13, fontweight='bold', y=1.01)

os.makedirs('./figures', exist_ok=True)
plt.savefig('./figures/01_seismic_and_wells.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/01_seismic_and_wells.png")


# ── Figure 2: Velocity QC ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) TWT vs Depth — use harmonized well depth
depth_w = f[f"wells/WELL_0001/depth"][:]
v_well = f[f"wells/WELL_0001/velocity"][:]
n_samp = len(depth_w)
twt_interp = 2 * np.cumsum(np.gradient(depth_w) / np.clip(v_well, 100, 10000))

# Simulate checkshot sparse measurements
cs_indices = np.arange(0, n_samp, n_samp // 10)
axes[0].plot(twt_interp, depth_w, 'b-', linewidth=1.5, label='Interpolated well velocity')
axes[0].scatter(twt_interp[cs_indices], depth_w[cs_indices], c='red', s=30,
                marker='o', label='Checkshot points', zorder=5)
axes[0].set_xlabel('Two-Way Travel Time (s)')
axes[0].set_ylabel('Depth (m)')
axes[0].set_title('TWT vs Depth (WELL_0001)\n(cf. paper Fig.4a)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) Velocity vs Depth
axes[1].plot(v_well, depth_w, 'b-', linewidth=1.5, label='Interpolated well velocity')
axes[1].scatter(v_well[cs_indices], depth_w[cs_indices], c='red', s=30,
                marker='o', label='Checkshot points', zorder=5)
axes[1].set_xlabel('Velocity (m/s)')
axes[1].set_ylabel('Depth (m)')
axes[1].set_title('Depth vs Velocity (WELL_0001)\n(cf. paper Fig.4b)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/02_velocity_qc.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/02_velocity_qc.png")


# ── Figure 3: 3D Velocity Volume Slice ───────────────────────────
if "velocity_volume_3d" in f:
    v3d = f["velocity_volume_3d"][:]
    ni, nx, nz = v3d.shape

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Inline slice
    im_i = axes[0].imshow(v3d[ni // 2, :, :].T, cmap='viridis', aspect='auto',
                          extent=[0, nx * 12.5, 2754, 0])
    axes[0].set_title(f'Velocity — Inline {ni // 2}')
    axes[0].set_xlabel('Xline distance (m)')
    axes[0].set_ylabel('Depth (m)')
    plt.colorbar(im_i, ax=axes[0], shrink=0.8, label='m/s')

    # Xline slice
    im_x = axes[1].imshow(v3d[:, nx // 2, :].T, cmap='viridis', aspect='auto',
                          extent=[0, ni * 12.5, 2754, 0])
    axes[1].set_title(f'Velocity — Xline {nx // 2}')
    axes[1].set_xlabel('Inline distance (m)')
    axes[1].set_ylabel('Depth (m)')
    plt.colorbar(im_x, ax=axes[1], shrink=0.8, label='m/s')

    # Depth slice
    im_z = axes[2].imshow(v3d[:, :, nz // 3], cmap='viridis', aspect='equal',
                          extent=[0, nx * 12.5, 0, ni * 12.5])
    axes[2].set_title(f'Velocity — Depth slice {nz // 3}')
    axes[2].set_xlabel('Xline distance (m)')
    axes[2].set_ylabel('Inline distance (m)')
    plt.colorbar(im_z, ax=axes[2], shrink=0.8, label='m/s')

    plt.suptitle('3D Average Velocity Volume (RBF-Interpolated from Checkshots)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('./figures/03_velocity_volume.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: figures/03_velocity_volume.png")


# ── Figure 4: Multi-well log comparison ──────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
log_pairs = [("GR", "DT"), ("RHOB", "NPHI"), ("RT", "velocity")]

for row, (k1, k2) in enumerate(log_pairs):
    for col in range(2):
        ax = axes[row, col]
        wname = f"WELL_{(col * 10 + 1):04d}"

        d1 = f[f"wells/{wname}/{k1}"][:]
        d2 = f[f"wells/{wname}/{k2}"][:]
        depth = f[f"wells/{wname}/depth"][:]

        u1 = f[f"wells/{wname}/{k1}"].attrs.get("unit", "")
        u2 = f[f"wells/{wname}/{k2}"].attrs.get("unit", "")

        ax.plot(d1, depth, 'b-', linewidth=1, label=f'{k1} ({u1})')
        ax_twin = ax.twiny()
        ax_twin.plot(d2, depth, 'r-', linewidth=1, label=f'{k2} ({u2})')

        ax.set_title(f'{wname}')
        ax.set_ylim(depth[-1], depth[0])
        ax.set_xlabel(f'{k1} ({u1})', color='blue')
        ax_twin.set_xlabel(f'{k2} ({u2})', color='red')
        ax.grid(alpha=0.3)

plt.suptitle('Cross-Log Comparisons at Different Wells', fontweight='bold')
plt.tight_layout()
plt.savefig('./figures/04_cross_log_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: figures/04_cross_log_comparison.png")


# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Dataset: {PATH}")
print(f"  Seismic sections: {len(f['seismic'])}  ×  {f['seismic/section_0000'].shape}")
print(f"  Wells:             {len(f['wells'])}")
print(f"  Log types:         GR, NPHI, RHOB, DT, RT, velocity")
print(f"{'='*60}")

f.close()
