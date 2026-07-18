"""
Task-specific prediction heads for geological feature detection.

Re-designed downstream tasks that REQUIRE well log information:

  1. Velocity prediction — absolute velocity from seismic (wells provide ground truth)
  2. Porosity (NPHI) prediction — cannot be directly derived from seismic
  3. Lithology (GR) classification — sand/shale discrimination needs well logs
  4. Density (RHOB) prediction — Gardner's relation is approximate, wells give truth
  5. Resistivity (RT) prediction — fluid indicator, totally absent from seismic

These tasks demonstrate why well log fusion MATTERS — unlike faults/horizons
which are structural and visible in seismic directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityHead(nn.Module):
    """Predict velocity in z-scored space (output ∈ [-3, 3])."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh(),   # bounded: [-1, 1] → z-scored range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 3.0 * self.net(x)  # scale to ~±3σ


class PorosityHead(nn.Module):
    """Predict neutron porosity NPHI (v/v) at each pixel."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # porosity ∈ [0, 0.45]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.45 * self.net(x)


class LithologyHead(nn.Module):
    """Classify lithology from GR response (shale/silt/sand)."""
    # 3 classes: shale (high GR), silt (mid GR), sand (low GR)

    def __init__(self, in_channels: int = 128, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DensityHead(nn.Module):
    """Predict density in z-scored space (output ∈ [-3, 3])."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 3.0 * self.net(x)


class ResistivityHead(nn.Module):
    """Predict resistivity RT (ohm·m) at each pixel — fluid indicator."""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),  # RT > 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHeads(nn.Module):
    """
    Combined multi-task head for petrophysical property prediction.

    These tasks specifically require cross-modal fusion because:
    - Velocity is a slow-varying absolute quantity (seismic = changes)
    - Porosity, density, resistivity have NO expression in seismic amplitudes
    - Lithology (GR) measures clay content, optically invisible in seismic
    """

    def __init__(self, in_channels: int = 128, n_litho: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.velocity = VelocityHead(128)
        self.porosity = PorosityHead(128)
        self.lithology = LithologyHead(128, n_litho)
        self.density = DensityHead(128)
        self.resistivity = ResistivityHead(128)

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.shared(x)
        return {
            "velocity": self.velocity(shared),
            "porosity": self.porosity(shared),
            "lithology": self.lithology(shared),
            "density": self.density(shared),
            "resistivity": self.resistivity(shared),
        }
