"""Per-panel detector corrections.

Pure-Python (numpy) port of FF_HEDM/src/Panel.h + Panel.c.

A Panel describes a rectangular region of the detector that has been shifted,
rotated, and offset in distance/distortion relative to the nominal geometry.
``mapper_build_map`` looks up each pixel's owning panel and uses its
``dY/dZ/dTheta`` to displace the pixel coordinates and its ``dLsd/dP2`` to
adjust the radial geometry inside the forward transform.

The panel array layout used by the numba kernels is a flat
``(n_panels, 11)`` float64 array:

    col 0:  yMin     col 4:  dY        col 8:  dP2
    col 1:  yMax     col 5:  dZ        col 9:  centerY
    col 2:  zMin     col 6:  dTheta    col 10: centerZ
    col 3:  zMax     col 7:  dLsd

The integer panel id equals its row index in the array.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass + I/O
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Panel:
    id: int
    yMin: int
    yMax: int           # inclusive
    zMin: int
    zMax: int           # inclusive
    dY: float = 0.0
    dZ: float = 0.0
    dTheta: float = 0.0     # in-plane rotation, degrees
    dLsd: float = 0.0       # per-panel Lsd offset, microns
    dP2: float = 0.0        # per-panel p2 distortion offset

    @property
    def centerY(self) -> float:
        return (self.yMin + self.yMax) / 2.0

    @property
    def centerZ(self) -> float:
        return (self.zMin + self.zMax) / 2.0


def generate_panels(
    n_panels_y: int,
    n_panels_z: int,
    panel_size_y: int,
    panel_size_z: int,
    gaps_y: Optional[Sequence[int]] = None,
    gaps_z: Optional[Sequence[int]] = None,
) -> List[Panel]:
    """Lay out ``n_panels_y × n_panels_z`` panels with optional gaps.

    Mirrors ``GeneratePanels`` in Panel.c.

    Args:
        n_panels_y: panels along Y.
        n_panels_z: panels along Z.
        panel_size_y: pixel width of each panel.
        panel_size_z: pixel height of each panel.
        gaps_y: gap (in pixels) between adjacent Y panels; length n_panels_y - 1.
        gaps_z: gap (in pixels) between adjacent Z panels; length n_panels_z - 1.

    Returns:
        A list of Panel objects, ordered Y-outer × Z-inner (matches C).
    """
    if gaps_y is None:
        gaps_y = [0] * max(n_panels_y - 1, 0)
    if gaps_z is None:
        gaps_z = [0] * max(n_panels_z - 1, 0)
    if len(gaps_y) < n_panels_y - 1 or len(gaps_z) < n_panels_z - 1:
        raise ValueError(
            f"gaps_y / gaps_z too short: need {n_panels_y - 1} / {n_panels_z - 1}"
        )

    panels: List[Panel] = []
    current_y = 0
    idx = 0
    for i in range(n_panels_y):
        y_start = current_y
        y_end = y_start + panel_size_y - 1
        if i < n_panels_y - 1:
            current_y = y_end + 1 + int(gaps_y[i])

        current_z = 0
        for j in range(n_panels_z):
            z_start = current_z
            z_end = z_start + panel_size_z - 1
            if j < n_panels_z - 1:
                current_z = z_end + 1 + int(gaps_z[j])

            panels.append(Panel(
                id=idx,
                yMin=y_start, yMax=y_end,
                zMin=z_start, zMax=z_end,
            ))
            idx += 1
    return panels


def load_panel_shifts(filename: str | Path,
                      panels: List[Panel]) -> List[Panel]:
    """Apply shifts from a text file (mirrors ``LoadPanelShifts``).

    File format (one line per panel)::

        # ID dY dZ [dTheta] [dLsd] [dP2]
        0 0.5  0.0
        1 0.0 -0.3 0.01

    Comments (``#``) and blank lines are skipped. Missing columns default to
    zero (matching the C version).
    """
    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                pid = int(tokens[0])
                dY = float(tokens[1])
                dZ = float(tokens[2])
                dTheta = float(tokens[3]) if len(tokens) > 3 else 0.0
                dLsd = float(tokens[4]) if len(tokens) > 4 else 0.0
                dP2 = float(tokens[5]) if len(tokens) > 5 else 0.0
            except ValueError:
                continue
            if 0 <= pid < len(panels):
                panels[pid].dY = dY
                panels[pid].dZ = dZ
                panels[pid].dTheta = dTheta
                panels[pid].dLsd = dLsd
                panels[pid].dP2 = dP2
    return panels


def save_panel_shifts(filename: str | Path, panels: Sequence[Panel]) -> None:
    """Write panel shifts in the same format read by ``load_panel_shifts``."""
    with open(filename, "w") as f:
        f.write("# ID dY dZ dTheta dLsd dP2\n")
        for p in panels:
            f.write(
                f"{p.id} {p.dY:.10f} {p.dZ:.10f} {p.dTheta:.10f} "
                f"{p.dLsd:.10f} {p.dP2:.10f}\n"
            )


def get_panel_index(y: float, z: float, panels: Sequence[Panel]) -> int:
    """Return panel index containing ``(y, z)``, or -1 if none."""
    for i, p in enumerate(panels):
        if p.yMin <= y <= p.yMax and p.zMin <= z <= p.zMax:
            return i
    return -1


def apply_panel_correction(y: float, z: float, panel: Panel) -> tuple[float, float]:
    """Apply rotation around panel center, then translational shift.

    Mirrors ``ApplyPanelCorrection`` in Panel.h.
    """
    dy = y - panel.centerY
    dz = z - panel.centerZ
    if panel.dTheta != 0.0:
        rad = math.radians(panel.dTheta)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        y_out = panel.centerY + dy * cos_t - dz * sin_t + panel.dY
        z_out = panel.centerZ + dy * sin_t + dz * cos_t + panel.dZ
    else:
        y_out = y + panel.dY
        z_out = z + panel.dZ
    return y_out, z_out


def panels_to_array(panels: Sequence[Panel]) -> np.ndarray:
    """Pack panel parameters into a contiguous (n_panels, 11) float64 array.

    Layout matches ``midas_integrate._mapper_numba`` panel-correction helpers.
    """
    if len(panels) == 0:
        return np.zeros((0, 11), dtype=np.float64)
    arr = np.empty((len(panels), 11), dtype=np.float64)
    for i, p in enumerate(panels):
        arr[i, 0] = p.yMin
        arr[i, 1] = p.yMax
        arr[i, 2] = p.zMin
        arr[i, 3] = p.zMax
        arr[i, 4] = p.dY
        arr[i, 5] = p.dZ
        arr[i, 6] = p.dTheta
        arr[i, 7] = p.dLsd
        arr[i, 8] = p.dP2
        arr[i, 9] = p.centerY
        arr[i, 10] = p.centerZ
    return arr


def empty_panel_array() -> np.ndarray:
    """Return the sentinel `(0, 11)` array used when no panels are configured."""
    return np.zeros((0, 11), dtype=np.float64)
