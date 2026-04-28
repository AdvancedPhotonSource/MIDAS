"""Multi-panel detector geometry. Mirrors FF_HEDM/src/Panel.{c,h}."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Panel:
    id: int = 0
    yMin: int = 0
    yMax: int = 0
    zMin: int = 0
    zMax: int = 0
    dY: float = 0.0
    dZ: float = 0.0
    dTheta: float = 0.0
    dLsd: float = 0.0
    dP2: float = 0.0
    centerY: float = 0.0
    centerZ: float = 0.0


def generate_panels(
    nPanelsY: int,
    nPanelsZ: int,
    panelSizeY: int,
    panelSizeZ: int,
    gapsY: Optional[List[int]],
    gapsZ: Optional[List[int]],
) -> List[Panel]:
    """Replicates ``GeneratePanels`` in Panel.c — row-major Y outer, Z inner."""
    if nPanelsY <= 0 or nPanelsZ <= 0:
        return []
    panels: List[Panel] = []
    currentY = 0
    idx = 0
    for i in range(nPanelsY):
        yStart = currentY
        yEnd = yStart + panelSizeY - 1
        currentZ = 0
        for j in range(nPanelsZ):
            zStart = currentZ
            zEnd = zStart + panelSizeZ - 1
            panels.append(
                Panel(
                    id=idx,
                    yMin=yStart,
                    yMax=yEnd,
                    zMin=zStart,
                    zMax=zEnd,
                    centerY=(yStart + yEnd) / 2.0,
                    centerZ=(zStart + zEnd) / 2.0,
                )
            )
            idx += 1
            if j < nPanelsZ - 1:
                currentZ = zEnd + 1 + (gapsZ[j] if gapsZ else 0)
        if i < nPanelsY - 1:
            currentY = yEnd + 1 + (gapsY[i] if gapsY else 0)
    return panels


def load_panel_shifts(filename: str, panels: List[Panel]) -> bool:
    """Read text file: ``id dY dZ [dTheta dLsd dP2]``. Mutates panels in place."""
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = line.split()
                try:
                    pid = int(tokens[0])
                    dY = float(tokens[1])
                    dZ = float(tokens[2])
                except (IndexError, ValueError):
                    continue
                if not (0 <= pid < len(panels)):
                    continue
                panels[pid].dY = dY
                panels[pid].dZ = dZ
                if len(tokens) >= 4:
                    panels[pid].dTheta = float(tokens[3])
                if len(tokens) >= 5:
                    panels[pid].dLsd = float(tokens[4])
                if len(tokens) >= 6:
                    panels[pid].dP2 = float(tokens[5])
        return True
    except (OSError, ValueError):
        return False


def get_panel_index(y: float, z: float, panels: List[Panel]) -> int:
    """Return panel index containing (y, z), or -1 if not in any panel."""
    for p in panels:
        if p.yMin <= y <= p.yMax and p.zMin <= z <= p.zMax:
            return p.id
    return -1


def panel_index_map(
    NrPixelsY: int, NrPixelsZ: int, panels: List[Panel]
) -> np.ndarray:
    """Build a (NrPixelsY, NrPixelsZ) int32 map of panel-IDs (-1 if not covered).

    Vectorized: avoids the O(P) per-pixel lookup of the C version.
    """
    pmap = np.full((NrPixelsY, NrPixelsZ), -1, dtype=np.int32)
    for p in panels:
        y0, y1 = max(0, p.yMin), min(NrPixelsY - 1, p.yMax)
        z0, z1 = max(0, p.zMin), min(NrPixelsZ - 1, p.zMax)
        if y0 <= y1 and z0 <= z1:
            pmap[y0 : y1 + 1, z0 : z1 + 1] = p.id
    return pmap


def apply_panel_correction(
    y: float, z: float, p: Panel
) -> Tuple[float, float]:
    """Replicates ``ApplyPanelCorrection`` (Panel.h)."""
    dy = y - p.centerY
    dz = z - p.centerZ
    if p.dTheta != 0.0:
        rad = math.radians(p.dTheta)
        cosT, sinT = math.cos(rad), math.sin(rad)
        return (
            p.centerY + dy * cosT - dz * sinT + p.dY,
            p.centerZ + dy * sinT + dz * cosT + p.dZ,
        )
    return y + p.dY, z + p.dZ


__all__ = [
    "Panel",
    "generate_panels",
    "load_panel_shifts",
    "get_panel_index",
    "panel_index_map",
    "apply_panel_correction",
]
