"""Scan geometry helpers — positions.csv parsing + scan→spatial mapping.

The PF scan grid is 1-D along Y. ``positions.csv`` lists the Y positions
in file order; the 2-D voxel grid is the Cartesian product of two sorted
copies of that list (see ``SaveBinDataScanning.c:1667-1683``). Sinograms
are written in spatially-sorted order, so we need a mapping from the
file-order scan index to the spatially-sorted column index.

Matches the C global ``scan_to_spatial[fileIdx] = spatialIdx`` computed
in ``findSingleSolutionPFRefactored.c:127`` via ``cmp_argsort_asc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ScanGrid:
    """Parsed scan-position layout.

    Attributes
    ----------
    file_positions : ndarray (n_scans,) float64
        Positions as listed in ``positions.csv`` (file order).
    spatial_positions : ndarray (n_scans,) float64
        Sorted ascending — same values, in spatial order.
    scan_to_spatial : ndarray (n_scans,) int64
        ``scan_to_spatial[fileIdx] = spatialIdx``. Use this to convert
        the spotwise ``scanNr`` (file order) to the column index used in
        sinos / spotMapping output.
    n_scans : int
    """

    file_positions: np.ndarray
    spatial_positions: np.ndarray
    scan_to_spatial: np.ndarray
    n_scans: int


def read_positions_csv(path: str | Path) -> ScanGrid:
    """Parse ``positions.csv`` and compute the scan→spatial mapping.

    The file is a column of floats — one Y position per line. Blank
    lines and ``#``-comment lines are tolerated.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    vals: list[float] = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # tolerate "x  y" and take the first; positions.csv historically
            # has one column but be permissive.
            tok = s.split()
            vals.append(float(tok[0]))
    file_pos = np.asarray(vals, dtype=np.float64)
    return build_scan_grid(file_pos)


def build_scan_grid(file_positions: np.ndarray) -> ScanGrid:
    """Construct a :class:`ScanGrid` from a 1-D array of Y positions.

    Mirrors the C argsort_asc logic exactly.
    """
    arr = np.asarray(file_positions, dtype=np.float64).ravel()
    n = int(arr.shape[0])
    if n == 0:
        return ScanGrid(arr, arr.copy(), np.zeros(0, dtype=np.int64), 0)
    order = np.argsort(arr, kind="stable")
    spatial = arr[order].copy()
    # scan_to_spatial[fileIdx] = spatialIdx; need inverse of order.
    scan_to_spatial = np.empty(n, dtype=np.int64)
    scan_to_spatial[order] = np.arange(n, dtype=np.int64)
    return ScanGrid(
        file_positions=arr,
        spatial_positions=spatial,
        scan_to_spatial=scan_to_spatial,
        n_scans=n,
    )


def voxel_to_xy_um(vox_nr: int, n_scans: int, spatial_positions: np.ndarray) -> tuple[float, float]:
    """Voxel index → ``(x_V, y_V)`` in micrometers.

    Layout from the C scanning indexer / find_grains: the 2-D voxel grid
    is the Cartesian product of two sorted-Y axes; ``voxNr = row*nScans
    + col``. Both axes use the same ``spatial_positions`` array.

    Matches ``generate_sinograms_from_indexing`` lines 1995-1998.
    """
    if vox_nr < 0 or vox_nr >= n_scans * n_scans:
        raise IndexError(vox_nr)
    row = vox_nr // n_scans
    col = vox_nr % n_scans
    return float(spatial_positions[col]), float(spatial_positions[row])
