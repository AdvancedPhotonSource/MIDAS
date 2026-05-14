"""``UniqueIndexSingleKey.bin`` + ``SpotsToIndex.csv`` writers.

``UniqueIndexSingleKey.bin`` is written by both find-single (one row per
voxel = best-solution key bundle) and find-multiple (one row per cluster
per voxel — multiple rows per voxel). The C code uses ``pwrite`` to
write at byte offset ``5*sizeof(size_t)*voxNr`` (=40-byte rows assuming
8-byte ``size_t``); we replicate that with explicit row-indexed writes
to a single ``uint64`` array.

Layout per row (5 cols, size_t aka uint64 on 64-bit):
  ``[voxNr, SpotID, nMatches, nIDs, bestSolIdx]``

``SpotsToIndex.csv`` is the find-multiple output — one row per
(voxel, cluster), the same 5 columns formatted as space-separated text:

  ``voxNr SpotID nMatches nIDs bestSolIdx``

Note: the C ``findMultipleSolutionsPF`` writes the per-voxel
``UniqueIndexKey_voxNr_<NNNNNN>.txt`` files first, then aggregates them
into ``SpotsToIndex.csv``. We skip the per-voxel intermediate text files
(they're an artifact of the C OpenMP fan-out) and aggregate directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def write_unique_index_single_key(
    path: str | Path,
    n_voxels: int,
    rows: Iterable[tuple[int, np.ndarray]],
) -> None:
    """Write ``UniqueIndexSingleKey.bin`` with C-pwrite semantics.

    Parameters
    ----------
    path : path-like
    n_voxels : int
        Total number of voxels (file is pre-sized to
        ``5 * 8 * n_voxels`` bytes).
    rows : iterable of (voxNr, row_uint64)
        ``row_uint64`` is a 5-element array of uint64. For voxels with no
        valid solution, do NOT emit a row — they stay all-zero. C
        semantics: every voxel slot has a 5-tuple; invalid voxels get
        all zeros via the ``size_t outarr[5] = {0}`` pwrite.

    Notes
    -----
    Matches ``pwrite(ib, outarr, 5*sizeof(size_t), 5*sizeof(size_t)*voxNr)``
    in ``findSingleSolutionPFRefactored.c:611, 695, 791``.
    """
    arr = np.zeros((n_voxels, 5), dtype=np.uint64)
    for vox_nr, row in rows:
        if vox_nr < 0 or vox_nr >= n_voxels:
            raise IndexError(vox_nr)
        r = np.asarray(row, dtype=np.uint64)
        if r.shape != (5,):
            raise ValueError(f"row {vox_nr}: shape {r.shape}, expected (5,)")
        arr[vox_nr] = r
    Path(path).write_bytes(arr.tobytes())


def read_unique_index_single_key(path: str | Path) -> np.ndarray:
    """Read back ``UniqueIndexSingleKey.bin`` as (n_voxels, 5) uint64."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint64)
    if data.size % 5 != 0:
        raise ValueError(f"file size not divisible by 5*8: {data.size}")
    return data.reshape(-1, 5)


def write_spots_to_index_csv(
    path: str | Path,
    rows_per_voxel: dict[int, np.ndarray],
) -> None:
    """Write ``SpotsToIndex.csv`` from per-voxel cluster rows.

    Parameters
    ----------
    path : path-like
    rows_per_voxel : dict[int, ndarray (n_clusters_v, 5) uint64]
        For each voxel, the (n_clusters_v, 5) uint64 rows. Voxels not in
        the dict (or with empty rows) contribute no lines. Output is
        emitted in ascending voxNr order to match the C ``writeSpotsToIndex``
        which iterates ``for voxNr in [0, nScans*nScans)``.

    Output format (matches C lines 213-218): one space-separated line
    per cluster, ``voxNr v0 v1 v2 v3 v4\\n`` where ``v0..v4`` is the
    5-tuple ``[SpotID, nMatches, nIDs, ?, bestSolIdx]`` from the C
    per-voxel UniqueIndexKey file.

    Notes
    -----
    In the C path, the per-voxel ``uniqueArr[i*5+0..4]`` columns are
    ``[SpotID, nMatches, 0, 0, bestSolIdx]`` (col 2 and 3 are zero-init
    placeholders). We follow the same convention.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for vox_nr in sorted(rows_per_voxel.keys()):
            rows = rows_per_voxel[vox_nr]
            for r in rows:
                f.write(f"{int(vox_nr)} {int(r[0])} {int(r[1])} {int(r[2])} {int(r[3])} {int(r[4])}\n")


def write_unique_orientations_csv(
    path: str | Path,
    unique_key_arr: np.ndarray,
    unique_OM_arr: np.ndarray,
) -> None:
    """Write ``UniqueOrientations.csv``.

    Layout from C ``save_orientation_results`` (lines 978–1011): one
    header line starting with ``#``, then one row per unique grain with
    5 uint columns followed by 9 float columns (OM, row-major).

    ``unique_key_arr`` has shape ``(n_uniques, 5)``: ``[GrainID, RowNr,
    nSpots, StartRowNr, ListStartPos]``. ``unique_OM_arr`` has shape
    ``(n_uniques, 9)``.

    Formatting matches the C ``fprintf("%zu ")`` (no zero-padding,
    space separator) and ``fprintf("%lf ")`` (default 6-digit precision)
    — bit-equivalent text output.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(unique_key_arr.shape[0])
    with open(p, "w") as f:
        f.write(
            "# GrainID RowNr nSpots StartRowNr ListStartPos OM1 OM2 OM3 OM4 OM5 OM6 OM7 OM8 OM9\n"
        )
        for i in range(n):
            keys = unique_key_arr[i]
            oms = unique_OM_arr[i]
            for j in range(5):
                f.write(f"{int(keys[j])} ")
            for j in range(9):
                f.write(f"{float(oms[j]):f} ")
            f.write("\n")
