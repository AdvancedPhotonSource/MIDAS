"""Consolidated binary writer / reader (scanning indexer format).

Replaces the C binary triplet emitted by ``IndexerScanningOMP``:

- ``IndexBest_all.bin``       — per-voxel orientation candidates (16 floats each)
- ``IndexKey_all.bin``        — per-candidate key metadata (4 size_t each)
- ``IndexBest_IDs_all.bin``   — variable-length matched spot IDs (int32 each)

Byte layout (from ``FF_HEDM/src/IndexerConsolidatedIO.h:8-20`` and the
``IndexerScanningOMP.c`` write paths; matches the parser at
``FF_HEDM/workflows/pf_MIDAS.py:_read_indexbest:87``):

    nVoxels:    int32                       (4 bytes)
    nSolArr:    int32 × nVoxels             (4 * nVoxels bytes)
    offArr:     int64 × nVoxels             (8 * nVoxels bytes)
    records:    float64 × 16 × Σ nSolArr    (variable)

``offArr[v]`` gives the **byte offset from the start of file** at which
voxel v's records begin (so header_size + Σ_{v' < v} 16 * 8 * nSolArr[v']).

The vals layout is 16 columns of float64 per candidate per voxel, mirroring
``IndexerScanningOMP.c``'s writer (and the find_grains consumer at
``midas_pipeline.find_grains._consolidation_io``).

This module provides the **scanning** layout used by ``IndexerScanningOMP``
and consumed by find_grains. The non-scanning ``IndexBest.bin`` written
by ``IndexerOMP`` is a different, simpler format and lives in ``output.py``.

Parity contract: the writer here is bit-exact against the C output when
given the same per-voxel solution arrays. The round-trip test in
``test_consolidated_io.py`` pins this contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


VALS_COLS = 16          # CONSOLIDATED_VALS_COLS from IndexerConsolidatedIO.h
KEY_COLS = 4            # CONSOLIDATED_KEY_COLS — used by IndexKey_all.bin


def header_size_bytes(n_voxels: int) -> int:
    """Byte offset where the records begin: 4 + 4·nVox + 8·nVox."""
    return 4 + 4 * n_voxels + 8 * n_voxels


@dataclass
class ConsolidatedReadResult:
    """Parsed ``IndexBest_all.bin`` payload."""

    n_voxels: int
    n_sol_arr: np.ndarray   # int32 (n_voxels,)
    off_arr: np.ndarray     # int64 (n_voxels,) — byte offset from file start
    vals: np.ndarray        # float64 (Σ n_sol_arr, 16)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_index_best_all(
    path: str | Path,
    per_voxel_records: List[np.ndarray],
) -> None:
    """Serialize per-voxel candidate records to ``IndexBest_all.bin``.

    Parameters
    ----------
    path
        Output file path.
    per_voxel_records
        List of length ``n_voxels``. Each entry is a ``(n_solutions, 16)``
        float64 array of candidate records for that voxel. An empty
        array (shape ``(0, 16)``) signifies a voxel that produced no
        solutions.

    The file is written as: header (nVoxels + nSolArr + offArr)
    followed by concatenated record blocks. ``offArr[v]`` points at
    the start of voxel v's records relative to the file start.
    """
    p = Path(path)
    n_voxels = len(per_voxel_records)
    n_sol_arr = np.zeros(n_voxels, dtype=np.int32)
    for v, rec in enumerate(per_voxel_records):
        rec = np.asarray(rec, dtype=np.float64)
        if rec.ndim != 2 or (rec.shape[0] > 0 and rec.shape[1] != VALS_COLS):
            raise ValueError(
                f"per_voxel_records[{v}] must be shape (n_sol, {VALS_COLS}); "
                f"got {rec.shape}"
            )
        n_sol_arr[v] = rec.shape[0]

    header_size = header_size_bytes(n_voxels)
    bytes_per_voxel = (VALS_COLS * 8) * n_sol_arr.astype(np.int64)
    # cumulative[v] = bytes before voxel v's records.
    cumulative = np.concatenate(([0], np.cumsum(bytes_per_voxel)[:-1]))
    off_arr = (header_size + cumulative).astype(np.int64)

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_sol_arr.tobytes())
        f.write(off_arr.tobytes())
        for rec in per_voxel_records:
            rec_arr = np.asarray(rec, dtype=np.float64)
            if rec_arr.shape[0] > 0:
                f.write(np.ascontiguousarray(rec_arr).tobytes())


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_index_best_all(path: str | Path) -> ConsolidatedReadResult:
    """Inverse of :func:`write_index_best_all`.

    Returns the parsed header arrays plus a stacked ``(total_solutions,
    16)`` float64 record table — same shape callers like ``find_grains``
    consume via ``pf_MIDAS.py:_read_indexbest``.
    """
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < 4:
        raise ValueError(f"IndexBest_all.bin at {p!s} too short ({len(raw)} bytes)")
    n_voxels = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    if n_voxels < 0:
        raise ValueError(
            f"IndexBest_all.bin at {p!s} has invalid n_voxels={n_voxels}"
        )
    cursor = 4
    n_sol_arr = np.frombuffer(
        raw[cursor:cursor + 4 * n_voxels], dtype=np.int32,
    ).copy()
    cursor += 4 * n_voxels
    off_arr = np.frombuffer(
        raw[cursor:cursor + 8 * n_voxels], dtype=np.int64,
    ).copy()
    cursor += 8 * n_voxels
    if cursor != header_size_bytes(n_voxels):
        raise ValueError(
            f"header size mismatch reading {p!s}: expected "
            f"{header_size_bytes(n_voxels)}, got cursor={cursor}"
        )
    total_solutions = int(n_sol_arr.sum())
    vals_bytes = total_solutions * VALS_COLS * 8
    if len(raw) - cursor != vals_bytes:
        raise ValueError(
            f"vals byte count mismatch reading {p!s}: expected {vals_bytes}, "
            f"got {len(raw) - cursor}"
        )
    vals = np.frombuffer(raw[cursor:], dtype=np.float64).reshape(
        total_solutions, VALS_COLS,
    ).copy()
    return ConsolidatedReadResult(
        n_voxels=n_voxels,
        n_sol_arr=n_sol_arr,
        off_arr=off_arr,
        vals=vals,
    )


def split_records_by_voxel(
    result: ConsolidatedReadResult,
) -> List[np.ndarray]:
    """Return a list of ``(n_solutions, 16)`` arrays — one per voxel."""
    out: List[np.ndarray] = []
    cursor = 0
    for v in range(result.n_voxels):
        n = int(result.n_sol_arr[v])
        out.append(result.vals[cursor:cursor + n].copy())
        cursor += n
    return out
