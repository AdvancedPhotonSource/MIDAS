"""Byte-parity Python port of ``NF_HEDM/src/ParseMic.c``.

Reads a binary ``MicFileBinary`` (N rows × 11 doubles each) and produces
the same five output files the C executable does:

  - ``<basename>``         — plain-text mic file
  - ``<basename>.map``     — 2D rasterised orientation map
                              (``size_map * 7 + 4`` doubles, see :func:`write_map`)
  - ``<basename>.map.kam`` — kernel-average misorientation (radians)
  - ``<basename>.map.grainId`` — connected-component grain labels
  - ``<basename>.map.grod`` — grain-reference orientation deviation (radians)

When the optional ``<MicFileBinary>.AllMatches`` companion file exists,
its text counterpart ``<basename>.AllMatches`` is also produced.

All binary outputs are little-endian IEEE-754 doubles (``np.float64``)
matching the C ``fwrite`` byte order on x86-64 / ARM64.

Crystal symmetry comes from :mod:`midas_stress.orientation` (the
canonical port of MIDAS C's ``MakeSymmetries`` / ``GetMisOrientationAngle``
already in use by :mod:`midas_nf_fitorientation`).
"""
from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Per-voxel record layout of MicFileBinary (11 doubles, see C source):
#   [0] OrientationRowNr     [6] UpDown
#   [1] (NrMatches/unused)   [7] Eul1 (rad)
#   [2] (RunTime/unused)     [8] Eul2 (rad)
#   [3] X (µm)               [9] Eul3 (rad)
#   [4] Y (µm)               [10] Confidence (0 = invalid)
#   [5] TriEdgeSize (µm)
_NCOLS = 11
_RECORD_BYTES = _NCOLS * 8


@dataclass
class ParseMicParams:
    """Mirror of the C ``MicParams`` struct (NF_HEDM/src/ParseMic.c:26-35)."""
    PhaseNr: int
    NumPhases: int
    GlobalPosition: float
    inputfile: str                 # MicFileBinary path
    outputfile: str                # MicFileText basename
    nSaves: int = 1
    SGNr: int = 225
    GBAngle: float = 5.0           # degrees


# ---------------------------------------------------------------------------
#  Param-file reader (matches the C `ReadParameters` exactly)
# ---------------------------------------------------------------------------

def read_parsemic_params(param_file: str | Path) -> ParseMicParams:
    """Read the keys ``ReadParameters`` reads, in the same way it does:

    - ``strncmp(line, "Key ", len("Key "))`` style prefix match
    - one value per matching line, last occurrence wins (the C parser
      breaks on a match, but if the key appears multiple times the
      parser will overwrite — matches the FF/NF MIDAS_ParamParser
      behaviour the rest of the pipeline relies on).
    """
    p = {
        "PhaseNr": 0, "NumPhases": 0, "GlobalPosition": 0.0,
        "MicFileBinary": "", "MicFileText": "",
        "SaveNSolutions": 1, "SGNr": 225, "GBAngle": 5.0,
    }
    with open(param_file, "r") as f:
        for line in f:
            for key in p:
                prefix = key + " "
                if line.startswith(prefix):
                    val = line[len(prefix):].split()[0]
                    if isinstance(p[key], int):
                        p[key] = int(val)
                    elif isinstance(p[key], float):
                        p[key] = float(val)
                    else:
                        p[key] = val
                    break
    return ParseMicParams(
        PhaseNr=p["PhaseNr"],
        NumPhases=p["NumPhases"],
        GlobalPosition=p["GlobalPosition"],
        inputfile=p["MicFileBinary"],
        outputfile=p["MicFileText"],
        nSaves=p["SaveNSolutions"],
        SGNr=p["SGNr"],
        GBAngle=p["GBAngle"],
    )


# ---------------------------------------------------------------------------
#  Binary mic reader
# ---------------------------------------------------------------------------

def read_mic_binary(path: str | Path) -> np.ndarray:
    """Read ``MicFileBinary`` into ``(N, 11)`` float64 array."""
    arr = np.fromfile(str(path), dtype=np.float64)
    if arr.size % _NCOLS != 0:
        raise ValueError(
            f"{path}: byte count {arr.size * 8} is not a multiple of "
            f"{_RECORD_BYTES}"
        )
    return arr.reshape(-1, _NCOLS)


# ---------------------------------------------------------------------------
#  Output 1: text .mic file
# ---------------------------------------------------------------------------

def write_mic_text(out_path: str | Path, mic: np.ndarray, params: ParseMicParams) -> None:
    """Replicate ``WriteMicText`` (NF_HEDM/src/ParseMic.c:153-181).

    Header (3 ``%`` lines + 1 column-name ``%`` line):

        %TriEdgeSize <TriEdgeSize>      ← from row 0 col 5, ``%lf``
        %NumPhases <NumPhases>          ← ``%d``
        %GlobalPosition <GlobalPosition>← ``%lf``
        %OrientationRowNr\\tOrientationID\\tRunTime\\tX\\tY\\tTriEdgeSize\\tUpDown\\t
        Eul1\\tEul2\\tEul3\\tConfidence\\tPhaseNr

    Per-voxel rows: skip if ``Confidence == 0``; otherwise

        %lf<TAB> × 11   then   %d<NEWLINE>   for PhaseNr.

    C ``%lf`` and ``%f`` are aliases (printf widens float→double); both
    default to 6 decimal places — we mirror that with ``:.6f``.
    """
    n_rows = mic.shape[0]
    tri_edge = float(mic[0, 5]) if n_rows > 0 else 0.0
    with open(out_path, "w") as f:
        f.write(f"%TriEdgeSize {tri_edge:.6f}\n")
        f.write(f"%NumPhases {params.NumPhases}\n")
        f.write(f"%GlobalPosition {params.GlobalPosition:.6f}\n")
        f.write(
            "%OrientationRowNr\tOrientationID\tRunTime\tX\tY\tTriEdgeSize\t"
            "UpDown\tEul1\tEul2\tEul3\tConfidence\tPhaseNr\n"
        )
        for i in range(n_rows):
            if mic[i, 10] == 0:
                continue
            row = mic[i]
            for j in range(_NCOLS):
                f.write(f"{row[j]:.6f}\t")
            f.write(f"{params.PhaseNr}\n")


# ---------------------------------------------------------------------------
#  2D map rasterisation (irregular hex voxels → square pixel grid)
# ---------------------------------------------------------------------------

@dataclass
class MapDims:
    x_size: int
    y_size: int
    min_x: float
    min_y: float

    @property
    def size_map(self) -> int:
        return self.x_size * self.y_size


def _build_raster(mic: np.ndarray) -> tuple[MapDims, np.ndarray, np.ndarray]:
    """Replicate ``GenerateMap``'s pixel-assignment loop
    (NF_HEDM/src/ParseMic.c:214-290).

    Returns
    -------
    dims : :class:`MapDims`
    row_nr_mat : np.ndarray (size_map,) int64
        ``row_nr_mat[pixel] == i`` if pixel was claimed by voxel ``i``;
        ``-1`` if no voxel claimed it.
    length_mat : np.ndarray (size_map,) float64
        Distance from the claimed voxel to the pixel centre.
    """
    valid = mic[:, 10] != 0
    if not np.any(valid):
        # Empty input — nothing to raster.
        return MapDims(0, 0, 0.0, 0.0), np.zeros(0, dtype=np.int64), np.zeros(0)
    valid_x = mic[valid, 3]
    valid_y = mic[valid, 4]
    edge_size = float(mic[0, 5])

    min_x = float(valid_x.min()) - (edge_size + 25.0)
    max_x = float(valid_x.max()) + (edge_size + 25.0)
    min_y = float(valid_y.min()) - (edge_size + 25.0)
    max_y = float(valid_y.max()) + (edge_size + 25.0)

    x_size = int(math.ceil(max_x) - math.floor(min_x) + 1)
    y_size = int(math.ceil(max_y) - math.floor(min_y) + 1)
    size_map = x_size * y_size

    # The C uses `int(double)` truncation toward zero; numpy ``.astype(np.int64)``
    # also truncates toward zero, matching exactly.
    row_nr_mat = np.full(size_map, -1, dtype=np.int64)
    length_mat = np.zeros(size_map, dtype=np.float64)

    # Loop bound `j <= edge_size + 5`: with int j, C compares as float, so
    # j ranges from int(-(edge_size + 5)) to int(edge_size + 5) inclusive.
    half_window = int(edge_size + 5)
    half_window_neg = int(-(edge_size + 5))   # truncates toward zero on doubles

    n_rows = mic.shape[0]
    for i in range(n_rows):
        if mic[i, 10] == 0:
            continue
        x_i = mic[i, 3]
        y_i = mic[i, 4]
        int_x = int(x_i)            # truncates toward zero
        int_y = int(y_i)

        for j in range(half_window_neg, half_window + 1):
            pos_x = int(int_x + j - min_x)
            for k in range(half_window_neg, half_window + 1):
                pos_y = int(int_y + k - min_y)
                if pos_x < 0 or pos_x >= x_size or pos_y < 0 or pos_y >= y_size:
                    continue
                pos_this = pos_y * x_size + pos_x
                # ⚠ C-parity: ParseMic.c uses the unparenthesised macro
                # `CalcNorm2(a, b, c, d) sqrt((a-b)*(a-b) + (c-d)*(c-d))`.
                # Called as ``CalcNorm2(X, intX+j, Y, intY+k)`` it expands to
                # ``sqrt((X - intX + j)² + (Y - intY + k)²)``, NOT the
                # expected ``sqrt((X - (intX+j))² + …)``. We replicate the
                # exact arithmetic so the pixel→voxel assignment is byte-
                # identical with the C output.
                dy = x_i - int_x + j
                dx = y_i - int_y + k
                diff_len = math.sqrt(dy * dy + dx * dx)
                if row_nr_mat[pos_this] == -1 or diff_len < length_mat[pos_this]:
                    row_nr_mat[pos_this] = i
                    length_mat[pos_this] = diff_len

    return MapDims(x_size, y_size, min_x, min_y), row_nr_mat, length_mat


# ---------------------------------------------------------------------------
#  Orientation helpers — all delegate to midas_stress.orientation, the
#  byte-for-byte port of MIDAS C's GetMisorientation.h.
# ---------------------------------------------------------------------------

def _euler_zxz_to_om_batch(eulers: np.ndarray) -> np.ndarray:
    """Bunge ZXZ Euler (radians, ``(N, 3)``) → orient matrices ``(N, 3, 3)``."""
    from midas_stress.orientation import euler_to_orient_mat_batch
    eul = np.asarray(eulers, dtype=np.float64)
    if eul.ndim == 1:
        eul = eul[None, :]
    om_flat = np.asarray(euler_to_orient_mat_batch(eul), dtype=np.float64)
    return om_flat.reshape(-1, 3, 3)


def _pairwise_miso_om_rad(om_new: np.ndarray, om_existing: np.ndarray, sg_nr: int) -> np.ndarray:
    """One OM vs N OMs misorientation in radians, via
    :func:`midas_stress.orientation.misorientation_om_batch`.

    ``om_new`` is a single ``(3, 3)`` matrix; ``om_existing`` is ``(N, 3, 3)``.

    midas-stress dispatches to its **torch-vectorised** kernel when the
    inputs are torch tensors — that's ~100× faster than the pure-Python
    fallback for the per-pixel loops in KAM / BFS / GROD. We pass torch
    CPU tensors and convert the result back to numpy.
    """
    import torch
    from midas_stress.orientation import misorientation_om_batch
    om_e_t = torch.as_tensor(np.asarray(om_existing, dtype=np.float64).reshape(-1, 3, 3))
    om_n_t = torch.as_tensor(np.asarray(om_new, dtype=np.float64).reshape(3, 3))
    om_n_t = om_n_t.unsqueeze(0).expand_as(om_e_t).contiguous()
    angles = misorientation_om_batch(om_n_t, om_e_t, sg_nr)
    return angles.detach().cpu().numpy().astype(np.float64).ravel()


# ---------------------------------------------------------------------------
#  Output 2: .map binary (orientation raster)
# ---------------------------------------------------------------------------

def _build_map_array(
    dims: MapDims, row_nr_mat: np.ndarray, length_mat: np.ndarray,
    mic: np.ndarray, params: ParseMicParams,
) -> np.ndarray:
    """Replicate NF_HEDM/src/ParseMic.c:246-310.

    Initial fill is ``-15`` for the entire ``(size_map * 7 + 4)`` buffer,
    then the 4-element header is overwritten with
    ``(xSizeMap, ySizeMap, minXRange, minYRange)``, and assigned pixels
    get their 7 fields:

        field 0: Confidence (col 10)
        field 1: Eul1       (col 7)
        field 2: Eul2       (col 8)
        field 3: Eul3       (col 9)
        field 4: OrientationRowNr (col 0)
        field 5: PhaseNr (constant)
        field 6: distance from voxel centre (length_mat)
    """
    size_map = dims.size_map
    arr = np.full(size_map * 7 + 4, -15.0, dtype=np.float64)
    arr[0] = float(dims.x_size)
    arr[1] = float(dims.y_size)
    arr[2] = dims.min_x
    arr[3] = dims.min_y

    if size_map == 0:
        return arr
    assigned = row_nr_mat != -1
    if not np.any(assigned):
        return arr
    idxs = np.nonzero(assigned)[0]
    rows = row_nr_mat[idxs]

    arr[4 + idxs + 0 * size_map] = mic[rows, 10]
    arr[4 + idxs + 1 * size_map] = mic[rows, 7]
    arr[4 + idxs + 2 * size_map] = mic[rows, 8]
    arr[4 + idxs + 3 * size_map] = mic[rows, 9]
    arr[4 + idxs + 4 * size_map] = mic[rows, 0]
    arr[4 + idxs + 5 * size_map] = float(params.PhaseNr)
    arr[4 + idxs + 6 * size_map] = length_mat[idxs]
    return arr


# ---------------------------------------------------------------------------
#  KAM (kernel-average misorientation, radians)
# ---------------------------------------------------------------------------

# 8-neighbor offsets (3×3 minus centre), in C order (NF_HEDM/src/ParseMic.c:340-341).
_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)


def _batched_miso_om_rad(om_a: np.ndarray, om_b: np.ndarray, sg_nr: int) -> np.ndarray:
    """Batched ``misorientation_om_batch`` via midas-stress's torch path.

    Both inputs are ``(N, 3, 3)`` numpy; result is a ``(N,)`` numpy of
    angles in radians.
    """
    import torch
    from midas_stress.orientation import misorientation_om_batch
    a = torch.as_tensor(np.ascontiguousarray(om_a, dtype=np.float64))
    b = torch.as_tensor(np.ascontiguousarray(om_b, dtype=np.float64))
    return misorientation_om_batch(a, b, sg_nr).detach().cpu().numpy().astype(np.float64)


def _precompute_neighbor_misos(
    dims: MapDims, row_nr_mat: np.ndarray, mic: np.ndarray, sg_nr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the 8-neighbor misorientation grid in one batched call.

    Returns
    -------
    miso_grid : (size_map, 8) float64 — misorientation in **radians**
        (NaN where the neighbor is OOB or unassigned, so callers can
        easily filter with ``np.isfinite``).
    n_idx_grid : (size_map, 8) int64 — neighbor pixel index, ``-1`` if invalid.
    om_for_row : (n_unique_voxels, 3, 3)
        Cached OM per referenced voxel (used by the GROD pass too).
    """
    size_map = dims.size_map
    miso_grid = np.full((size_map, 8), np.nan, dtype=np.float64)
    n_idx_grid = np.full((size_map, 8), -1, dtype=np.int64)
    if size_map == 0:
        return miso_grid, n_idx_grid, np.zeros((0, 3, 3), dtype=np.float64)

    assigned_pixels = np.nonzero(row_nr_mat != -1)[0]
    if assigned_pixels.size == 0:
        return miso_grid, n_idx_grid, np.zeros((0, 3, 3), dtype=np.float64)

    rows_used = np.unique(row_nr_mat[assigned_pixels])
    om_for_row = _euler_zxz_to_om_batch(mic[rows_used, 7:10])
    row_to_idx_arr = np.full(int(rows_used.max()) + 1, -1, dtype=np.int64)
    row_to_idx_arr[rows_used] = np.arange(len(rows_used))

    # Build per-pixel neighbor indices (size_map, 8).
    cx_all = np.arange(size_map, dtype=np.int64) % dims.x_size
    cy_all = np.arange(size_map, dtype=np.int64) // dims.x_size
    nx = cx_all[:, None] + _DX[None, :]                       # (size_map, 8)
    ny = cy_all[:, None] + _DY[None, :]
    in_bounds = (nx >= 0) & (nx < dims.x_size) & (ny >= 0) & (ny < dims.y_size)
    nidx_raw = ny * dims.x_size + nx
    nidx_raw[~in_bounds] = -1

    # Mask: pixel itself assigned AND neighbor in bounds AND neighbor assigned.
    self_assigned = (row_nr_mat != -1)[:, None]
    valid = in_bounds & self_assigned
    flat_n = nidx_raw.copy()
    # Mark neighbors that point to unassigned pixels as invalid.
    flat_n[valid] = nidx_raw[valid]
    nb_rows = np.where(valid, row_nr_mat[flat_n.clip(min=0)], -1)
    valid_pair = valid & (nb_rows != -1)

    n_idx_grid[valid_pair] = flat_n[valid_pair]

    # Flatten to (M, 2) pixel/neighbor pairs and compute miso in one shot.
    pix_idx, slot_idx = np.nonzero(valid_pair)
    if pix_idx.size == 0:
        return miso_grid, n_idx_grid, om_for_row
    self_rows = row_nr_mat[pix_idx]
    nb_rows_flat = nb_rows[pix_idx, slot_idx]
    om_self = om_for_row[row_to_idx_arr[self_rows]]
    om_nb = om_for_row[row_to_idx_arr[nb_rows_flat]]
    misos = _batched_miso_om_rad(om_self, om_nb, sg_nr)
    miso_grid[pix_idx, slot_idx] = misos

    return miso_grid, n_idx_grid, om_for_row


def _build_kam_array(
    dims: MapDims, row_nr_mat: np.ndarray, mic: np.ndarray,
    params: ParseMicParams, *, miso_grid: np.ndarray | None = None,
) -> np.ndarray:
    """KAM: per assigned pixel, mean misorientation (radians) to its
    assigned 8-neighbors. Uses the precomputed ``miso_grid`` from
    :func:`_precompute_neighbor_misos` to avoid re-doing the misorientation
    work per pixel."""
    size_map = dims.size_map
    arr = np.zeros(size_map + 4, dtype=np.float64)
    arr[0] = float(dims.x_size)
    arr[1] = float(dims.y_size)
    arr[2] = dims.min_x
    arr[3] = dims.min_y
    if size_map == 0 or miso_grid is None:
        return arr

    finite = np.isfinite(miso_grid)
    counts = finite.sum(axis=1)
    sums = np.where(finite, miso_grid, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kam = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    arr[4:] = kam
    return arr


# ---------------------------------------------------------------------------
#  Grain-id BFS connected components
# ---------------------------------------------------------------------------

def _build_grain_id_array(
    dims: MapDims, row_nr_mat: np.ndarray, mic: np.ndarray,
    params: ParseMicParams, *,
    miso_grid: np.ndarray | None = None,
    n_idx_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Replicate NF_HEDM/src/ParseMic.c:404-465.

    BFS in 8-neighborhood; an edge is added when misorientation between
    pixels' voxels ≤ ``GBAngle * π/180`` rad. Uses precomputed
    ``miso_grid`` and ``n_idx_grid`` from
    :func:`_precompute_neighbor_misos`.
    """
    size_map = dims.size_map
    arr = np.zeros(size_map + 4, dtype=np.float64)
    arr[0] = float(dims.x_size)
    arr[1] = float(dims.y_size)
    arr[2] = dims.min_x
    arr[3] = dims.min_y

    if size_map == 0:
        return arr, 0

    threshold_rad = params.GBAngle * (math.pi / 180.0)

    if size_map == 0 or miso_grid is None:
        return arr, 0
    if not np.any(row_nr_mat != -1):
        return arr, 0

    grain_id = np.zeros(size_map, dtype=np.int64)
    current_id = 1
    edge_below_threshold = (miso_grid <= threshold_rad) & np.isfinite(miso_grid)

    for seed in range(size_map):
        if row_nr_mat[seed] == -1 or grain_id[seed] != 0:
            continue
        grain_id[seed] = current_id
        queue: deque[int] = deque([seed])
        while queue:
            cur = queue.popleft()
            slots_under = np.nonzero(edge_below_threshold[cur])[0]
            for slot in slots_under:
                n_idx = int(n_idx_grid[cur, slot])
                if n_idx == -1 or grain_id[n_idx] != 0:
                    continue
                grain_id[n_idx] = current_id
                queue.append(n_idx)
        current_id += 1

    arr[4:] = grain_id.astype(np.float64)
    return arr, current_id - 1


# ---------------------------------------------------------------------------
#  GROD (grain-reference orientation deviation, radians)
# ---------------------------------------------------------------------------

def _build_grod_array(
    dims: MapDims, row_nr_mat: np.ndarray, grain_id_arr: np.ndarray,
    mic: np.ndarray, params: ParseMicParams, total_grains: int,
    *, om_for_row: np.ndarray | None = None,
) -> np.ndarray:
    size_map = dims.size_map
    arr = np.zeros(size_map + 4, dtype=np.float64)
    arr[0] = float(dims.x_size)
    arr[1] = float(dims.y_size)
    arr[2] = dims.min_x
    arr[3] = dims.min_y

    if size_map == 0 or total_grains == 0:
        return arr

    sg_nr = params.SGNr
    grain_id = grain_id_arr[4:].astype(np.int64)

    assigned = (grain_id != 0) & (row_nr_mat != -1)
    if not np.any(assigned):
        return arr
    idxs = np.nonzero(assigned)[0]
    g_ids = grain_id[idxs]
    rows = row_nr_mat[idxs]
    confs = mic[rows, 10]

    # Per-grain reference voxel = highest-confidence pixel in the grain.
    best_pix_for_grain = np.full(total_grains, -1, dtype=np.int64)
    best_conf = np.full(total_grains, -np.inf, dtype=np.float64)
    for pix, gid, conf in zip(idxs, g_ids, confs):
        gid0 = int(gid) - 1
        if conf > best_conf[gid0]:
            best_conf[gid0] = conf
            best_pix_for_grain[gid0] = int(row_nr_mat[pix])

    valid_grains = best_pix_for_grain >= 0
    if not np.any(valid_grains):
        return arr

    ref_om = np.zeros((total_grains, 3, 3), dtype=np.float64)
    ref_om[valid_grains] = _euler_zxz_to_om_batch(
        mic[best_pix_for_grain[valid_grains], 7:10]
    )

    # Reuse cached om_for_row if available; else build it.
    rows_used = np.unique(row_nr_mat[idxs])
    if om_for_row is None or om_for_row.shape[0] == 0:
        om_for_row = _euler_zxz_to_om_batch(mic[rows_used, 7:10])
    row_to_idx_arr = np.full(int(rows_used.max()) + 1, -1, dtype=np.int64)
    row_to_idx_arr[rows_used] = np.arange(len(rows_used))

    # Build the (M, 3, 3) per-pixel + (M, 3, 3) per-pixel-grain-ref tensors,
    # call the batched midas-stress misorientation in one shot.
    keep = valid_grains[g_ids - 1]
    if not np.any(keep):
        return arr
    pix_keep = idxs[keep]
    g_keep = g_ids[keep]
    om_pix = om_for_row[row_to_idx_arr[row_nr_mat[pix_keep]]]
    om_ref = ref_om[g_keep - 1]
    misos = _batched_miso_om_rad(om_pix, om_ref, sg_nr)
    arr[4 + pix_keep] = misos
    return arr


# ---------------------------------------------------------------------------
#  AllMatches text writer
# ---------------------------------------------------------------------------

def _write_all_matches(
    mic: np.ndarray, params: ParseMicParams,
) -> Optional[Path]:
    """Replicate ``ProcessAllMatches`` (NF_HEDM/src/ParseMic.c:558-623).

    Reads ``<MicFileBinary>.AllMatches``, writes ``<MicFileText>.AllMatches``.
    The companion file's record width is ``7 + 4 * nSaves`` doubles per row.
    Returns the output path on success, ``None`` if the input is missing.
    """
    in_path = Path(str(params.inputfile) + ".AllMatches")
    if not in_path.exists():
        return None
    out_path = Path(str(params.outputfile) + ".AllMatches")
    n_cols = 7 + 4 * params.nSaves
    arr = np.fromfile(str(in_path), dtype=np.float64)
    if arr.size % n_cols != 0:
        # Mismatch → mirror the C: we just bail rather than half-writing.
        return None
    rows = arr.reshape(-1, n_cols)

    tri_edge = float(mic[0, 5]) if mic.shape[0] > 0 else 0.0
    with open(out_path, "w") as f:
        f.write(f"%TriEdgeSize {tri_edge:.6f}\n")
        f.write(f"%NumPhases {params.NumPhases}\n")
        f.write(f"%GlobalPosition {params.GlobalPosition:.6f}\n")
        f.write(
            "%OrientationRowNr\tNrMatches\tRunTime\tX\tY\tTriEdgeSize\t"
            "UpDown\tEul1\tEul2\tEul3\tConfidence"
            + "\t...\t...\t...\t...\t...\t...\tPhaseNr\n"
        )
        for i in range(rows.shape[0]):
            if rows[i, 10] == 0:
                continue
            for j in range(n_cols):
                f.write(f"{rows[i, j]:.6f}\t")
            f.write(f"{params.PhaseNr}\n")
    return out_path


# ---------------------------------------------------------------------------
#  Top-level entry point
# ---------------------------------------------------------------------------

def parse_mic(params: ParseMicParams) -> dict:
    """Run the full ParseMic pipeline. Returns a dict of output paths."""
    mic = read_mic_binary(params.inputfile)
    n_rows = mic.shape[0]
    if n_rows == 0:
        return {}

    out_paths: dict = {}

    out_paths["mic"] = Path(params.outputfile)
    write_mic_text(out_paths["mic"], mic, params)

    dims, row_nr_mat, length_mat = _build_raster(mic)

    map_arr = _build_map_array(dims, row_nr_mat, length_mat, mic, params)
    out_paths["map"] = Path(str(params.outputfile) + ".map")
    map_arr.astype(np.float64, copy=False).tofile(out_paths["map"])

    # Precompute the (size_map, 8) neighbor misorientation grid once; KAM
    # averages over it, BFS thresholds it. Single batched midas-stress
    # call instead of a per-pixel Python loop.
    miso_grid, n_idx_grid, om_for_row = _precompute_neighbor_misos(
        dims, row_nr_mat, mic, params.SGNr,
    )

    kam_arr = _build_kam_array(dims, row_nr_mat, mic, params, miso_grid=miso_grid)
    out_paths["kam"] = Path(str(params.outputfile) + ".map.kam")
    kam_arr.astype(np.float64, copy=False).tofile(out_paths["kam"])

    grain_id_arr, total_grains = _build_grain_id_array(
        dims, row_nr_mat, mic, params,
        miso_grid=miso_grid, n_idx_grid=n_idx_grid,
    )
    out_paths["grainId"] = Path(str(params.outputfile) + ".map.grainId")
    grain_id_arr.astype(np.float64, copy=False).tofile(out_paths["grainId"])

    grod_arr = _build_grod_array(
        dims, row_nr_mat, grain_id_arr, mic, params, total_grains,
        om_for_row=om_for_row,
    )
    out_paths["grod"] = Path(str(params.outputfile) + ".map.grod")
    grod_arr.astype(np.float64, copy=False).tofile(out_paths["grod"])

    am_path = _write_all_matches(mic, params)
    if am_path is not None:
        out_paths["allMatches"] = am_path

    return out_paths


def parse_mic_from_paramfile(param_file: str | Path) -> dict:
    """Convenience: do everything ``ParseMic <param_file>`` does."""
    return parse_mic(read_parsemic_params(param_file))
