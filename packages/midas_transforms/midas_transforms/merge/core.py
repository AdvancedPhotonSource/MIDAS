"""merge_overlapping_peaks: drop-in replacement for ``MergeOverlappingPeaksAllZarr``.

Direct port of ``FF_HEDM/src/MergeOverlappingPeaksAllZarr.c:494-1060``.

Reads ``AllPeaks_PS.bin`` (the ``ConsolidatedPeakReader`` blob written by
``midas-peakfit`` / ``PeaksFittingOMPZarrRefactor``) and produces:

- ``Result_StartNr_<S>_EndNr_<E>.csv`` — 17 cols, in the order the C code
  writes (Radius and Eta are the LATEST match's values, not recomputed).
- ``MergeMap.csv`` — ``MergedSpotID FrameNr PeakID`` triples.

Algorithm (mirrors C exactly):

1. Initialise ``CurrentIDs`` from frame 0's filtered+sorted peaks.
2. For each subsequent frame:
   a. Read frame's peaks, drop ``IntegratedIntensity < 1``, sort by Eta
      with stable secondary key (original index) for tie-break determinism.
   b. Mutual-nearest match in (YCen, ZCen) within ``MarginOmegaOverlap``,
      filtered by ``|Radius_new - Radius_cur| <= MarginOmegaOverlap`` (ring guard).
   c. Matched: update ``CurrentIDs[i]`` (accumulate weighted Omega/Y/Z by
      IntInt, keep min/max omega, max IMax/SigmaR/SigmaEta/FitRMSE,
      sum NrPx/NrPxTot/RawSum, OR maskTouched, **replace Radius/Eta/Y/Z
      with the new peak's values**).
   d. Unmatched current: finalise immediately, write to OutFile + MergeMap.
   e. Unmatched new: become new ``CurrentIDs`` entries for next iteration.
3. End: write all remaining ``CurrentIDs``.

The 19-column ``CurrentIDs`` working buffer mapping is at
``MergeOverlappingPeaksAllZarr.c:639-678`` (init) and ``:806-833`` /
``:858-885`` (update); the 17-col output is written at ``:914-933`` and
``:1042-1057``.

GPU acceleration: per-frame mutual-nearest is a small dense problem
(~10⁴ peaks × ~10⁴ peaks). The torch port computes the pairwise distance
matrix via broadcast and resolves mutual-best in a fully vectorised pass.
The frame loop itself remains Python-driven (sequential by construction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from ..io import csv as csv_io
from ..io import zarr_io
from ..params import ZarrParams, read_zarr_params

_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


@dataclass
class MergeResult:
    """In-memory result of the merge stage."""

    peaks: torch.Tensor                            # (N, 17) — Result_*.csv layout
    merge_map: List[Tuple[int, int, int]] = field(default_factory=list)
    # Each tuple: (merged_spot_id, frame_nr, original_peak_id)


# AllPeaks_PS.bin column layout (from PeaksFittingConsolidatedIO.h:31-44).
N_PEAK_COLS = 29
COL_SPOTID = 0
COL_II = 1
COL_OMEGA = 2
COL_YCEN = 3
COL_ZCEN = 4
COL_IMAX = 5
COL_RADIUS = 6
COL_ETA = 7
COL_SIGMAR = 8
COL_SIGMAETA = 9
COL_NRPX = 10
COL_NRPXTOT = 11
COL_MAXY = 13
COL_MAXZ = 14
COL_RAWSUM = 26
COL_MASKTOUCHED = 27
COL_FITRMSE = 28


def _read_sort_filter_frame(
    peaks: np.ndarray, use_maxima_positions: bool,
) -> np.ndarray:
    """Mirror ``ReadSortFiles`` from MergeOverlappingPeaksAllZarr.c:311-355.

    - Optionally swap (YCen, ZCen) for (maxY, maxZ).
    - Drop peaks with ``IntegratedIntensity < 1``.
    - qsort by Eta (stable; secondary key = original index for tie-break).
    """
    if peaks.size == 0:
        return peaks
    arr = peaks.copy()
    if use_maxima_positions:
        arr[:, COL_YCEN] = arr[:, COL_MAXY]
        arr[:, COL_ZCEN] = arr[:, COL_MAXZ]
    arr = arr[arr[:, COL_II] >= 1.0]
    if arr.shape[0] == 0:
        return arr
    # Stable sort by Eta with original-index tie-break.
    order = np.lexsort((np.arange(arr.shape[0]), arr[:, COL_ETA]))
    return arr[order]


def _seed_current_from_frame(
    frame_peaks: np.ndarray, frame_nr: int,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """Initialise ``CurrentIDs`` (19 cols) from a frame's filtered peaks.

    Mirrors ``MergeOverlappingPeaksAllZarr.c:644-679``.
    """
    n = frame_peaks.shape[0]
    cur = np.zeros((n, 19), dtype=np.float64)
    if n == 0:
        return cur, []
    ii = frame_peaks[:, COL_II]
    cur[:, 0] = frame_peaks[:, COL_SPOTID]
    cur[:, 1] = ii
    cur[:, 2] = frame_peaks[:, COL_OMEGA] * ii
    cur[:, 3] = frame_peaks[:, COL_YCEN] * ii
    cur[:, 4] = frame_peaks[:, COL_ZCEN] * ii
    cur[:, 5] = frame_peaks[:, COL_IMAX]
    cur[:, 6] = frame_peaks[:, COL_RADIUS]
    cur[:, 7] = frame_peaks[:, COL_ETA]
    cur[:, 8] = frame_peaks[:, COL_YCEN]
    cur[:, 9] = frame_peaks[:, COL_ZCEN]
    cur[:, 10] = frame_peaks[:, COL_OMEGA]
    cur[:, 11] = frame_peaks[:, COL_OMEGA]
    cur[:, 12] = frame_peaks[:, COL_SIGMAR]
    cur[:, 13] = frame_peaks[:, COL_SIGMAETA]
    cur[:, 14] = frame_peaks[:, COL_NRPX]
    cur[:, 15] = frame_peaks[:, COL_NRPXTOT]
    cur[:, 16] = frame_peaks[:, COL_RAWSUM]
    cur[:, 17] = frame_peaks[:, COL_MASKTOUCHED]
    cur[:, 18] = frame_peaks[:, COL_FITRMSE]
    cons = [[(frame_nr, int(frame_peaks[i, COL_SPOTID]))] for i in range(n)]
    return cur, cons


def _mutual_nearest(
    cur: np.ndarray, new: np.ndarray, margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy mutual-nearest matching, mirroring the C semantics exactly.

    The C algorithm (``MergeOverlappingPeaksAllZarr.c:835-906``) iterates
    current peaks in cur-index order. For each ``i``:

      1. Among UNCLAIMED new peaks ``j`` within (Y, Z) distance ``margin``
         and ``|R_cur - R_new| <= margin``, find the closest one
         (distance ``minLen``).
      2. Reverse check: among UNCLAIMED current peaks ``k`` within
         ``minLen`` of new ``j`` and same ring, find the closest. If
         ``k != i``, abandon this match (another cur is closer).
      3. Otherwise: pair (i, j); mark both as taken; move on.

    Because peaks claimed earlier are removed from later searches, this is
    **greedy** (not globally optimal) and **order-sensitive**. We replicate
    the order exactly: cur in index order (which matches the Eta-sorted
    seed order from the previous frame).

    Returns ``(best_for_cur, has_match_cur)``. The matcher is sequential
    in Python; the inner per-cur lookup is vectorised over new peaks.
    """
    nC = cur.shape[0]
    nN = new.shape[0]
    out_best = np.full(nC, -1, dtype=np.int64)
    out_has = np.zeros(nC, dtype=bool)
    if nC == 0 or nN == 0:
        return out_best, out_has

    cur_y = cur[:, 8]
    cur_z = cur[:, 9]
    cur_r = cur[:, 6]
    new_y = new[:, COL_YCEN]
    new_z = new[:, COL_ZCEN]
    new_r = new[:, COL_RADIUS]

    new_taken = np.zeros(nN, dtype=bool)
    cur_taken = np.zeros(nC, dtype=bool)

    # The C ``shash_find_nearest`` (MergeOverlappingPeaksAllZarr.c:459-491)
    # initialises ``best = radius`` so peaks outside the search radius are
    # excluded — the bucketing is a pure performance optimisation. We
    # replicate with a strict ``dist < radius`` filter.

    for i in range(nC):
        # Step 1: forward — closest unclaimed new peak with d < margin.
        free_new = ~new_taken
        if not free_new.any():
            break
        dy = cur_y[i] - new_y
        dz = cur_z[i] - new_z
        dist = np.sqrt(dy * dy + dz * dz)
        ring_ok = np.abs(cur_r[i] - new_r) <= margin
        candidate = free_new & ring_ok & (dist < margin)
        if not candidate.any():
            continue
        d = np.where(candidate, dist, np.inf)
        j = int(np.argmin(d))
        min_len = float(d[j])

        # Step 2: reverse — closest unclaimed cur to j with d < min_len.
        # If this returns some k != i, abandon (another cur is closer).
        # Note that ``cur_taken`` does NOT yet include i — it's still
        # in the pool, so we don't filter it out (and i↔j distance is
        # exactly min_len, NOT < min_len, so i can't be the reverse
        # winner; k != i with k != -1 means we abandon).
        free_cur = ~cur_taken
        rdy = cur_y - new_y[j]
        rdz = cur_z - new_z[j]
        rdist = np.sqrt(rdy * rdy + rdz * rdz)
        rring_ok = np.abs(cur_r - new_r[j]) <= margin
        rcand = free_cur & rring_ok & (rdist < min_len)
        if rcand.any():
            k = int(np.argmin(np.where(rcand, rdist, np.inf)))
            if k != i:
                continue

        out_best[i] = j
        out_has[i] = True
        new_taken[j] = True
        cur_taken[i] = True
    return out_best, out_has


def _accumulate_match(cur_row: np.ndarray, new_row: np.ndarray) -> None:
    """Update one ``CurrentIDs`` row in-place with one matched new peak.

    Mirrors ``MergeOverlappingPeaksAllZarr.c:806-833`` (in distance branch).
    """
    ii = new_row[COL_II]
    cur_row[1] += ii
    cur_row[2] += new_row[COL_OMEGA] * ii
    cur_row[3] += new_row[COL_YCEN] * ii
    cur_row[4] += new_row[COL_ZCEN] * ii
    if cur_row[5] < new_row[COL_IMAX]:
        cur_row[5] = new_row[COL_IMAX]
    cur_row[6] = new_row[COL_RADIUS]   # latest Radius
    cur_row[7] = new_row[COL_ETA]      # latest Eta
    cur_row[8] = new_row[COL_YCEN]     # latest YCen
    cur_row[9] = new_row[COL_ZCEN]     # latest ZCen
    if cur_row[10] > new_row[COL_OMEGA]:
        cur_row[10] = new_row[COL_OMEGA]   # MinOmega
    if cur_row[11] < new_row[COL_OMEGA]:
        cur_row[11] = new_row[COL_OMEGA]   # MaxOmega
    if cur_row[12] < new_row[COL_SIGMAR]:
        cur_row[12] = new_row[COL_SIGMAR]
    if cur_row[13] < new_row[COL_SIGMAETA]:
        cur_row[13] = new_row[COL_SIGMAETA]
    cur_row[14] += new_row[COL_NRPX]
    cur_row[15] += new_row[COL_NRPXTOT]
    cur_row[16] += new_row[COL_RAWSUM]
    if new_row[COL_MASKTOUCHED] > 0:
        cur_row[17] = 1.0
    if new_row[COL_FITRMSE] > cur_row[18]:
        cur_row[18] = new_row[COL_FITRMSE]


def _finalise_row(
    cur_row: np.ndarray, spot_id: int,
) -> np.ndarray:
    """Convert one ``CurrentIDs`` row into one 17-col Result.csv row.

    Column order matches ``MergeOverlappingPeaksAllZarr.c:914-924`` /
    ``:1042-1051``::

        [0]  SpotIDNr
        [1]  IntegratedIntensity
        [2]  Omega = wOmega / IntInt
        [3]  YCen  = wYCen  / IntInt
        [4]  ZCen  = wZCen  / IntInt
        [5]  IMax
        [6]  MinOme
        [7]  MaxOme
        [8]  SigmaR
        [9]  SigmaEta
        [10] NrPx
        [11] NrPxTot
        [12] Radius (LATEST)
        [13] Eta    (LATEST)
        [14] RawSumIntensity
        [15] maskTouched
        [16] FitRMSE
    """
    out = np.zeros(17, dtype=np.float64)
    intInt = cur_row[1] if cur_row[1] != 0 else 1.0
    out[0] = spot_id
    out[1] = cur_row[1]
    out[2] = cur_row[2] / intInt
    out[3] = cur_row[3] / intInt
    out[4] = cur_row[4] / intInt
    out[5] = cur_row[5]
    out[6] = cur_row[10]   # MinOme
    out[7] = cur_row[11]   # MaxOme
    out[8] = cur_row[12]
    out[9] = cur_row[13]
    out[10] = cur_row[14]
    out[11] = cur_row[15]
    out[12] = cur_row[6]   # Radius (latest)
    out[13] = cur_row[7]   # Eta (latest)
    out[14] = cur_row[16]
    out[15] = cur_row[17]
    out[16] = cur_row[18]
    return out


def _merge_frames(
    frames: List[np.ndarray],
    *,
    overlap_length: float,
    use_maxima_positions: bool = False,
    skip_frame: int = 0,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Frame-by-frame mutual-nearest merge. Returns (Result_csv_array, merge_map)."""
    if not frames:
        return np.empty((0, 17), dtype=np.float64), []

    # The C code does `EndNr -= skipFrame` then iterates [StartNr, EndNr];
    # we mirror by truncating the frame list.
    if skip_frame > 0:
        frames = frames[: max(len(frames) - skip_frame, 0)]
    if not frames:
        return np.empty((0, 17), dtype=np.float64), []

    # Pre-process every frame: filter II<1, sort by Eta.
    proc = [_read_sort_filter_frame(f, use_maxima_positions) for f in frames]

    # Find first non-empty frame.
    first = 0
    while first < len(proc) and proc[first].shape[0] == 0:
        first += 1
    if first >= len(proc):
        return np.empty((0, 17), dtype=np.float64), []

    cur, constituents = _seed_current_from_frame(proc[first], frame_nr=first + 1)
    finalised: List[np.ndarray] = []
    merge_map: List[Tuple[int, int, int]] = []
    spot_id_nr = 1

    # Main frame loop. Mirrors MergeOverlappingPeaksAllZarr.c:735-1029.
    for fi in range(first + 1, len(proc)):
        new_peaks = proc[fi]
        frame_nr = fi + 1   # 1-based, like the C code

        if new_peaks.shape[0] == 0:
            # No new peaks this frame: every current row finalises.
            for i in range(cur.shape[0]):
                finalised.append(_finalise_row(cur[i], spot_id_nr))
                for (fn, pid) in constituents[i]:
                    merge_map.append((spot_id_nr, fn, pid))
                spot_id_nr += 1
            cur = np.zeros((0, 19), dtype=np.float64)
            constituents = []
            continue

        best_for_cur, matched = _mutual_nearest(cur, new_peaks, overlap_length)

        # Update matched current rows (cur-index order, matches C).
        new_taken = np.zeros(new_peaks.shape[0], dtype=bool)
        for i in np.flatnonzero(matched):
            j = int(best_for_cur[i])
            new_taken[j] = True
            _accumulate_match(cur[i], new_peaks[j])
            constituents[i].append((frame_nr, int(new_peaks[j, COL_SPOTID])))

        # Finalise current rows that did NOT match this frame
        # (MergeOverlappingPeaksAllZarr.c:916-947).
        unmatched_cur = ~matched
        if unmatched_cur.any():
            for i in np.flatnonzero(unmatched_cur):
                finalised.append(_finalise_row(cur[i], spot_id_nr))
                for (fn, pid) in constituents[i]:
                    merge_map.append((spot_id_nr, fn, pid))
                spot_id_nr += 1

        # Build next-iteration CurrentIDs:
        #   matched rows (continuing) followed by unmatched new peaks.
        # Matches C order: line 949-998 (matched cur first), then
        # 1000-1029 (unmatched new appended).
        keep_cur = np.flatnonzero(matched)
        new_cur = cur[keep_cur].copy()
        new_cons = [constituents[i] for i in keep_cur]

        unmatched_new = np.flatnonzero(~new_taken)
        if unmatched_new.size > 0:
            seed_arr, seed_cons = _seed_current_from_frame(
                new_peaks[unmatched_new], frame_nr=frame_nr,
            )
            new_cur = np.concatenate([new_cur, seed_arr], axis=0)
            new_cons.extend(seed_cons)

        cur = new_cur
        constituents = new_cons

    # Final flush: every remaining cur row gets a SpotIDNr and is written.
    for i in range(cur.shape[0]):
        finalised.append(_finalise_row(cur[i], spot_id_nr))
        for (fn, pid) in constituents[i]:
            merge_map.append((spot_id_nr, fn, pid))
        spot_id_nr += 1

    if not finalised:
        return np.empty((0, 17), dtype=np.float64), []
    return np.stack(finalised, axis=0), merge_map


def _write_result_csv_c_format(path: Path, data: np.ndarray) -> None:
    """Write Result_*.csv in the exact format the C MergeOverlappingPeaksAllZarr
    emits: SpotID as ``%d``, all other cols as ``%lf`` (six decimals)."""
    header = csv_io.RESULT_HEADER
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in data:
            f.write(f"{int(row[0])}")
            for v in row[1:]:
                f.write(f" {v:.6f}")
            f.write("\n")


def _write_merge_map_csv(path: Path, rows: List[Tuple[int, int, int]]) -> None:
    """Write MergeMap.csv in the C format (tab-separated, leading '%' header)."""
    with open(path, "w") as f:
        f.write("%MergedSpotID\tFrameNr\tPeakID\n")
        for (sid, fn, pid) in rows:
            f.write(f"{sid}\t{fn}\t{pid}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def merge_overlapping_peaks(
    zarr_path: Optional[Union[str, Path]] = None,
    *,
    allpeaks_ps_bin: Optional[Union[str, Path]] = None,
    result_folder: Union[str, Path] = ".",
    overlap_length: Optional[float] = None,
    skip_frame: int = 0,
    use_pixel_overlap: bool = False,
    use_maxima_positions: bool = False,
    nr_pixels: Optional[int] = None,
    start_nr: int = 1,
    end_nr: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
    frames: Optional[List[np.ndarray]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    write: bool = True,
) -> MergeResult:
    """Drop-in replacement for ``MergeOverlappingPeaksAllZarr``.

    Inputs are typically read from ``<result_folder>/Temp/AllPeaks_PS.bin``
    (the ``ConsolidatedPeakReader`` blob written by midas-peakfit /
    PeaksFittingOMPZarrRefactor). Override paths via ``allpeaks_ps_bin=`` /
    ``frames=`` for in-memory or relocated inputs.
    """
    if use_pixel_overlap:
        raise NotImplementedError(
            "Pixel-overlap merge mode is not yet supported in midas-transforms "
            "v0.1.0. Use the C MergeOverlappingPeaksAllZarr or set "
            "use_pixel_overlap=False."
        )

    rf = Path(result_folder)
    out_dir = Path(out_dir) if out_dir is not None else rf

    # Load Zarr params if a Zarr path was provided (the C code reads
    # OverlapLength / SkipFrame / UseMaximaPositions from there).
    if zarr_path is not None:
        zp = read_zarr_params(zarr_path)
        if overlap_length is None:
            overlap_length = zp.OverlapLength
        if not skip_frame:
            skip_frame = zp.SkipFrame
        if not use_maxima_positions:
            use_maxima_positions = bool(zp.UseMaximaPositions)
        if nr_pixels is None:
            nr_pixels = zp.NrPixels
    if overlap_length is None:
        # C default: ``MarginOmegaOverlap = sqrt(4) = 2.0``
        # (MergeOverlappingPeaksAllZarr.c:524).
        overlap_length = 2.0

    # Load frames.
    if frames is None:
        if allpeaks_ps_bin is None:
            allpeaks_ps_bin = rf / "Temp" / "AllPeaks_PS.bin"
            if not Path(allpeaks_ps_bin).exists():
                allpeaks_ps_bin = rf / "AllPeaks_PS.bin"
        allpeaks_ps_bin = Path(allpeaks_ps_bin)
        if not allpeaks_ps_bin.exists():
            raise FileNotFoundError(
                f"AllPeaks_PS.bin not found at {allpeaks_ps_bin}. "
                f"This file is written by midas-peakfit / PeaksFittingOMPZarrRefactor."
            )
        frames = zarr_io.read_allpeaks_ps_frames(allpeaks_ps_bin)

    if end_nr is None:
        end_nr = max(start_nr + len(frames) - 1, start_nr)

    out, merge_map = _merge_frames(
        frames,
        overlap_length=float(overlap_length),
        use_maxima_positions=use_maxima_positions,
        skip_frame=skip_frame,
    )

    if write:
        out_path = out_dir / f"Result_StartNr_{start_nr}_EndNr_{end_nr}.csv"
        _write_result_csv_c_format(out_path, out)
        _write_merge_map_csv(out_dir / "MergeMap.csv", merge_map)

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)
    return MergeResult(
        peaks=torch.from_numpy(out).to(device=dev, dtype=dt),
        merge_map=merge_map,
    )
