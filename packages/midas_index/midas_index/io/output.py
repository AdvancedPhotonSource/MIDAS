"""Indexer block output writers.

Mirrors `WriteBestMatchBin` from `FF_HEDM/src/IndexerOMP.c:1613`. Output
is a pair of pre-allocated, pwrite-addressed binary files:

  IndexBest.bin       [N_total_seeds, 15]     float64  (per-seed best result)
  IndexBestFull.bin   [N_total_seeds, MAX_N_HKLS, 2]  float64  (matched-spot lookup)

Layout of the 15 doubles in IndexBest.bin (per IndexerOMP.c:1620-1628):

    [0]      avg_ia
    [1..9]   orientation matrix flat row-major (O11..O33)
    [10..12] best position (ga, gb, gc)
    [13]     n_matches (stored as double)
    [14]     n_t_spots / score encoding

The seed's slot is `offset_loc`, which is its row in `SpotsToIndex.csv`.
Empty slots (no result) read as zeros (the file is `ftruncate`-zeroed at
block_nr == 0; subsequent blocks just pwrite their own slots).

`MAX_N_HKLS = 5000` is fixed by the legacy layout (see compute.constants).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..compute.constants import MAX_N_HKLS

if TYPE_CHECKING:
    from ..result import IndexerResult, SeedResult

INDEX_BEST_RECORD_DOUBLES = 15
INDEX_BEST_RECORD_BYTES = INDEX_BEST_RECORD_DOUBLES * 8

INDEX_BEST_FULL_RECORD_DOUBLES = MAX_N_HKLS * 2
INDEX_BEST_FULL_RECORD_BYTES = INDEX_BEST_FULL_RECORD_DOUBLES * 8


def open_output_files(
    output_folder: str | Path,
    n_total_seeds: int,
    block_nr: int,
) -> tuple[int, int]:
    """Open + pre-allocate `IndexBest.bin` and `IndexBestFull.bin`.

    Mirrors IndexerOMP.c:2256-2284. On block_nr == 0 the files are
    truncated to their full size (`ftruncate` zero-fills). For later
    blocks we open existing files for pwrite.

    Returns raw file descriptors (int) suitable for `os.pwrite`.
    """
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    flags = os.O_CREAT | os.O_WRONLY
    if block_nr == 0:
        flags |= os.O_TRUNC

    fd_best = os.open(folder / "IndexBest.bin", flags, 0o600)
    fd_full = os.open(folder / "IndexBestFull.bin", flags, 0o600)

    if block_nr == 0:
        os.ftruncate(fd_best, n_total_seeds * INDEX_BEST_RECORD_BYTES)
        os.ftruncate(fd_full, n_total_seeds * INDEX_BEST_FULL_RECORD_BYTES)

    return fd_best, fd_full


def close_output_files(fd_best: int, fd_full: int) -> None:
    os.close(fd_best)
    os.close(fd_full)


def _seed_record(seed: "SeedResult") -> np.ndarray:
    """Pack one SeedResult into the 15-double IndexBest.bin record.

    Matches `WriteBestMatchBin` from IndexerOMP.c:1620-1628:
        rec[13] = GrainMatches[0][12] = (double)nTspots
        rec[14] = GrainMatches[0][13] = (double)nMatches
    """
    rec = np.zeros(INDEX_BEST_RECORD_DOUBLES, dtype=np.float64)
    rec[0] = float(seed.avg_ia)
    rec[1:10] = seed.best_or_mat.detach().to(torch.float64).cpu().reshape(-1).numpy()
    rec[10:13] = seed.best_pos.detach().to(torch.float64).cpu().numpy()
    rec[13] = float(seed.n_t_spots)     # TOTAL theor spots (nTspots in C)
    rec[14] = float(seed.n_matches)      # matches found
    return rec


def write_seed_record(
    fd_best: int,
    seed: "SeedResult",
    offset_loc: int,
) -> None:
    """pwrite one seed's 15-double record at slot `offset_loc` in IndexBest.bin.

    `offset_loc` is the seed's row in SpotsToIndex.csv (NOT just within the
    block). This matches `WriteBestMatchBin(... offsetLoc)` from C.
    """
    rec = _seed_record(seed)
    byte_offset = offset_loc * INDEX_BEST_RECORD_BYTES
    n = os.pwrite(fd_best, rec.tobytes(), byte_offset)
    if n != INDEX_BEST_RECORD_BYTES:
        raise IOError(
            f"pwrite to IndexBest.bin wrote {n}/{INDEX_BEST_RECORD_BYTES} bytes "
            f"at offset {byte_offset}"
        )


def write_full_record(
    fd_full: int,
    matched_pairs: np.ndarray,
    offset_loc: int,
) -> None:
    """pwrite the matched-spot record for one seed into IndexBestFull.bin.

    `matched_pairs` is `[MAX_N_HKLS, 2]` float64 (or shorter; gets padded).
    Per IndexerOMP.c:1635-1640 the C code writes one row per matched theoretical
    spot with `[matched_obs_id, delta_omega]`, padded to MAX_N_HKLS rows of zeros.
    """
    if matched_pairs.dtype != np.float64:
        matched_pairs = matched_pairs.astype(np.float64)
    if matched_pairs.shape[1] != 2:
        raise ValueError(
            f"matched_pairs must have shape [n, 2]; got {matched_pairs.shape}"
        )
    n = matched_pairs.shape[0]
    if n > MAX_N_HKLS:
        raise ValueError(
            f"matched_pairs has {n} rows; legacy format caps at MAX_N_HKLS={MAX_N_HKLS}"
        )
    rec = np.zeros(INDEX_BEST_FULL_RECORD_DOUBLES, dtype=np.float64)
    rec[: n * 2] = matched_pairs.reshape(-1)
    byte_offset = offset_loc * INDEX_BEST_FULL_RECORD_BYTES
    written = os.pwrite(fd_full, rec.tobytes(), byte_offset)
    if written != INDEX_BEST_FULL_RECORD_BYTES:
        raise IOError(
            f"pwrite to IndexBestFull.bin wrote {written}/{INDEX_BEST_FULL_RECORD_BYTES}"
        )


def write_block(
    result: "IndexerResult",
    output_folder: str | Path,
    n_total_seeds: int,
    block_nr: int,
    seed_to_offset_loc: dict[int, int] | None = None,
) -> None:
    """Convenience wrapper: open + write all seeds in `result` + close.

    `seed_to_offset_loc` maps `SeedResult.spot_id` to its row in SpotsToIndex.csv.
    If omitted, seeds are written in iteration order starting at the block's
    `startRowNr` (caller is responsible for ordering).
    """
    fd_best, fd_full = open_output_files(output_folder, n_total_seeds, block_nr)
    try:
        for i, seed in enumerate(result.seeds):
            offset = (
                seed_to_offset_loc[seed.spot_id]
                if seed_to_offset_loc is not None
                else i
            )
            write_seed_record(fd_best, seed, offset)
            # IndexBestFull.bin: matched_pairs population is the per-seed
            # responsibility of the matching stage. `SeedResult.matched_ids`
            # alone is not enough; the pairs (matched_id, delta_omega) come
            # from the matching kernel. v0.1.0 lands the writer here; the
            # producer wires up in P5.
    finally:
        close_output_files(fd_best, fd_full)


# Back-compat alias kept for the original plan section reference.
write_best_pos_csv = write_block
