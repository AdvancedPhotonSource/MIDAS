"""Tolerance-mode spot association.

Pure-Python port of :c:func:`process_spots`
(findSingleSolutionPFRefactored.c:1027–1264). For each unique grain,
collect its matched spot IDs from ``IndexBest_IDs_all.bin`` (the
best-solution's IDs), look them up in ``Spots.bin``, and deduplicate
within-grain via the (ringNr, omega, eta) tuple within tolerances.

Output is a :class:`SpotList` ready for :mod:`._sinogen`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ._consolidation_io import (
    CONSOLIDATED_KEY_COLS,
    ConsolidatedReader,
)


SPOTS_ARRAY_COLS = 10  # [x, y, omega, intensity, spotID, ringNum, eta, theta, dspacing, scanNum]


@dataclass
class SpotData:
    """Per-spot metadata for sinogen lookups."""

    omega: float
    eta: float
    ring_nr: int
    merged_id: int
    scan_nr: int
    grain_nr: int
    spot_nr: int


@dataclass
class SpotList:
    """Output of :func:`process_spots`."""

    spots: List[SpotData] = field(default_factory=list)
    max_n_hkls: int = 0
    n_invalid_ids: int = 0
    n_misaligned: int = 0


def process_spots(
    unique_key_arr: np.ndarray,
    all_spots: np.ndarray,
    *,
    keys_reader: ConsolidatedReader,
    ids_reader: ConsolidatedReader,
    tol_ome: float,
    tol_eta: float,
) -> SpotList:
    """Associate spots with unique grains (tolerance-mode dedup).

    Parameters
    ----------
    unique_key_arr : ndarray (n_unique, 5) uint64
        ``[voxNr, SpotID, nMatches, nIDs, bestSolIdx]`` from
        :func:`._cluster.global_cluster`.
    all_spots : ndarray (n_spots_all, 10) float64
        ``Spots.bin`` data; row order = spotID-1.
    keys_reader, ids_reader : :class:`ConsolidatedReader`
        Open readers for ``IndexKey_all.bin`` and
        ``IndexBest_IDs_all.bin``.
    tol_ome, tol_eta : float
        Tolerances (degrees) for within-grain duplicate detection.

    Returns
    -------
    SpotList — list of unique-only spots, plus the maximum spot count
    over any grain.
    """
    all_spots = np.ascontiguousarray(all_spots, dtype=np.float64)
    n_spots_all = int(all_spots.shape[0])
    n_unique = int(unique_key_arr.shape[0])

    # All collected spots (pre-dedup), per-grain ring/omega/eta lists.
    collected: list[SpotData] = []
    is_dup: list[bool] = []

    # Per-grain running spot-count (for assigning spot_nr; sequential after dedup).
    n_invalid_ids = 0
    n_misaligned = 0

    for g in range(n_unique):
        vox_nr = int(unique_key_arr[g, 0])
        best_sol_idx = int(unique_key_arr[g, 4])
        n_ids_best = int(unique_key_arr[g, 3])
        if n_ids_best <= 0:
            continue

        all_ids_for_vox = ids_reader.get_ids(vox_nr)
        vox_keys = keys_reader.get_keys(vox_nr)
        if all_ids_for_vox is None or vox_keys is None:
            continue

        # Offset into the IDs list for this voxel's best solution =
        # sum(nIDs of solutions [0..best_sol_idx-1]). Matches C:1096.
        if best_sol_idx > 0:
            id_offset = int(vox_keys[:best_sol_idx, 2].sum())
        else:
            id_offset = 0
        if id_offset + n_ids_best > all_ids_for_vox.size:
            continue

        ids_this = all_ids_for_vox[id_offset : id_offset + n_ids_best]

        for j, sid in enumerate(ids_this):
            sid_i = int(sid)
            if sid_i < 1 or sid_i > n_spots_all:
                n_invalid_ids += 1
                continue
            spot_idx = sid_i - 1
            if int(all_spots[spot_idx, 4]) != sid_i:
                n_misaligned += 1
                continue

            sd = SpotData(
                omega=float(all_spots[spot_idx, 2]),
                eta=float(all_spots[spot_idx, 6]),
                ring_nr=int(all_spots[spot_idx, 5]),
                merged_id=sid_i,
                scan_nr=int(all_spots[spot_idx, 9]),
                grain_nr=g,
                spot_nr=j,
            )

            # Within-grain ring/omega/eta dedup against ALL prior collected
            # spots (C scans across grains; we reproduce that exactly even
            # though it's per-grain in spirit, because the C code looks at
            # collected[0..nAllSpots+j-1] including other grains).
            is_dupe_here = False
            for k in range(len(collected)):
                prev = collected[k]
                if (
                    prev.ring_nr == sd.ring_nr
                    and abs(sd.omega - prev.omega) < tol_ome
                    and abs(sd.eta - prev.eta) < tol_eta
                ):
                    is_dup[k] = True
                    is_dupe_here = True
                    break
            collected.append(sd)
            is_dup.append(is_dupe_here)

    # Filter to unique-only and renumber spot_nr sequentially per grain.
    n_per_grain = np.zeros(max(n_unique, 1), dtype=np.int64)
    spots_out: list[SpotData] = []
    for sd, dup in zip(collected, is_dup):
        if dup:
            continue
        sd.spot_nr = int(n_per_grain[sd.grain_nr])
        n_per_grain[sd.grain_nr] += 1
        spots_out.append(sd)

    max_n_hkls = int(n_per_grain.max()) if n_per_grain.size else 0

    return SpotList(
        spots=spots_out,
        max_n_hkls=max_n_hkls,
        n_invalid_ids=n_invalid_ids,
        n_misaligned=n_misaligned,
    )
