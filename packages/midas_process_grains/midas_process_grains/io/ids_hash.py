"""``IDsHash.csv`` — SpotID-range-per-ring lookup.

Each line of ``IDsHash.csv`` is::

    <ring_nr> <id_min> <id_max> <d_spacing_A>

The C code uses this table to look up the reference d-spacing for a matched
SpotID inside the strain solver — see ``CalcStrains.c::StrainTensorKenesei``
and ``ProcessGrains.c:797-832``. We reuse the same convention so our Phase-4
strain has identical reference d-values.

Note: ``id_max`` is **exclusive** in the C convention (the next ring starts
where this one ends — e.g. ``3 1 738333 ...`` then ``4 738333 ...``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class IDsHash:
    """Sorted SpotID-range table.

    Attributes
    ----------
    ring_nrs : np.ndarray
        ``(n_rings,)`` int — ring numbers in ascending order.
    id_starts : np.ndarray
        ``(n_rings,)`` int64 — inclusive lower SpotID bound for each ring.
    id_ends : np.ndarray
        ``(n_rings,)`` int64 — exclusive upper SpotID bound.
    d_spacings : np.ndarray
        ``(n_rings,)`` float64 — d-spacing in Å for each ring.
    """

    ring_nrs: np.ndarray
    id_starts: np.ndarray
    id_ends: np.ndarray
    d_spacings: np.ndarray

    def ring_for_spot_id(self, spot_id: int) -> int:
        """Return the ring number for a SpotID, or ``-1`` if out of range."""
        idx = np.searchsorted(self.id_starts, spot_id, side="right") - 1
        if idx < 0 or idx >= self.ring_nrs.size:
            return -1
        if spot_id >= self.id_ends[idx]:
            return -1
        return int(self.ring_nrs[idx])

    def d_for_spot_id(self, spot_id: int) -> float:
        """Return the reference d-spacing for a SpotID, or ``0.0`` if missing."""
        idx = np.searchsorted(self.id_starts, spot_id, side="right") - 1
        if idx < 0 or idx >= self.ring_nrs.size:
            return 0.0
        if spot_id >= self.id_ends[idx]:
            return 0.0
        return float(self.d_spacings[idx])

    def d_for_spot_ids(self, spot_ids: np.ndarray) -> np.ndarray:
        """Vectorised lookup; returns ``0.0`` for out-of-range SpotIDs."""
        out = np.zeros(spot_ids.shape, dtype=np.float64)
        # binary-search per element using sorted starts
        idx = np.searchsorted(self.id_starts, spot_ids, side="right") - 1
        valid = (idx >= 0) & (idx < self.ring_nrs.size)
        if valid.any():
            v = idx[valid]
            in_range = spot_ids[valid] < self.id_ends[v]
            sub = np.where(valid)[0][in_range]
            out[sub] = self.d_spacings[idx[sub]]
        return out


def load_ids_hash(path: Union[str, Path]) -> IDsHash:
    """Parse ``IDsHash.csv`` into an :class:`IDsHash`.

    Lines of the form ``ring_nr id_min id_max d_spacing`` (whitespace-separated).
    """
    rings: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    ds: List[float] = []
    with open(path, "r") as f:
        for raw in f:
            tokens = raw.split()
            if len(tokens) < 4:
                continue
            try:
                rings.append(int(tokens[0]))
                starts.append(int(tokens[1]))
                ends.append(int(tokens[2]))
                ds.append(float(tokens[3]))
            except (ValueError, IndexError):
                continue
    if not rings:
        raise ValueError(f"{path} contained no parseable rows")
    rings_a = np.asarray(rings, dtype=np.int64)
    starts_a = np.asarray(starts, dtype=np.int64)
    ends_a = np.asarray(ends, dtype=np.int64)
    ds_a = np.asarray(ds, dtype=np.float64)
    order = np.argsort(starts_a)
    return IDsHash(
        ring_nrs=rings_a[order],
        id_starts=starts_a[order],
        id_ends=ends_a[order],
        d_spacings=ds_a[order],
    )
