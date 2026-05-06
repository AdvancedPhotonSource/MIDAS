"""HKL table loader for the new pipeline.

Wraps ``midas_index.io.csv.read_hkls_csv`` and adds the integer ``(h, k, l)``
representation required by the symmetry-permutation builder. The C code uses
``hkls.csv`` filtered by ``RingNumbers``; this module provides the same view
plus the integer-tuple → row-index lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

from midas_index.io.csv import read_hkls_csv


@dataclass(frozen=True)
class HklTable:
    """The set of theoretical hkls used by the indexer for this run.

    Attributes
    ----------
    real : (n, 7) float64
        ``(g1, g2, g3, ring_nr, d_spacing, theta_rad, ring_radius)``.
        Theta is in radians.
    integers : (n, 4) int32
        ``(h, k, l, ring_nr)``.
    hkl_to_row : dict[(h,k,l), int]
        Reverse lookup of integer hkl → row index.
    """

    real: np.ndarray
    integers: np.ndarray
    hkl_to_row: Dict[Tuple[int, int, int], int]

    def __len__(self) -> int:
        return int(self.integers.shape[0])

    @property
    def n_hkls(self) -> int:
        return len(self)


def load_hkl_table(
    hkls_path: Union[str, Path],
    ring_numbers: Optional[Iterable[int]] = None,
) -> HklTable:
    """Read ``hkls.csv``, optionally filtered by ring numbers.

    Returns the canonical row order — same order the indexer iterates, so
    column 0 of ``IndexBestFull.bin`` indexes into this table directly.
    """
    real, integers = read_hkls_csv(hkls_path, ring_numbers=ring_numbers)
    hkl_to_row: Dict[Tuple[int, int, int], int] = {
        (int(integers[i, 0]), int(integers[i, 1]), int(integers[i, 2])): i
        for i in range(integers.shape[0])
    }
    return HklTable(real=real, integers=integers, hkl_to_row=hkl_to_row)
