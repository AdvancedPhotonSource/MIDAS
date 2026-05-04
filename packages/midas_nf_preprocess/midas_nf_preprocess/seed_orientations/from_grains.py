"""Parse an FF-HEDM ``Grains.csv`` file into seed orientations.

Port of ``NF_HEDM/src/GenSeedOrientationsFF2NFHEDM.c``. The C code reads each
grain row, extracts the 9 ``OrientMatrix`` columns + the 6 ``LatC`` columns +
the ``GrainID``, and writes the per-grain quaternion (no rotation; just
OrientMat -> quat).

The Grains.csv column convention parsed by the C code (sscanf at L54-L61):

    GrainID
    OrientMatrix[0..8]  (9 doubles)
    <3 dummies>         (typically positions, ignored)
    LatC[0..5]          (6 doubles: a, b, c, alpha, beta, gamma)
    <4 dummies>         (typically stress / strain / GoF, ignored)

Lines starting with ``%`` are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from ..device import resolve_device, resolve_dtype
from ..diffr_spots.orientations import orient_matrix_to_quat


@dataclass
class GrainOrientation:
    """One row of an FF Grains.csv file (relevant fields only)."""

    grain_id: int
    orient_matrix: torch.Tensor   # (3, 3)
    quat: torch.Tensor            # (4,) -- (w, x, y, z)
    lattice: tuple[float, ...]    # (a, b, c, alpha, beta, gamma)


def read_grains_orientations(
    path: Union[str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> list[GrainOrientation]:
    """Parse an FF Grains.csv file into a list of ``GrainOrientation``.

    The OrientMat-to-quat conversion is delegated to
    :func:`midas_nf_preprocess.diffr_spots.orientations.orient_matrix_to_quat`
    (same numerically stable Shepperd's-method implementation that the rest of
    the package uses).

    Parameters
    ----------
    path : Grains.csv path.
    device, dtype : standard torch construction kwargs.

    Returns
    -------
    list of GrainOrientation, in file order.
    """
    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)

    grains: list[GrainOrientation] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            # Need at least 1 (id) + 9 (OM) + 3 (dummies) + 6 (latc) = 19 cols.
            if len(parts) < 19:
                continue
            try:
                grain_id = int(float(parts[0]))
                om_vals = [float(p) for p in parts[1:10]]
                latc = tuple(float(p) for p in parts[13:19])
            except ValueError:
                continue
            om = torch.tensor(om_vals, device=device, dtype=dtype).reshape(3, 3)
            quat = orient_matrix_to_quat(om.unsqueeze(0)).squeeze(0)
            grains.append(
                GrainOrientation(
                    grain_id=grain_id,
                    orient_matrix=om,
                    quat=quat,
                    lattice=latc,
                )
            )
    return grains


def grains_to_quaternions(grains: list[GrainOrientation]) -> torch.Tensor:
    """Stack the ``.quat`` field of each grain into a ``(N, 4)`` tensor."""
    if not grains:
        # Caller should treat this as "no grains found".
        return torch.empty((0, 4))
    return torch.stack([g.quat for g in grains], dim=0)
