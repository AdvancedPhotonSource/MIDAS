"""Symmetry-op table + hkl-row permutation builder.

Two artefacts per (space group, hkl table):

  ``ops_quat[s] : (4,)`` — quaternion of the s-th proper rotation. Built via
  ``midas_stress.orientation.make_symmetries(space_group)`` which mirrors the
  MIDAS C ``MakeSymmetries`` exactly.

  ``ops_R[s] : (3, 3)`` — rotation matrix form of the same op (lazy cache).

  ``hkl_perm[s, k] : int`` — row index of ``(h, k, l)' = R_s · (h, k, l)_k``
  in the supplied :class:`HklTable`. ``-1`` if the image is not in the
  table (rare; flagged as a warning if it ever occurs for rings the user
  asked the indexer to enumerate).

Why we need the row-permutation
-------------------------------
Two seeds in a Phase-1 cluster may explain the same physical grain via
*different* symmetry-equivalent orientation variants. Their per-row
``IndexBestFull`` tables list the matched SpotID at row k, where row k means
"theoretical hkl (h,k,l)_k of the fixed indexer hkl list". That row's
*physical* meaning rotates with the chosen variant. Aligning row indices
across seeds before comparing matched SpotIDs requires applying the inverse
permutation of the symmetry op that brought the seed into the rep's frame.

See Phase 2 of ``implementation_plan.md`` for full algorithm context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from midas_stress.orientation import make_symmetries, quat_to_orient_mat

from ..io.hkls import HklTable


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SymmetryTable:
    """Cached symmetry artefacts for one (space group, hkl table) pair.

    Attributes
    ----------
    space_group : int
        Space group number (e.g. 225 for FCC FmAm).
    n_sym : int
        Number of proper rotations in the point group (24 for cubic,
        12 for hexagonal, etc.).
    ops_quat : torch.Tensor
        ``(n_sym, 4)`` quaternion form of each op, layout (w, x, y, z).
    ops_R : torch.Tensor
        ``(n_sym, 3, 3)`` rotation-matrix form of each op.
    hkl_perm : torch.Tensor
        ``(n_sym, n_hkls)`` int64. Row-permutation under each op.
        ``-1`` in cells where the image is not in the supplied table.
    n_hkls : int
        Number of rows in the hkl table.
    """

    space_group: int
    n_sym: int
    ops_quat: torch.Tensor
    ops_R: torch.Tensor
    hkl_perm: torch.Tensor
    n_hkls: int


# ---------------------------------------------------------------------------
# Build path
# ---------------------------------------------------------------------------


def apply_sym_to_hkl_int(
    R: np.ndarray,
    hkl_int: np.ndarray,
) -> np.ndarray:
    """Apply rotation ``R`` (3×3) to integer hkl rows ``(n, 3)``.

    Returns rounded integer ``(n, 3)`` so caller can look up the resulting
    triples in an ``hkl_to_row`` dict.
    """
    if hkl_int.ndim != 2 or hkl_int.shape[1] != 3:
        raise ValueError(f"hkl_int shape must be (n, 3); got {hkl_int.shape}")
    rotated = hkl_int.astype(np.float64) @ R.T   # (n, 3)
    return np.rint(rotated).astype(np.int64)


def build_symmetry_table(
    space_group: int,
    hkl_table: HklTable,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    warn_missing: bool = True,
) -> SymmetryTable:
    """Build the (sym ops, hkl-row permutation) cache.

    Parameters
    ----------
    space_group : int
        Space group number passed to ``midas_stress.make_symmetries``.
    hkl_table : HklTable
        From :func:`midas_process_grains.io.hkls.load_hkl_table`. Must contain
        the integer (h, k, l) triples and a reverse-lookup dict.
    device, dtype : torch
        Where to place ``ops_quat``, ``ops_R``, ``hkl_perm``. The permutation
        table is int64 on the same device.
    warn_missing : bool
        If True, prints a one-line warning when a row's image under some op
        is not in the supplied hkl table. This is normal when the user has
        filtered to a partial ring set; set False to silence.

    Returns
    -------
    SymmetryTable
    """
    n_sym, sym_quats = make_symmetries(space_group)
    if n_sym == 0:
        raise ValueError(f"make_symmetries({space_group}) returned 0 ops.")

    sym_quats_np = np.asarray(sym_quats, dtype=np.float64)        # (n_sym, 4)
    # Defensive renormalization: midas_stress's hard-coded symmetry quats
    # carry small float-literal drift that compounds in quat_to_orient_mat
    # (det of the resulting rotation matrix can drift to ~1+2e-5). Normalise.
    norms = np.linalg.norm(sym_quats_np, axis=1, keepdims=True)
    sym_quats_np = sym_quats_np / np.clip(norms, 1e-30, None)

    # Rotation matrices via midas_stress conversion (vectorised loop).
    ops_R_np = np.empty((n_sym, 3, 3), dtype=np.float64)
    for s in range(n_sym):
        ops_R_np[s] = np.asarray(quat_to_orient_mat(sym_quats_np[s])).reshape(3, 3)
    # Final orthogonalize via SVD to scrub residual numerical drift; the
    # closest orthogonal matrix is U @ V^T with Σ replaced by I.
    for s in range(n_sym):
        U, _, Vt = np.linalg.svd(ops_R_np[s])
        R = U @ Vt
        # Enforce proper rotation (det = +1).
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt
        ops_R_np[s] = R

    # Build the hkl row permutation.
    hkl_int = np.asarray(hkl_table.integers[:, :3], dtype=np.int64)   # (n_hkls, 3)
    n_hkls = hkl_int.shape[0]
    hkl_perm = np.full((n_sym, n_hkls), -1, dtype=np.int64)
    n_missing = 0
    for s in range(n_sym):
        rotated = apply_sym_to_hkl_int(ops_R_np[s], hkl_int)
        for k in range(n_hkls):
            key = (int(rotated[k, 0]), int(rotated[k, 1]), int(rotated[k, 2]))
            row = hkl_table.hkl_to_row.get(key, -1)
            if row < 0:
                n_missing += 1
            hkl_perm[s, k] = row

    if warn_missing and n_missing > 0:
        # Use stdlib warnings to be importable without extra deps.
        import warnings
        warnings.warn(
            f"{n_missing} hkl rows have a symmetry image outside the supplied "
            f"hkl table (n_sym={n_sym}, n_hkls={n_hkls}). Cells with no image "
            "are marked -1 and skipped in row-aligned comparisons.",
            stacklevel=2,
        )

    # Move to torch.
    dev = None if device is None else torch.device(device)
    ops_quat_t = torch.from_numpy(sym_quats_np).to(device=dev, dtype=dtype)
    ops_R_t = torch.from_numpy(ops_R_np).to(device=dev, dtype=dtype)
    hkl_perm_t = torch.from_numpy(hkl_perm).to(device=dev, dtype=torch.int64)

    return SymmetryTable(
        space_group=space_group,
        n_sym=int(n_sym),
        ops_quat=ops_quat_t,
        ops_R=ops_R_t,
        hkl_perm=hkl_perm_t,
        n_hkls=n_hkls,
    )
