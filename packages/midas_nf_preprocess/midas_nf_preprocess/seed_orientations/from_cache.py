"""Load pre-computed seed-orientation libraries from disk.

The MIDAS workflow ships pre-extracted CSVs and binary lookup tables under
``NF_HEDM/seedOrientations/`` for all 12 Laue-group buckets. This module
gives a torch-tensor view over those files.

Two file formats are supported:

  - ``seed_<lookup_type>.csv`` -- one CSV per bucket, comma-separated quaternions
    ``w, x, y, z``. The fastest path: a single ``np.loadtxt`` (or csv parse) and
    we are done.
  - ``orientations_master.bin`` + ``lookup_<lookup_type>.bin`` -- the binary
    layout used by ``GenerateSeedLookupTables.c``. Useful when the CSV has not
    yet been extracted; we use the ``np.fromfile`` path identical to
    ``utils/extract_seed_orientations.ensure_seed_orientations``.

When neither form is present we raise ``SeedCacheNotFound`` with a clear
message pointing the user at the from-scratch path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ..device import resolve_device, resolve_dtype
from .crystal import space_group_to_lookup_type


# Default location of the seed cache, relative to the MIDAS repo. Users can
# override with the env var ``MIDAS_NF_SEED_DIR`` or by passing ``seed_dir=``.
DEFAULT_SEED_DIR = Path(__file__).resolve().parents[3] / ".." / "NF_HEDM" / "seedOrientations"


class SeedCacheNotFound(FileNotFoundError):
    """No cached seed file found for the requested lookup type."""


def _resolve_seed_dir(seed_dir: Optional[Union[str, Path]]) -> Path:
    if seed_dir is not None:
        return Path(seed_dir)
    env = os.environ.get("MIDAS_NF_SEED_DIR")
    if env:
        return Path(env)
    return DEFAULT_SEED_DIR


def _load_csv(path: Path) -> np.ndarray:
    """Load a comma-separated quaternion CSV (w,x,y,z per line).

    ``np.loadtxt`` collapses a single-row file to a 1D ``(4,)`` array; promote
    back to ``(1, 4)`` so the caller can rely on the 2D shape.
    """
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _load_from_master_lookup(seed_dir: Path, lookup_type: str) -> np.ndarray:
    """Reconstitute quaternions from orientations_master.bin + lookup_*.bin.

    Mirrors ``utils/extract_seed_orientations.ensure_seed_orientations`` L73-L75.
    """
    master_path = seed_dir / "orientations_master.bin"
    lookup_path = seed_dir / f"lookup_{lookup_type}.bin"
    if not master_path.exists() or not lookup_path.exists():
        raise SeedCacheNotFound(
            f"Neither seed_{lookup_type}.csv nor "
            f"(orientations_master.bin + lookup_{lookup_type}.bin) found "
            f"in {seed_dir}. "
            "Run NF_HEDM/bin/GenerateSeedLookupTables to populate the cache, "
            "or use generate_uniform_seeds(...) instead."
        )
    master = np.fromfile(master_path, dtype=np.float64).reshape(-1, 4)
    indices = np.fromfile(lookup_path, dtype=np.int32)
    return master[indices]


def load_seeds_for_lookup_type(
    lookup_type: str,
    *,
    seed_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Load a cached seed-orientation file by MIDAS lookup-type name.

    Parameters
    ----------
    lookup_type : one of the 12 names in ``LOOKUP_TYPES``
        (``cubic_high``, ``hexagonal_high``, ...).
    seed_dir : override the cache directory. Default = ``$MIDAS_NF_SEED_DIR``
        or the bundled ``NF_HEDM/seedOrientations/``.

    Returns
    -------
    Tensor of shape ``(N, 4)`` -- quaternions ``(w, x, y, z)``.
    """
    seed_dir = _resolve_seed_dir(seed_dir)
    if not seed_dir.exists():
        raise SeedCacheNotFound(
            f"Seed cache directory not found: {seed_dir}. "
            "Set MIDAS_NF_SEED_DIR or pass seed_dir=..."
        )

    csv_path = seed_dir / f"seed_{lookup_type}.csv"
    if csv_path.exists():
        arr = _load_csv(csv_path)
    else:
        arr = _load_from_master_lookup(seed_dir, lookup_type)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"{csv_path}: expected (N, 4) quaternions, got shape {arr.shape}"
        )

    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def load_seeds_for_space_group(
    space_group: int,
    *,
    seed_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Convenience wrapper: SG -> lookup type -> cached seeds.

    See :func:`load_seeds_for_lookup_type` for parameter docs.
    """
    lookup_type = space_group_to_lookup_type(space_group)
    return load_seeds_for_lookup_type(
        lookup_type, seed_dir=seed_dir, device=device, dtype=dtype
    )
