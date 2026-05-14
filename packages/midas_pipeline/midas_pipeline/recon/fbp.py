"""Filtered Back-Projection reconstruction wrapper.

This module is intentionally **NOT** a relocation of
``TOMO/midas_tomo_python.py``. That file is consumed by people outside
the pipeline (notebook callers, the standalone TOMO workflows), so we
keep it in place and import it here through a ``sys.path`` hop. The
public entry point ``fbp_recon`` matches the contract expected by
``midas_pipeline.stages.reconstruct``.

The underlying ``run_tomo_from_sinos`` shells out to the
``MIDAS_TOMO_C`` (or ``MIDAS_TOMO_GPU``) binary; everything else stays
in Python here. We do not duplicate the C entrypoint.
"""

from __future__ import annotations

import os
import sys
from math import ceil, log2
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


def _ensure_tomo_on_path() -> None:
    """Lazily prepend the legacy ``TOMO/`` directory to ``sys.path``.

    Done at function-call time, not import time, so importing this
    module never touches the filesystem.
    """
    # packages/midas_pipeline/midas_pipeline/recon/fbp.py
    # parents[0]=recon, [1]=midas_pipeline, [2]=midas_pipeline pkg root,
    # [3]=packages, [4]=MIDAS repo root → MIDAS_ROOT/TOMO
    here = Path(__file__).resolve()
    midas_root = here.parents[4]
    tomo_dir = midas_root / "TOMO"
    if tomo_dir.is_dir() and str(tomo_dir) not in sys.path:
        sys.path.insert(0, str(tomo_dir))


def _load_run_tomo_from_sinos():
    """Import ``run_tomo_from_sinos`` from the in-place TOMO module."""
    _ensure_tomo_on_path()
    try:
        from midas_tomo_python import run_tomo_from_sinos  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "midas_pipeline.recon.fbp could not import "
            "midas_tomo_python.run_tomo_from_sinos. Ensure TOMO/ "
            "is part of the MIDAS source tree and importable."
        ) from exc
    return run_tomo_from_sinos


def fbp_recon(
    sinograms: np.ndarray,
    thetas: np.ndarray,
    workingdir: Union[str, Path],
    *,
    n_scans: Optional[int] = None,
    filter_nr: int = 2,
    do_log: int = 0,
    extra_pad: int = 0,
    auto_centering: int = 1,
    num_cpus: int = 1,
    do_cleanup: int = 1,
    use_gpu: bool = False,
) -> np.ndarray:
    """Run FBP on a single sinogram, return a centered nScans×nScans recon.

    Thin wrapper around ``TOMO.midas_tomo_python.run_tomo_from_sinos``
    that does the standard pf-HEDM post-crop: MIDAS_TOMO upscales
    ``detXdim`` to the next power of two; we crop the central
    ``nScans × nScans`` region so the recon center aligns with the
    voxel-grid center ``(nScans - 1) / 2``.

    Parameters
    ----------
    sinograms : ndarray, shape (nThetas, detXdim) or (nSlices, nThetas, detXdim)
        Pre-formed sinograms.
    thetas : ndarray, shape (nThetas,)
        Projection angles in degrees.
    workingdir : str | Path
        Temp directory for MIDAS_TOMO inputs / outputs.
    n_scans : int, optional
        If given, crop to (n_scans, n_scans). Default: full upscaled recon.
    filter_nr, do_log, extra_pad, auto_centering, num_cpus, do_cleanup, use_gpu
        Forwarded to ``run_tomo_from_sinos``; see that function's
        docstring for semantics.

    Returns
    -------
    ndarray
        If ``n_scans`` is given: ``(n_scans, n_scans)``, cropped + transposed
        to match the voxel-grid spatial convention. Otherwise the
        upscaled square returned by MIDAS_TOMO.
    """
    run_tomo_from_sinos = _load_run_tomo_from_sinos()
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)

    recon_arr = run_tomo_from_sinos(
        np.asarray(sinograms, dtype=np.float32),
        str(workingdir),
        np.asarray(thetas, dtype=np.float64),
        shifts=0.0,
        filterNr=filter_nr,
        doLog=do_log,
        extraPad=extra_pad,
        autoCentering=auto_centering,
        numCPUs=num_cpus,
        doCleanup=do_cleanup,
        useGPU=use_gpu,
    )
    # recon_arr shape: (nrShifts, nSlices, xDimNew, xDimNew)
    full = recon_arr[0, 0]
    if n_scans is None:
        return full
    recon_dim = 1 << int(ceil(log2(n_scans))) if n_scans > 1 else 1
    if extra_pad == 1:
        recon_dim *= 2
    crop_start = recon_dim // 2 - n_scans // 2
    crop_end = crop_start + n_scans
    return full[crop_start:crop_end, crop_start:crop_end]


def fbp_recon_per_grain(
    sinos_by_grain: np.ndarray,
    omegas_by_grain: np.ndarray,
    n_hkls_per_grain: np.ndarray,
    n_scans: int,
    workingdir: Union[str, Path],
    *,
    filter_nr: int = 2,
    do_log: int = 0,
    extra_pad: int = 0,
    auto_centering: int = 1,
    num_cpus: int = 1,
    do_cleanup: int = 1,
    use_gpu: bool = False,
    transpose_to_voxel: bool = True,
) -> np.ndarray:
    """FBP every grain in one call, return the ``(nGrs, nScans, nScans)`` stack.

    Parameters
    ----------
    sinos_by_grain : ndarray, shape (nGrs, maxNHKLs, nScans), float64
    omegas_by_grain : ndarray, shape (nGrs, maxNHKLs), float64
        Per-grain projection angles in degrees.
    n_hkls_per_grain : ndarray, shape (nGrs,), int
        Number of valid HKLs per grain.
    n_scans : int
        Spatial dimension; used to crop the upscaled recon.
    workingdir : str | Path
        Per-grain temp directory.
    transpose_to_voxel : bool
        Apply the historical ``.T`` swap that aligns the recon with
        the voxel-grid spatial convention used everywhere else in
        pf-HEDM. True by default to match legacy.

    Returns
    -------
    ndarray, shape (nGrs, nScans, nScans), float32
    """
    n_grs = int(n_hkls_per_grain.shape[0])
    out = np.zeros((n_grs, n_scans, n_scans), dtype=np.float32)
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)
    for g in range(n_grs):
        n_sp = int(n_hkls_per_grain[g])
        if n_sp <= 0:
            continue
        thetas = np.asarray(omegas_by_grain[g, :n_sp], dtype=np.float64)
        sino = np.asarray(sinos_by_grain[g, :n_sp, :], dtype=np.float32)
        if (sino > 0).sum() == 0:
            continue
        recon = fbp_recon(
            sino, thetas, workingdir,
            n_scans=n_scans,
            filter_nr=filter_nr, do_log=do_log, extra_pad=extra_pad,
            auto_centering=auto_centering, num_cpus=num_cpus,
            do_cleanup=do_cleanup, use_gpu=use_gpu,
        )
        if transpose_to_voxel:
            recon = recon.T
        out[g] = recon.astype(np.float32, copy=False)
    return out
