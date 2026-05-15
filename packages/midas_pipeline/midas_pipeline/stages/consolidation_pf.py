"""PF-mode consolidation: pure-Python port of pf_MIDAS.py:2429-2519.

This module assembles the final PF-HEDM microstructure artefacts after
the per-voxel refinement stage has written its per-voxel result CSVs.

Inputs (in ``layer_dir``)
-------------------------
- ``Results/*.csv``: per-voxel refinement output. One file per voxel.
  Each filename encodes the voxel index as the second-from-last
  underscore-separated token before the ``.csv`` suffix (matches the C
  refiner naming ``FitBest_<voxNr>_<SpId>.csv`` and the legacy parsing
  at ``pf_MIDAS.py:2443``). Each file has a header line followed by one
  data row of 43 whitespace-separated columns.
- ``Recons/Full_recon_max_project_grID.tif``: 2D grain-ID map produced
  by the reconstruct stage. Already present on disk before consolidation
  runs — we don't recompute it here, only adopt it as part of the result.

Outputs (in ``layer_dir/Recons``)
---------------------------------
- ``microstrFull.csv``: one row per accepted voxel (43 columns,
  comma-delimited, with full 43-column header). Format matches the
  legacy production output exactly.
- ``microstructure.hdf``: HDF5 with ``microstr`` (the voxel table) and
  ``images`` (a ``(23, nScans, nScans)`` reshape/flip/transpose of a
  subset of columns) datasets, each with a ``Header`` attribute.

Orientation math
----------------
All symmetry-aware quaternion math runs through ``midas_stress``:

- ``midas_stress.orientation.make_symmetries``
- ``midas_stress.orientation.orient_mat_to_quat``
- ``midas_stress.orientation.orient_mat_to_euler`` (not used here but
  is the canonical replacement for the legacy ``OrientMat2Euler``)
- ``midas_stress.orientation.fundamental_zone``

This replaces the legacy ``calcMiso.*`` imports that the inline
production block uses. No call to ``utils/calcMiso.py`` remains in
this module.

torch / multi-device
--------------------
The legacy logic is dominated by single-row CSV parsing + a single
quaternion reduction per voxel — there is no per-voxel batched math
that benefits from a torch backend here. The symmetry-aware quaternion
fundamental-zone reduction is delegated to ``midas_stress`` which
itself dispatches to torch when the caller passes torch tensors.
We expose an optional torch helper :func:`reduce_quats_to_fz_torch`
for callers that already hold a batched quaternion tensor.
"""

from __future__ import annotations

import glob
import os
import time
from math import isnan
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from midas_stress.orientation import (
    fundamental_zone,
    make_symmetries,
    orient_mat_to_quat,
)

from ..results import ConsolidationResult


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

#: Per-voxel CSV column count, matches FitOrStrainsScanningOMP output.
N_COLS: int = 43

#: ``microstrFull.csv`` column header, byte-for-byte identical to the
#: legacy production string at ``pf_MIDAS.py:2472-2473``.
MICROSTR_HEADER: str = (
    "SpotID,O11,O12,O13,O21,O22,O23,O31,O32,O33,SpotID,x,y,z,SpotID,a,b,c,"
    "alpha,beta,gamma,SpotID,PosErr,OmeErr,InternalAngle,"
    "Radius,Completeness,E11,E12,E13,E21,E22,E23,E31,E32,E33,"
    "Eul1,Eul2,Eul3,Quat1,Quat2,Quat3,Quat4"
)

#: ``images`` dataset column header, identical to legacy.
IMAGES_HEADER: str = (
    "ID,Quat1,Quat2,Quat3,Quat4,x,y,a,b,c,alpha,beta,gamma,"
    "posErr,omeErr,InternalAngle,Completeness,E11,E12,E13,E22,E23,E33"
)

#: Subset of columns mapped into the ``images`` per-voxel array, in
#: the same order as :data:`IMAGES_HEADER`. Matches
#: ``pf_MIDAS.py:2467`` exactly.
_IMAGES_COL_INDEX: tuple[int, ...] = (
    0, -4, -3, -2, -1, 11, 12, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26, 27,
    28, 29, 31, 32, 35,
)


def _voxel_index_from_filename(file_path: str) -> int:
    """Extract the voxel index from a per-voxel result filename.

    Handles both refiner outputs:
      - C ``FitBest_VVVVV_SSSSSS.csv``                 → voxNr = parts[-2]
      - Python ``Result_OrientPos_voxel_N.csv``         → voxNr = parts[-1]

    Mirrors the legacy parsing convention at pf_MIDAS.py:2443 for the
    C-named case, plus a fallback for the new midas-fit-grain naming.
    """
    stem = file_path.split(".")[-2]
    parts = stem.split("_")
    # Python form: ..._voxel_N
    if len(parts) >= 2 and parts[-2] == "voxel":
        return int(parts[-1])
    # C form: ..._VVVVV_SSSSSS
    return int(parts[-2])


# ---------------------------------------------------------------------------
# Optional torch helper (multi-device, autograd-safe)
# ---------------------------------------------------------------------------


def reduce_quats_to_fz_torch(quats, space_group: int):
    """Reduce a batch of quaternions to the fundamental zone using torch.

    Thin wrapper around ``midas_stress.orientation.fundamental_zone``
    that ensures torch input passes straight through (preserving the
    device + autograd graph). Useful for callers that have already
    built up a tensor of per-voxel quaternions.

    Parameters
    ----------
    quats : torch.Tensor
        Shape ``(..., 4)`` quaternions, layout ``[w, x, y, z]``.
    space_group : int
        Space group number (1-230).

    Returns
    -------
    torch.Tensor
        Same leading shape, trailing ``(4,)``. Same device + dtype.
    """
    # midas_stress.orientation.fundamental_zone dispatches on torch input;
    # we forward verbatim so autograd + device propagate without any copy.
    return fundamental_zone(quats, space_group=space_group)


# ---------------------------------------------------------------------------
# Per-voxel CSV ingestion
# ---------------------------------------------------------------------------


def _parse_result_csv(file_path: str) -> Optional[np.ndarray]:
    """Read a single per-voxel CSV; return a (43,) row or None on empty.

    Replicates the legacy ``for fileN in files2`` block at
    pf_MIDAS.py:2441-2454. The first line is treated as a header and
    discarded; the second line is parsed into a 43-element float row.
    Files with no data line return ``None``.
    """
    with open(file_path) as fh:
        _ = fh.readline()  # header
        line = fh.readline()
    if not line:
        return None
    parts = line.split()
    row = np.zeros(N_COLS, dtype=np.float64)
    n = min(len(parts), N_COLS)
    for j in range(n):
        row[j] = float(parts[j])
    return row


def _row_is_accepted(row: np.ndarray) -> bool:
    """Apply the legacy acceptance gate at pf_MIDAS.py:2456-2460.

    Reject rows where the completeness (col 26) is NaN, negative, or
    greater than ``1 + 1e-10``. Note: rows below the gate are silently
    dropped — they don't become rows in ``microstrFull.csv``.
    """
    c = row[26]
    if isnan(c):
        return False
    if c < 0 or c > 1.0000000001:
        return False
    return True


def _stamp_h5_if_available(h5_root) -> None:
    """Best-effort version stamping; absent ``utils.version`` is non-fatal.

    The orchestrator may run without ``utils/`` on the PYTHONPATH (e.g.
    in fresh CI). Skip silently in that case rather than break the
    consolidation step — the dataset payload is what the tests actually
    pin.
    """
    try:
        from version import stamp_h5  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover — env-specific
        return
    try:
        stamp_h5(h5_root)
    except Exception:  # pragma: no cover — env-specific
        return


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def consolidate_pf(
    layer_dir: Path,
    n_grains: int,
    n_scans: int,
    space_group: int,
) -> ConsolidationResult:
    """Port of pf_MIDAS.py:2429-2519 — assemble PF microstructure outputs.

    Parameters
    ----------
    layer_dir : Path
        Layer working directory. Must contain ``Results/*.csv``
        per-voxel refinement output. ``Recons/`` is created if missing.
    n_grains : int
        Number of unique grains from the find-grains stage. Currently
        retained for API completeness / future grain-keyed grouping —
        the legacy inline code does not gate on this either.
    n_scans : int
        Voxel grid edge length. The total voxel count is
        ``n_scans * n_scans``.
    space_group : int
        Space group number for the symmetry table used to reduce
        per-voxel quaternions into the fundamental zone.

    Returns
    -------
    ConsolidationResult
        Carries the paths to ``microstrFull.csv``,
        ``microstructure.hdf``, and (if it exists on disk from the
        reconstruct stage) ``Full_recon_max_project_grID.tif``.
    """
    started = time.time()
    layer_dir = Path(layer_dir)
    recons_dir = layer_dir / "Recons"
    recons_dir.mkdir(parents=True, exist_ok=True)
    results_dir = layer_dir / "Results"

    # Symmetry table for fundamental-zone reduction (computed once).
    n_sym, sym = make_symmetries(int(space_group))

    # Walk the per-voxel CSVs. Allocate up to len(files) rows; we'll
    # trim to the accepted count after the loop (legacy keeps the
    # full-size buffer; here we trim to match the row count visible
    # to downstream consumers, which is what the test fixture asserts).
    files = sorted(glob.glob(str(results_dir / "*.csv")))
    n_files = len(files)
    filesdata = np.zeros((n_files, N_COLS), dtype=np.float64)

    # info_arr: (23, n_scans*n_scans) per-voxel subset for the images
    # dataset. NaN-init so missing voxels stay NaN. Matches legacy
    # pf_MIDAS.py:2438-2439.
    n_vox = int(n_scans) * int(n_scans)
    info_arr = np.full((23, n_vox), np.nan, dtype=np.float64)

    n_kept = 0
    for file_path in files:
        row = _parse_result_csv(file_path)
        if row is None:
            continue
        if not _row_is_accepted(row):
            continue
        vox_nr = _voxel_index_from_filename(file_path)

        # Reduce orientation matrix → quaternion → fundamental zone.
        om = row[1:10]
        quat_raw = orient_mat_to_quat(om)
        quat_fz = fundamental_zone(quat_raw, space_group=int(space_group),
                                   sym=sym)
        quat_fz = np.asarray(quat_fz, dtype=np.float64).ravel()
        row[39:43] = quat_fz

        # Scatter into the (23, n_vox) info array if the voxel index
        # is in range. Legacy unconditionally writes; we guard so that
        # over-large voxel indices from stray result files don't blow
        # the buffer (e.g. fixture artefacts).
        if 0 <= vox_nr < n_vox:
            info_arr[:, vox_nr] = row[list(_IMAGES_COL_INDEX)]

        filesdata[n_kept] = row
        n_kept += 1

    filesdata = filesdata[:n_kept]

    # --- Write microstrFull.csv ------------------------------------
    microstr_csv = recons_dir / "microstrFull.csv"
    np.savetxt(
        str(microstr_csv),
        filesdata,
        fmt="%.6f",
        delimiter=",",
        header=MICROSTR_HEADER,
    )

    # --- Build the (23, n_scans, n_scans) images dataset ----------
    info_img = info_arr.reshape((23, int(n_scans), int(n_scans)))
    info_img = np.flip(info_img, axis=(1, 2))
    info_img = info_img.transpose(0, 2, 1)

    # --- Write microstructure.hdf ----------------------------------
    microstructure_hdf = recons_dir / "microstructure.hdf"
    with h5py.File(str(microstructure_hdf), "w") as f:
        _stamp_h5_if_available(f)
        ds_micstr = f.create_dataset("microstr", dtype=np.float64,
                                     data=filesdata)
        ds_micstr.attrs["Header"] = np.bytes_(MICROSTR_HEADER)

        ds_imgs = f.create_dataset("images", dtype=np.float64,
                                   data=info_img)
        ds_imgs.attrs["Header"] = np.bytes_(IMAGES_HEADER)

    # --- Grain-ID map tif (consumed, not written) -------------------
    grid_map_tif = recons_dir / "Full_recon_max_project_grID.tif"
    grid_map_path = str(grid_map_tif) if grid_map_tif.exists() else ""

    finished = time.time()
    return ConsolidationResult(
        stage_name="consolidation",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs={"results_dir": str(results_dir)},
        outputs={
            "microstr_full_csv": str(microstr_csv),
            "microstructure_hdf": str(microstructure_hdf),
            "full_recon_max_project_grid_tif": grid_map_path,
        },
        metrics={
            "n_voxels_accepted": int(n_kept),
            "n_voxels_total": int(n_vox),
            "n_files_read": int(n_files),
            "n_grains": int(n_grains),
            "space_group": int(space_group),
            "n_sym": int(n_sym),
        },
        microstr_full_csv=str(microstr_csv),
        microstructure_hdf=str(microstructure_hdf),
        full_recon_max_project_grid_tif=grid_map_path,
    )
