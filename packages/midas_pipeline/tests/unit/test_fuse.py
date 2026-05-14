"""Unit tests for midas_pipeline.fuse.

Two scenarios:

- ``test_bayesian_fusion_boundary_disambiguates`` — a synthetic 2-grain
  shape map with a boundary voxel; orientation likelihood favors one
  grain at the boundary. Assert the posterior picks the right grain.
- ``test_mask_sino_friedel_keeps_both`` — synthetic Friedel pair; the
  dual-sign filter must keep cells satisfying either branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.fuse import bayesian_fusion, mask_sino_by_assignment


def _write_indexbest_fixture(tmp_path: Path, n_vox: int, sols_per_vox, all_records):
    out_dir = tmp_path / "Output"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_sol_arr = np.asarray(sols_per_vox, dtype=np.int32)
    header_bytes = 4 + 4 * n_vox + 8 * n_vox
    off_arr = np.empty(n_vox, dtype=np.int64)
    cum = 0
    for v in range(n_vox):
        off_arr[v] = header_bytes + cum * 8
        cum += n_sol_arr[v] * 16

    flat_records = np.asarray(all_records, dtype=np.float64).reshape(-1)
    path = out_dir / "IndexBest_all.bin"
    with open(path, "wb") as f:
        np.asarray([n_vox], dtype=np.int32).tofile(f)
        n_sol_arr.tofile(f)
        off_arr.tofile(f)
        flat_records.tofile(f)
    return path


def _write_unique_orientations(tmp_path: Path, oms_3x3):
    rows = []
    for g, om in enumerate(oms_3x3):
        rows.append([float(g), 0.0, 0.0, 0.0, 0.0] + list(np.asarray(om).flatten()))
    data = np.asarray(rows, dtype=np.float64)
    path = tmp_path / "UniqueOrientations.csv"
    np.savetxt(path, data, fmt="%.10f", delimiter=" ")
    return path


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_bayesian_fusion_boundary_disambiguates(tmp_path):
    """At a voxel where shape favors grain 0 strongly, the fused posterior
    should favor grain 0 even if grain 1 has a similar orient-score there."""
    n_scans = 2
    n_vox = 4
    om_g0 = np.eye(3)
    om_g1 = _rot_z(np.deg2rad(30.0))

    def _record(om, conf):
        r = np.zeros(16, dtype=np.float64)
        r[2:11] = np.asarray(om).flatten()
        r[14] = 10.0
        r[15] = 10.0 * conf
        return r

    # Each voxel has two candidates (one matching each grain). Voxel 0 is
    # the "boundary": both grains have similar orient-score.
    records = [
        # v0 (boundary): candidates for both grains
        _record(om_g0, 0.8), _record(om_g1, 0.8),
        # v1: only grain 0 candidate
        _record(om_g0, 0.9),
        # v2: only grain 1 candidate
        _record(om_g1, 0.95),
        # v3: only grain 1 candidate
        _record(om_g1, 0.85),
    ]
    sols = [2, 1, 1, 1]
    _write_indexbest_fixture(tmp_path, n_vox, sols, records)
    _write_unique_orientations(tmp_path, [om_g0, om_g1])

    # Per-grain shape (recon): grain 0 occupies upper half, grain 1 the lower.
    # The boundary voxel v0 has stronger shape support for grain 0.
    shape = np.zeros((2, n_scans, n_scans), dtype=np.float32)
    shape[0, 0, 0] = 1.0       # v0 — boundary
    shape[0, 0, 1] = 0.9       # v1
    shape[1, 1, 0] = 1.0       # v2
    shape[1, 1, 1] = 0.95      # v3

    posterior = bayesian_fusion(
        shape, tmp_path, sgnum=225, n_grains=2,
        max_ang_deg=1.0, min_conf=0.5,
    )
    # At the boundary v0, posterior[grain 0] should win because of shape.
    assert posterior[0, 0, 0] > posterior[1, 0, 0]
    # v2 should still be grain 1 (no grain 0 candidate there).
    assert posterior[1, 1, 0] > posterior[0, 1, 0]


def test_mask_sino_friedel_keeps_both():
    """A Friedel pair: a voxel V and its omega+180° image should both pass
    the mask. The s_proj for omega and omega+180° are negatives of each other,
    so the dual-sign filter (|s − ypos| < tol or |−s − ypos| < tol) must
    keep both."""
    n_grains = 1
    n_scans = 3
    max_nhkl = 2
    # spatial_pos: [−1, 0, 1] um
    spatial_pos = np.array([-1.0, 0.0, 1.0])

    # One grain assigned to voxel (row=2, col=0) → y=spatial_pos[2]=1, x=spatial_pos[0]=-1
    max_id = -np.ones((n_scans, n_scans), dtype=np.int32)
    max_id[2, 0] = 0

    # Two HKLs at omega=0° and omega=180° (Friedel pair).
    # s = -x*cos(omega) + y*sin(omega)
    # omega=0:   s = -x*1 + y*0 = -(-1) = 1
    # omega=180: s = -x*(-1) + y*0 = -1
    omegas = np.array([[0.0, 180.0]])

    # Build sinos that put intensity at the scan position the Friedel pair maps to
    sinos = np.zeros((n_grains, max_nhkl, n_scans), dtype=np.float64)
    sinos[0, :, :] = 1.0     # every scan filled — mask will pick out the ones that match
    nr_hkls = np.array([2], dtype=np.int32)

    masked = mask_sino_by_assignment(
        sinos, omegas, nr_hkls, max_id, n_grains, n_scans,
        spatial_pos, scan_tol=0.5,
    )
    # HKL0 (omega=0): s_proj=1, |1 - scan_pos| < 0.5 → scan_pos=1 only
    # HKL1 (omega=180): s_proj=-1, |-1 - scan_pos|<0.5 → scan_pos=-1 only
    # Friedel filter also adds |-s_proj - scan_pos| < 0.5, so for HKL0
    # |−1 − scan_pos| < 0.5 → scan_pos=−1 also kept; HKL1 similarly keeps scan_pos=1.
    # → both rows should have BOTH endpoints kept.
    assert masked[0, 0, 0] > 0   # HKL0, scan_pos=-1 kept by Friedel branch
    assert masked[0, 0, 2] > 0   # HKL0, scan_pos=1 kept by primary branch
    assert masked[0, 1, 0] > 0   # HKL1, scan_pos=-1 kept by primary branch
    assert masked[0, 1, 2] > 0   # HKL1, scan_pos=1 kept by Friedel branch
    # Middle scan (scan_pos=0) is not within tol of ±1, so it should be zero.
    assert masked[0, 0, 1] == 0
    assert masked[0, 1, 1] == 0
