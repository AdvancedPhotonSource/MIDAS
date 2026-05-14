"""Numba-jit ↔ pure-Python parity tests for merge_scans.

Per the plan §11c "Differentiable + multi-device" contract: this stage
isn't a torch-flowable compute kernel (it's pointer-chasing with hard
branches), so the contract demands a numba-jit inner loop **plus** a
pure-Python fallback for environments without numba — and the two paths
must agree bit-for-bit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages import merge_scans as ms_mod
from midas_pipeline.stages.merge_scans import (
    _merge_inner_nb,
    _merge_inner_py,
    _NUMBA_AVAILABLE,
    merge_scans,
)


# ---------------------------------------------------------------------------
# Fixture helpers (kept independent of test_merge_scans.py's helpers so
# this file is self-contained)
# ---------------------------------------------------------------------------


def _make_18col(y, z, omega, weight, spot_id, ring, *, fill=0.0):
    row = np.full(18, fill, dtype=np.float64)
    row[0] = y
    row[1] = z
    row[2] = omega
    row[3] = weight
    row[4] = spot_id
    row[5] = ring
    row[16] = 100.0 + ring
    row[17] = 200.0 + ring
    return row


def _write_18col_csv(path: Path, rows: np.ndarray) -> None:
    header = " ".join(f"col{i}" for i in range(18)) + "\n"
    with path.open("w") as fp:
        fp.write(header)
        for row in rows:
            fp.write(" ".join(f"{v:.6f}" for v in row))
            fp.write("\n")


def _three_scan_fixture(tmp_path: Path) -> tuple[list[Path], np.ndarray]:
    paths = []
    for s in range(3):
        rows = []
        for grain in (1, 2):
            for spot in range(3):
                y = 10.0 * grain + 1.0 * spot
                z = 20.0 * grain + 0.5 * spot
                omega = 5.0 * grain + 2.0 * spot
                ring = float(grain)
                spot_id = 100 * grain + spot + 1
                rows.append(_make_18col(y, z, omega, 1.0, spot_id, ring))
        p = tmp_path / f"InputAllExtraInfoFittingAll{s}.csv"
        _write_18col_csv(p, np.array(rows))
        paths.append(p)
    return paths, np.array([0.0, 2.0, 4.0])


# ---------------------------------------------------------------------------
# Inner-loop parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _NUMBA_AVAILABLE or _merge_inner_nb is None,
    reason="numba not installed",
)
def test_inner_loop_numba_matches_pure_python_synthetic():
    """numba and python inner loops produce identical all_spots arrays."""
    rng = np.random.default_rng(0)

    # Build a synthetic accumulator + this_spots set with mix of matches
    # and non-matches.
    n_initial = 6
    all_nb = np.zeros((50, 16), dtype=np.float64)
    all_py = np.zeros((50, 16), dtype=np.float64)
    for k in range(n_initial):
        for col in range(16):
            v = rng.standard_normal()
            all_nb[k, col] = v
            all_py[k, col] = v
        # Ring numbers are integers (col 5).
        ring = float((k % 2) + 1)
        all_nb[k, 5] = ring
        all_py[k, 5] = ring
        # Positive weights so the average is well-defined.
        w = float(rng.uniform(0.1, 2.0))
        all_nb[k, 3] = w
        all_py[k, 3] = w

    last = np.arange(n_initial, dtype=np.int64)

    # this_spots: half match an existing row within tol, half are new.
    n_this = 8
    this = np.zeros((n_this, 16), dtype=np.float64)
    for i in range(n_this):
        match_idx = i % n_initial
        target = all_nb[match_idx]
        if i < 4:
            # Match: same ring, small offsets.
            this[i, 0] = target[0] + rng.uniform(-0.001, 0.001)
            this[i, 1] = target[1] + rng.uniform(-0.001, 0.001)
            this[i, 2] = target[2] + rng.uniform(-0.001, 0.001)
            this[i, 5] = target[5]
        else:
            # Non-match: deliberately far away on omega.
            this[i, 0] = target[0]
            this[i, 1] = target[1]
            this[i, 2] = target[2] + 100.0
            this[i, 5] = target[5]
        for col in (3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            this[i, col] = rng.standard_normal()
        this[i, 3] = abs(this[i, 3]) + 0.1     # positive weight

    this_idx_nb = np.zeros(n_this, dtype=np.int64)
    this_idx_py = np.zeros(n_this, dtype=np.int64)

    n_all_nb = _merge_inner_nb(
        all_nb, n_initial, last, n_initial, this, n_this, this_idx_nb,
        tol_px=0.01, tol_ome=0.01,
    )
    n_all_py = _merge_inner_py(
        all_py, n_initial, last, n_initial, this, n_this, this_idx_py,
        tol_px=0.01, tol_ome=0.01,
    )

    assert n_all_nb == n_all_py
    np.testing.assert_array_equal(this_idx_nb, this_idx_py)
    np.testing.assert_allclose(all_nb, all_py, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# End-to-end parity via merge_scans()
# ---------------------------------------------------------------------------


def test_merge_scans_numba_matches_pure_python(tmp_path):
    """Full ``merge_scans`` pipeline: numba on vs off → identical outputs."""
    paths, sp = _three_scan_fixture(tmp_path)

    nb_dir = tmp_path / "nb"
    nb_dir.mkdir()
    for p in paths:
        shutil.copy(p, nb_dir / p.name)
    nb_paths = [nb_dir / p.name for p in paths]

    py_dir = tmp_path / "py"
    py_dir.mkdir()
    for p in paths:
        shutil.copy(p, py_dir / p.name)
    py_paths = [py_dir / p.name for p in paths]

    original = ms_mod.MERGE_SCANS_USE_NUMBA
    try:
        ms_mod.MERGE_SCANS_USE_NUMBA = True
        summary_nb = merge_scans(
            per_scan_csvs=nb_paths,
            scan_positions=sp,
            tol_px=0.5, tol_ome=0.5, n_merges=3,
            out_dir=nb_dir,
        )
        ms_mod.MERGE_SCANS_USE_NUMBA = False
        summary_py = merge_scans(
            per_scan_csvs=py_paths,
            scan_positions=sp,
            tol_px=0.5, tol_ome=0.5, n_merges=3,
            out_dir=py_dir,
        )
    finally:
        ms_mod.MERGE_SCANS_USE_NUMBA = original

    assert summary_nb.n_spots_out == summary_py.n_spots_out
    assert summary_nb.n_spots_in == summary_py.n_spots_in
    arr_nb = np.loadtxt(summary_nb.out_csvs[0], skiprows=1)
    arr_py = np.loadtxt(summary_py.out_csvs[0], skiprows=1)
    np.testing.assert_allclose(arr_nb, arr_py, rtol=0, atol=0)
    np.testing.assert_allclose(
        np.loadtxt(summary_nb.positions_csv),
        np.loadtxt(summary_py.positions_csv),
        rtol=0, atol=0,
    )


def test_merge_scans_pure_python_fallback_completes(tmp_path):
    """Force-disable numba and verify the fallback path runs to completion."""
    paths, sp = _three_scan_fixture(tmp_path)
    original = ms_mod.MERGE_SCANS_USE_NUMBA
    try:
        ms_mod.MERGE_SCANS_USE_NUMBA = False
        summary = merge_scans(
            per_scan_csvs=paths,
            scan_positions=sp,
            tol_px=0.5, tol_ome=0.5, n_merges=3,
            out_dir=tmp_path / "out",
        )
    finally:
        ms_mod.MERGE_SCANS_USE_NUMBA = original
    assert summary.n_spots_out == 6
