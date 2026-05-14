"""Unit tests for the pure-Python port of mergeScansScanning.c."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.merge_scans import (
    MergeScansSummary,
    _merge_inner_py,
    _read_per_scan_csv_16,
    merge_scans,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_18col(y, z, omega, weight, spot_id, ring, *, fill=0.0):
    """Synthesize one 18-col input row. The C reader keeps cols [0..13, 16, 17]
    (16 cols total), dropping cols 14 and 15.
    """
    row = np.full(18, fill, dtype=np.float64)
    row[0] = y
    row[1] = z
    row[2] = omega
    row[3] = weight
    row[4] = spot_id
    row[5] = ring
    row[16] = 100.0 + ring     # tag — should appear at output col 14
    row[17] = 200.0 + ring     # tag — should appear at output col 15
    return row


def _make_16col(y, z, omega, weight, spot_id, ring):
    """Same as _make_18col but already in the 16-col output layout."""
    row = _make_18col(y, z, omega, weight, spot_id, ring)
    # Apply 18→16 mapping: keep [0..13], replace 14+15 with 16+17.
    return np.concatenate([row[:14], [row[16], row[17]]])


def _write_18col_csv(path: Path, rows: np.ndarray) -> None:
    header = " ".join(f"col{i}" for i in range(18)) + "\n"
    with path.open("w") as fp:
        fp.write(header)
        for row in rows:
            fp.write(" ".join(f"{v:.6f}" for v in row))
            fp.write("\n")


# ---------------------------------------------------------------------------
# CSV reader — 18→16 column drop semantics
# ---------------------------------------------------------------------------


def test_read_csv_drops_input_cols_14_15(tmp_path):
    row = _make_18col(1.1, 2.2, 3.3, 0.5, 42, 1.0)
    row[14] = 999.0      # dummy — must NOT appear in output
    row[15] = 888.0
    p = tmp_path / "in0.csv"
    _write_18col_csv(p, row.reshape(1, 18))
    arr = _read_per_scan_csv_16(p)
    assert arr.shape == (1, 16)
    assert arr[0, 0] == pytest.approx(1.1)
    assert arr[0, 5] == pytest.approx(1.0)
    # Output cols 14, 15 come from input cols 16, 17.
    assert arr[0, 14] == pytest.approx(101.0)
    assert arr[0, 15] == pytest.approx(201.0)


# ---------------------------------------------------------------------------
# Inner-loop semantics (C-fidelity, pure-Python path)
# ---------------------------------------------------------------------------


def test_inner_loop_appends_when_ring_mismatches():
    """Different ring → no match even if y/z/ω are close."""
    all_spots = np.zeros((10, 16))
    all_spots[0] = _make_16col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
    n_all = 1
    last_idx = np.array([0], dtype=np.int64)

    this = np.zeros((1, 16))
    this[0] = _make_16col(1.0, 2.0, 3.0, 1.0, 99, 2.0)
    this_idx_out = np.zeros(1, dtype=np.int64)

    n_all_new = _merge_inner_py(
        all_spots, n_all, last_idx, 1, this, 1, this_idx_out,
        tol_px=0.5, tol_ome=0.5,
    )
    assert n_all_new == 2
    assert this_idx_out[0] == 1


def test_inner_loop_weighted_average_on_match():
    """Same ring + within tolerance → weight-averaged in place."""
    all_spots = np.zeros((10, 16))
    all_spots[0] = _make_16col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
    n_all = 1
    last_idx = np.array([0], dtype=np.int64)

    this = np.zeros((1, 16))
    this[0] = _make_16col(1.001, 2.001, 3.001, 3.0, 99, 1.0)
    this_idx_out = np.zeros(1, dtype=np.int64)

    n_all_new = _merge_inner_py(
        all_spots, n_all, last_idx, 1, this, 1, this_idx_out,
        tol_px=0.1, tol_ome=0.1,
    )
    assert n_all_new == 1
    assert this_idx_out[0] == 0
    # Weighted average: w1=1, w2=3 → col 0 = (1*1.0 + 3*1.001) / 4
    assert all_spots[0, 0] == pytest.approx((1.0 + 3 * 1.001) / 4)


def test_inner_loop_omega_outside_tol_appends():
    all_spots = np.zeros((10, 16))
    all_spots[0] = _make_16col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
    n_all = 1
    last_idx = np.array([0], dtype=np.int64)

    this = np.zeros((1, 16))
    this[0] = _make_16col(1.0, 2.0, 5.0, 1.0, 99, 1.0)
    this_idx_out = np.zeros(1, dtype=np.int64)

    n_all_new = _merge_inner_py(
        all_spots, n_all, last_idx, 1, this, 1, this_idx_out,
        tol_px=0.5, tol_ome=0.5,
    )
    assert n_all_new == 2


# ---------------------------------------------------------------------------
# End-to-end merge_scans()
# ---------------------------------------------------------------------------


def _setup_scenario(tmp_path: Path, n_scans: int = 4):
    """Write n_scans CSVs at positions 0..n_scans-1, each with one
    'common' ring-1 spot (identical across scans → will merge) and one
    'unique' ring-2 spot at a per-scan offset → will append."""
    positions = np.arange(n_scans, dtype=np.float64)
    paths = []
    for i in range(n_scans):
        common = _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
        unique = _make_18col(10.0 + i, 20.0 + i, 30.0 + i, 1.0, 2, 2.0)
        rows = np.vstack([common, unique])
        p = tmp_path / f"in_{i}.csv"
        _write_18col_csv(p, rows)
        paths.append(p)
    return paths, positions


def test_merge_all_consolidates_common_spot(tmp_path):
    paths, positions = _setup_scenario(tmp_path, n_scans=4)
    summary = merge_scans(
        paths, positions,
        tol_px=0.5, tol_ome=0.5, n_merges=4,
        out_dir=tmp_path / "out",
    )
    assert isinstance(summary, MergeScansSummary)
    assert summary.n_spots_in == 8
    assert len(summary.out_csvs) == 1
    arr = np.loadtxt(summary.out_csvs[0], skiprows=1)
    assert arr.shape == (5, 16)
    # The merged common spot has Y close to 1.0; the 4 uniques are at 10..13.
    assert int((np.abs(arr[:, 0] - 1.0) < 1e-9).sum()) == 1
    np.testing.assert_array_equal(arr[:, 4], np.arange(1, 6))


def test_positions_averaged(tmp_path):
    paths, positions = _setup_scenario(tmp_path, n_scans=4)
    summary = merge_scans(
        paths, positions,
        tol_px=0.5, tol_ome=0.5, n_merges=4,
        out_dir=tmp_path / "out",
    )
    new_positions = np.loadtxt(summary.positions_csv)
    assert float(new_positions) == pytest.approx((0 + 1 + 2 + 3) / 4)


def test_n_merges_2_yields_two_outputs(tmp_path):
    paths, positions = _setup_scenario(tmp_path, n_scans=4)
    summary = merge_scans(
        paths, positions,
        tol_px=0.5, tol_ome=0.5, n_merges=2,
        out_dir=tmp_path / "out",
    )
    assert len(summary.out_csvs) == 2
    new_positions = np.loadtxt(summary.positions_csv)
    np.testing.assert_allclose(new_positions, [0.5, 2.5])
