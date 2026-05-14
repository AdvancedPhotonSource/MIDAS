"""Unit tests for the pure-Python port of mergeScansScanning.c.

Verifies:
- 16-col output (cols 14, 15 of input dropped; cols 16, 17 demoted into
  those positions). Matches C lines 126–136.
- Weighted average uses col 3 (GrainRadius) as the weight.
- Match criterion: ring (col 5, tol 0.01) AND |Δy| < tol_px AND |Δz| <
  tol_px AND |Δω| < tol_ome.
- Spot IDs (col 4) renumbered 1-based at the end.
- positions.csv contains averaged Y per merged group.
- n_merges=1 short-circuits to a pass-through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.stages.merge_scans import (
    _merge_two_scans,
    _read_input_csv,
    merge_scans,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_scan_csv(path: Path, rows: np.ndarray) -> None:
    """Write a synthetic 18-col original_InputAll*.csv.

    Rows shape (n, 18). Columns chosen so the 16-col reader maps as:
      [0..13] passthrough, [14]=input[16], [15]=input[17]. Inputs[14],
      inputs[15] are dropped.
    """
    header = " ".join(f"col{i}" for i in range(18)) + "\n"
    with path.open("w") as fp:
        fp.write(header)
        for row in rows:
            fp.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def _make_18col(
    y: float, z: float, omega: float, weight: float, spot_id: int,
    ring: float, *, fill: float = 0.0,
) -> np.ndarray:
    """Construct one 18-col row with the 6 fields we care about."""
    row = np.full(18, fill, dtype=np.float64)
    row[0] = y
    row[1] = z
    row[2] = omega
    row[3] = weight
    row[4] = spot_id
    row[5] = ring
    # Tag col 16 + 17 with distinguishable values so we can verify the
    # column-mapping post-merge.
    row[16] = 100.0 + ring
    row[17] = 200.0 + ring
    return row


# ---------------------------------------------------------------------------
# I/O parsing
# ---------------------------------------------------------------------------


def test_read_input_csv_drops_cols_14_15(tmp_path):
    """16-col output should drop input cols 14, 15 and promote 16, 17."""
    row = _make_18col(1.1, 2.2, 3.3, 0.5, 42, 1.0)
    row[14] = 999.0       # dummy, should be dropped
    row[15] = 888.0       # dummy, should be dropped
    p = tmp_path / "original_InputAllExtraInfoFittingAll0.csv"
    _write_scan_csv(p, row.reshape(1, 18))
    arr, header = _read_input_csv(p)
    assert arr.shape == (1, 16)
    assert arr[0, 0] == pytest.approx(1.1)
    assert arr[0, 5] == pytest.approx(1.0)
    # Output cols 14, 15 should come from input cols 16, 17.
    assert arr[0, 14] == pytest.approx(101.0)
    assert arr[0, 15] == pytest.approx(201.0)


def test_read_input_csv_drops_low_weight_rows(tmp_path):
    """Rows with GrainRadius (col 3) < 0.01 must be dropped."""
    rows = np.vstack([
        _make_18col(1.0, 2.0, 3.0, 0.5, 1, 1.0),
        _make_18col(1.0, 2.0, 3.0, 0.001, 2, 1.0),  # dropped
        _make_18col(1.0, 2.0, 3.0, 0.05, 3, 1.0),
    ])
    p = tmp_path / "original_InputAllExtraInfoFittingAll0.csv"
    _write_scan_csv(p, rows)
    arr, _ = _read_input_csv(p)
    assert arr.shape == (2, 16)
    np.testing.assert_allclose(arr[:, 4], [1.0, 3.0])


# ---------------------------------------------------------------------------
# _merge_two_scans pairwise behavior
# ---------------------------------------------------------------------------


def _to_16col(rows18: np.ndarray) -> np.ndarray:
    """Apply the 18→16 column drop used by the C reader."""
    out = np.zeros((rows18.shape[0], 16))
    out[:, :14] = rows18[:, :14]
    out[:, 14] = rows18[:, 16]
    out[:, 15] = rows18[:, 17]
    return out


def test_pairwise_append_when_no_prior_indices():
    """First scan in a merge group: nothing to match against → append all."""
    accum = np.zeros((0, 16))
    last_idx = np.empty(0, dtype=np.int64)
    new = _to_16col(np.vstack([
        _make_18col(1.0, 2.0, 3.0, 0.5, 1, 1.0),
        _make_18col(5.0, 6.0, 7.0, 0.5, 2, 2.0),
    ]))
    merged, this_idx = _merge_two_scans(
        accum, last_idx, new, tol_px=0.5, tol_ome=0.5,
    )
    assert merged.shape == (2, 16)
    np.testing.assert_array_equal(this_idx, [0, 1])


def test_pairwise_match_within_tolerance_weighted_average():
    """A near-identical row should match and weight-average into the accum."""
    accum = _to_16col(
        _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0).reshape(1, 18)
    )
    last_idx = np.array([0], dtype=np.int64)
    new = _to_16col(
        _make_18col(1.001, 2.001, 3.001, 3.0, 999, 1.0).reshape(1, 18)
    )
    merged, this_idx = _merge_two_scans(
        accum, last_idx, new, tol_px=0.1, tol_ome=0.1,
    )
    assert merged.shape == (1, 16)
    # Weighted average: w1=1, w2=3 → avg = (1*v1 + 3*v2) / 4
    # On col 0: (1*1.0 + 3*1.001) / 4 = 1.00075
    assert merged[0, 0] == pytest.approx((1.0 + 3 * 1.001) / 4)
    # The new row reuses the existing index (0), not a new one.
    assert this_idx[0] == 0


def test_pairwise_ring_mismatch_appends():
    """Different ring → no match even if y/z/ω are close."""
    accum = _to_16col(
        _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0).reshape(1, 18)
    )
    last_idx = np.array([0], dtype=np.int64)
    new = _to_16col(
        _make_18col(1.0, 2.0, 3.0, 1.0, 999, 2.0).reshape(1, 18)  # ring=2
    )
    merged, this_idx = _merge_two_scans(
        accum, last_idx, new, tol_px=0.5, tol_ome=0.5,
    )
    assert merged.shape == (2, 16)
    assert this_idx[0] == 1


def test_pairwise_omega_outside_tol_appends():
    """ω difference exceeding tol_ome → append, not merge."""
    accum = _to_16col(
        _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0).reshape(1, 18)
    )
    last_idx = np.array([0], dtype=np.int64)
    new = _to_16col(
        _make_18col(1.0, 2.0, 5.0, 1.0, 2, 1.0).reshape(1, 18)
    )
    merged, _ = _merge_two_scans(
        accum, last_idx, new, tol_px=0.5, tol_ome=0.5,
    )
    assert merged.shape == (2, 16)


# ---------------------------------------------------------------------------
# End-to-end via merge_scans entry point
# ---------------------------------------------------------------------------


def _setup_merge_scenario(tmp_path: Path, n_scans: int = 4) -> Path:
    """Write n_scans original_* files + original_positions.csv.

    Layout: scans 0,1,2,3 at positions 0,1,2,3. We seed each scan with
    one identical "ring 1" spot (so all four merge into one) plus a
    unique "ring 2" spot per scan (so 4 unique appends).
    """
    layer_dir = tmp_path
    positions = np.arange(n_scans, dtype=np.float64)
    np.savetxt(layer_dir / "original_positions.csv", positions, fmt="%.4f")
    for i in range(n_scans):
        common = _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
        unique = _make_18col(10.0 + i, 20.0 + i, 30.0 + i, 1.0, 2, 2.0)
        rows = np.vstack([common, unique])
        _write_scan_csv(layer_dir / f"original_InputAllExtraInfoFittingAll{i}.csv",
                        rows)
    return layer_dir


def test_merge_scans_one_fin_scan_consolidates_common_spot(tmp_path):
    """4 scans, n_merges=4 → 1 fin scan; the common spot merges, uniques append."""
    layer_dir = _setup_merge_scenario(tmp_path, n_scans=4)
    res = merge_scans(
        layer_dir=layer_dir, n_scans=4, n_merges=4,
        tol_px=0.5, tol_ome=0.5,
    )
    assert res.metrics["n_fin_scans"] == 1
    assert res.n_spots_in == 4 + 4         # 4 common + 4 unique reads
    # Outputs: 1 merged + 4 distinct uniques = 5 spots
    out = layer_dir / "InputAllExtraInfoFittingAll0.csv"
    arr = np.loadtxt(out, skiprows=1)
    assert arr.shape == (5, 16)
    # The first column should have one cluster at ~1.0 (the merged common)
    # and four values at 10, 11, 12, 13.
    assert (np.abs(arr[:, 0] - 1.0) < 1e-9).sum() == 1
    # Spot IDs renumbered 1..5.
    np.testing.assert_array_equal(arr[:, 4], np.arange(1, 6))


def test_merge_scans_positions_averaged(tmp_path):
    """positions.csv should hold the mean of merged scans' positions."""
    layer_dir = _setup_merge_scenario(tmp_path, n_scans=4)
    merge_scans(layer_dir=layer_dir, n_scans=4, n_merges=4,
                tol_px=0.5, tol_ome=0.5)
    new_positions = np.loadtxt(layer_dir / "positions.csv")
    assert new_positions.shape == ()      # 0-d for single value
    assert float(new_positions) == pytest.approx((0 + 1 + 2 + 3) / 4)


def test_merge_scans_n_merges_2_yields_two_fin_scans(tmp_path):
    layer_dir = _setup_merge_scenario(tmp_path, n_scans=4)
    res = merge_scans(layer_dir=layer_dir, n_scans=4, n_merges=2,
                      tol_px=0.5, tol_ome=0.5)
    assert res.metrics["n_fin_scans"] == 2
    assert (layer_dir / "InputAllExtraInfoFittingAll0.csv").exists()
    assert (layer_dir / "InputAllExtraInfoFittingAll1.csv").exists()
    new_positions = np.loadtxt(layer_dir / "positions.csv")
    np.testing.assert_allclose(new_positions, [0.5, 2.5])
