"""Tests for seeding/merge_all.py (Stage B of merged-FF seeding).

Verifies the thin wrapper around merge_scans:
- Drives merge_scans with n_merges == n_scans → 1 fin scan.
- Copies/renames the merged output to ``InputAllExtraInfoFittingAll.csv``
  (the FF indexer's default RefinementFileName).
- Returns a MergeScansResult that points at the renamed file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.seeding.merge_all import merge_all_scans

# merge_scans emits 16-col rows (FF nine + per-scan extras); not exported
# as a named constant from the module.
N_COLS_OUT = 16


def _write_scan_csv(path: Path, rows: np.ndarray) -> None:
    """Write a synthetic 18-col original_InputAll*.csv."""
    header = " ".join(f"col{i}" for i in range(18)) + "\n"
    with path.open("w") as fp:
        fp.write(header)
        for row in rows:
            fp.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def _make_18col(y, z, omega, weight, spot_id, ring):
    row = np.zeros(18, dtype=np.float64)
    row[0] = y
    row[1] = z
    row[2] = omega
    row[3] = weight       # GrainRadius — used as weight
    row[4] = spot_id
    row[5] = ring
    row[16] = 100.0 + ring
    row[17] = 200.0 + ring
    return row


def _setup_scenario(tmp_path: Path, n_scans: int = 4) -> Path:
    positions = np.arange(n_scans, dtype=np.float64)
    np.savetxt(tmp_path / "original_positions.csv", positions, fmt="%.4f")
    for i in range(n_scans):
        # One shared "ring 1" spot + one unique "ring 2" spot per scan.
        common = _make_18col(1.0, 2.0, 3.0, 1.0, 1, 1.0)
        unique = _make_18col(10.0 + i, 20.0 + i, 30.0 + i, 1.0, 2, 2.0)
        rows = np.vstack([common, unique])
        _write_scan_csv(
            tmp_path / f"original_InputAllExtraInfoFittingAll{i}.csv", rows,
        )
    return tmp_path


def test_merge_all_produces_single_ff_input_csv(tmp_path: Path):
    layer_dir = _setup_scenario(tmp_path, n_scans=4)
    result = merge_all_scans(
        layer_dir=layer_dir, n_scans=4, tol_px=0.5, tol_ome=0.5,
    )
    # 1 fin scan (since n_merges == n_scans).
    assert result.metrics["n_fin_scans"] == 1
    # The renamed FF-input file should exist.
    out = layer_dir / "InputAllExtraInfoFittingAll.csv"
    assert out.exists()
    arr = np.loadtxt(out, skiprows=1)
    # 4 unique + 1 common merged = 5 spots.
    assert arr.shape == (5, N_COLS_OUT)


def test_merge_all_result_points_at_renamed_csv(tmp_path: Path):
    layer_dir = _setup_scenario(tmp_path, n_scans=3)
    result = merge_all_scans(
        layer_dir=layer_dir, n_scans=3, tol_px=0.5, tol_ome=0.5,
    )
    assert result.merged_csv == layer_dir / "InputAllExtraInfoFittingAll.csv"
    assert str(layer_dir / "InputAllExtraInfoFittingAll.csv") in result.outputs


def test_merge_all_respects_custom_ff_input_name(tmp_path: Path):
    layer_dir = _setup_scenario(tmp_path, n_scans=2)
    result = merge_all_scans(
        layer_dir=layer_dir, n_scans=2, tol_px=0.5, tol_ome=0.5,
        ff_input_name="MergedFF.csv",
    )
    assert (layer_dir / "MergedFF.csv").exists()
    assert result.merged_csv == layer_dir / "MergedFF.csv"


def test_merge_all_positions_csv_has_single_row(tmp_path: Path):
    """4 scans with positions [0, 1, 2, 3] → averaged position 1.5
    in the single-row positions.csv."""
    layer_dir = _setup_scenario(tmp_path, n_scans=4)
    merge_all_scans(layer_dir=layer_dir, n_scans=4,
                    tol_px=0.5, tol_ome=0.5)
    pos = np.loadtxt(layer_dir / "positions.csv")
    assert pos.shape == ()            # 0-d (single value)
    assert float(pos) == pytest.approx((0 + 1 + 2 + 3) / 4)
