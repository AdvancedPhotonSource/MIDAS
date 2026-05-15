"""Unit tests for the per-voxel scanning refinement orchestrator (P6 batch driver).

Verifies:
- ``_read_index_best_all`` round-trips an empty + a sparse IndexBest_all.bin.
- ``_top_candidate`` picks the highest-completeness row.
- Voxel-grid layout via Cartesian product (mirrors C IndexerScanningOMP.c:1667-1683).
- ``refine_scanning_block`` rejects single-scan inputs (use FF refine_grain).
- ``refine_scanning_block`` voxel sharding rejects invalid block indices.

Heavyweight end-to-end tests (actual refinement on a synthetic) reuse
test_refine_grain.py's fixture and are marked ``slow``; the
scaffolding here is fast enough to run in default CI.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from midas_fit_grain.config import FitConfig
from midas_fit_grain.scan_driver import (
    ScanVoxelResult,
    _read_index_best_all,
    _top_candidate,
    refine_scanning_block,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic IndexBest_all.bin payload
# ---------------------------------------------------------------------------


def _write_index_best_all(path: Path, per_voxel: List[np.ndarray]) -> None:
    """Mirrors midas_index.io.consolidated.write_index_best_all so we don't
    have to import across packages in this unit test."""
    n_voxels = len(per_voxel)
    n_sol = np.array([int(r.shape[0]) for r in per_voxel], dtype=np.int32)
    bytes_per_voxel = (16 * 8) * n_sol.astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(bytes_per_voxel)[:-1]))
    header_size = 4 + 4 * n_voxels + 8 * n_voxels
    off_arr = (header_size + cumulative).astype(np.int64)
    with path.open("wb") as f:
        f.write(np.int32(n_voxels).tobytes())
        f.write(n_sol.tobytes())
        f.write(off_arr.tobytes())
        for rec in per_voxel:
            if rec.shape[0]:
                f.write(np.ascontiguousarray(rec, dtype=np.float64).tobytes())


def _make_candidate(*, n_expected: float, n_matched: float,
                    om_flat: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """One 16-col candidate record."""
    row = np.zeros(16, dtype=np.float64)
    row[0] = 1                # placeholder
    row[1] = 0.0
    row[2:11] = om_flat
    row[11:14] = pos
    row[14] = n_expected
    row[15] = n_matched
    return row


# ---------------------------------------------------------------------------
# Round-trip parsing
# ---------------------------------------------------------------------------


def test_read_index_best_all_roundtrip_empty(tmp_path: Path):
    path = tmp_path / "IndexBest_all.bin"
    _write_index_best_all(path, [np.zeros((0, 16), dtype=np.float64)] * 4)
    n_sol, blocks = _read_index_best_all(path)
    assert len(blocks) == 4
    np.testing.assert_array_equal(n_sol, [0, 0, 0, 0])
    for b in blocks:
        assert b.shape == (0, 16)


def test_read_index_best_all_roundtrip_mixed(tmp_path: Path):
    path = tmp_path / "IndexBest_all.bin"
    cands_v0 = np.stack([
        _make_candidate(n_expected=10, n_matched=5,
                        om_flat=np.eye(3).ravel(), pos=np.array([0, 0, 0])),
        _make_candidate(n_expected=10, n_matched=8,
                        om_flat=np.eye(3).ravel(), pos=np.array([0, 0, 0])),
    ])
    cands_v1 = np.zeros((0, 16))
    cands_v2 = _make_candidate(
        n_expected=12, n_matched=11, om_flat=np.eye(3).ravel(),
        pos=np.array([1, 2, 3]),
    ).reshape(1, 16)
    _write_index_best_all(path, [cands_v0, cands_v1, cands_v2])
    n_sol, blocks = _read_index_best_all(path)
    np.testing.assert_array_equal(n_sol, [2, 0, 1])
    np.testing.assert_array_equal(blocks[0], cands_v0)
    assert blocks[1].shape == (0, 16)
    np.testing.assert_array_equal(blocks[2], cands_v2)


# ---------------------------------------------------------------------------
# Top-candidate pick uses completeness ratio = matched / expected
# ---------------------------------------------------------------------------


def test_top_candidate_picks_highest_completeness():
    block = np.stack([
        _make_candidate(n_expected=10, n_matched=5,
                        om_flat=np.eye(3).ravel(), pos=np.zeros(3)),
        _make_candidate(n_expected=10, n_matched=9,
                        om_flat=np.eye(3).ravel(), pos=np.ones(3)),
        _make_candidate(n_expected=10, n_matched=8,
                        om_flat=np.eye(3).ravel(), pos=np.full(3, 2.0)),
    ])
    cand = _top_candidate(block)
    assert cand is not None
    # cand 1 has 9/10 = 0.9 → highest.
    assert cand[15] == 9.0
    np.testing.assert_array_equal(cand[11:14], [1.0, 1.0, 1.0])


def test_top_candidate_returns_none_on_empty():
    assert _top_candidate(np.zeros((0, 16))) is None


# ---------------------------------------------------------------------------
# refine_scanning_block input validation
# ---------------------------------------------------------------------------


def test_refine_scanning_block_rejects_single_scan(tmp_path: Path):
    cfg = FitConfig(scan_pos_tol_um=2.0)
    np.savetxt(tmp_path / "positions.csv", np.array([0.0]))
    _write_index_best_all(tmp_path / "IndexBest_all.bin",
                          [np.zeros((0, 16), dtype=np.float64)])
    with pytest.raises(ValueError, match="n_scans >= 2"):
        refine_scanning_block(
            cfg,
            index_best_all=tmp_path / "IndexBest_all.bin",
            positions_csv=tmp_path / "positions.csv",
            results_dir=tmp_path / "Results",
            model=None,
            obs=None,
            pred_ring_slot=None,
        )


def test_refine_scanning_block_rejects_voxel_count_mismatch(tmp_path: Path):
    cfg = FitConfig(scan_pos_tol_um=2.0)
    np.savetxt(tmp_path / "positions.csv", np.array([0.0, 5.0]))   # 2 scans → 4 voxels
    _write_index_best_all(
        tmp_path / "IndexBest_all.bin",
        # 3 voxels — mismatched with 4 expected.
        [np.zeros((0, 16), dtype=np.float64) for _ in range(3)],
    )
    with pytest.raises(ValueError, match="voxel count mismatch"):
        refine_scanning_block(
            cfg,
            index_best_all=tmp_path / "IndexBest_all.bin",
            positions_csv=tmp_path / "positions.csv",
            results_dir=tmp_path / "Results",
            model=None, obs=None, pred_ring_slot=None,
        )


def test_refine_scanning_block_rejects_bad_voxel_sharding(tmp_path: Path):
    cfg = FitConfig(scan_pos_tol_um=2.0)
    np.savetxt(tmp_path / "positions.csv", np.array([0.0, 5.0]))   # 2 scans → 4 voxels
    _write_index_best_all(
        tmp_path / "IndexBest_all.bin",
        [np.zeros((0, 16), dtype=np.float64) for _ in range(4)],
    )
    with pytest.raises(ValueError, match="invalid voxel sharding"):
        refine_scanning_block(
            cfg,
            index_best_all=tmp_path / "IndexBest_all.bin",
            positions_csv=tmp_path / "positions.csv",
            results_dir=tmp_path / "Results",
            model=None, obs=None, pred_ring_slot=None,
            voxel_block_nr=2, voxel_n_blocks=2,        # 2 ≥ 2 → invalid
        )


def test_refine_scanning_block_empty_input_returns_empty_list(tmp_path: Path):
    """All voxels empty → no CSVs, no exceptions."""
    cfg = FitConfig(scan_pos_tol_um=2.0)
    np.savetxt(tmp_path / "positions.csv", np.array([0.0, 5.0]))
    _write_index_best_all(
        tmp_path / "IndexBest_all.bin",
        [np.zeros((0, 16), dtype=np.float64) for _ in range(4)],
    )
    out = refine_scanning_block(
        cfg,
        index_best_all=tmp_path / "IndexBest_all.bin",
        positions_csv=tmp_path / "positions.csv",
        results_dir=tmp_path / "Results",
        model=None, obs=None, pred_ring_slot=None,
    )
    assert out == []
    # No files should have been written.
    results_dir = tmp_path / "Results"
    if results_dir.exists():
        assert not any(results_dir.glob("Result_OrientPos_voxel_*.csv"))
