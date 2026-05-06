"""Phase 3 conflict-resolution tests."""

from __future__ import annotations

import numpy as np
import pytest

from midas_process_grains.compute.conflict import (
    PerHklClaim,
    ResolvedClaim,
    resolve_conflicts,
)


def test_unique_claim_resolves_to_that_spot_id():
    aligned_col0 = np.array([[101, 0, 0]], dtype=np.int64)
    aligned_col1 = np.array([[0.05, 0.0, 0.0]], dtype=np.float64)
    out = resolve_conflicts(aligned_col0, aligned_col1)
    assert len(out) == 1
    assert out[0].spot_id == 101
    assert out[0].policy_used == "unique"


def test_majority_wins():
    """3 members, two claim 101 at hkl 0, one claims 999."""
    aligned_col0 = np.array([
        [101, 0, 0],
        [101, 0, 0],
        [999, 0, 0],
    ], dtype=np.int64)
    aligned_col1 = np.zeros_like(aligned_col0, dtype=np.float64)
    out = resolve_conflicts(aligned_col0, aligned_col1)
    assert len(out) == 1
    assert out[0].spot_id == 101
    assert out[0].policy_used == "majority"
    assert out[0].n_supporters == 2
    assert out[0].n_total_claims == 3


def test_tie_breaks_by_minimum_residual():
    """Two members, two distinct SpotIDs, equal vote count → break on Δω."""
    aligned_col0 = np.array([
        [101, 0, 0],
        [999, 0, 0],
    ], dtype=np.int64)
    aligned_col1 = np.array([
        [0.05, 0.0, 0.0],
        [0.30, 0.0, 0.0],
    ], dtype=np.float64)
    out = resolve_conflicts(aligned_col0, aligned_col1)
    assert len(out) == 1
    assert out[0].spot_id == 101              # smaller |Δω|
    assert out[0].policy_used == "residual_tie"


def test_no_claim_means_no_output_row():
    aligned_col0 = np.zeros((2, 3), dtype=np.int64)
    aligned_col1 = np.zeros((2, 3), dtype=np.float64)
    out = resolve_conflicts(aligned_col0, aligned_col1)
    assert out == []


def test_forward_sim_overrides_tie():
    aligned_col0 = np.array([
        [101, 0, 0],
        [999, 0, 0],
    ], dtype=np.int64)
    aligned_col1 = np.array([
        [0.10, 0.0, 0.0],
        [0.10, 0.0, 0.0],
    ], dtype=np.float64)
    # Force the simulator to pick 999.
    sim = lambda k, claims: 999
    out = resolve_conflicts(
        aligned_col0, aligned_col1,
        policy="forward_sim",
        forward_sim_fn=sim,
    )
    assert len(out) == 1
    assert out[0].spot_id == 999
    assert out[0].policy_used == "forward_sim"


def test_unrecognized_policy_raises():
    with pytest.raises(ValueError, match="policy"):
        resolve_conflicts(
            np.zeros((1, 1), dtype=np.int64),
            np.zeros((1, 1), dtype=np.float64),
            policy="random",
        )


def test_multiple_hkls_each_resolved_independently():
    """3 members across 4 hkls, each hkl exercises a different policy branch.
    Rows are members (3); columns are hkl rows (4)."""
    aligned_col0 = np.array([
        # hkl: 0    1    2    3
        [101,  0,  200, 300],   # member 0
        [  0,  0,  200, 999],   # member 1
        [  0, 555, 200, 300],   # member 2
    ], dtype=np.int64)
    aligned_col1 = np.array([
        [0.01, 0.0, 0.05, 0.10],
        [0.0,  0.0, 0.06, 0.20],
        [0.0,  0.04, 0.07, 0.50],
    ], dtype=np.float64)
    out = resolve_conflicts(aligned_col0, aligned_col1)
    by_row = {r.hkl_row: r for r in out}
    # hkl 0: only member 0 claims 101 → unique
    assert by_row[0].spot_id == 101
    assert by_row[0].policy_used == "unique"
    # hkl 1: only member 2 claims 555 → unique
    assert by_row[1].spot_id == 555
    assert by_row[1].policy_used == "unique"
    # hkl 2: all 3 members claim 200 → unique (one distinct id)
    assert by_row[2].spot_id == 200
    assert by_row[2].policy_used == "unique"
    assert by_row[2].n_supporters == 3
    # hkl 3: 300 (members 0,2) vs 999 (member 1) → 300 wins by majority
    assert by_row[3].spot_id == 300
    assert by_row[3].policy_used == "majority"
    assert by_row[3].n_supporters == 2
    assert by_row[3].n_total_claims == 3
