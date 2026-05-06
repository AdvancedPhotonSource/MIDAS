"""Tests for compute/pass_a.py — spot-overlap cluster merge."""
from __future__ import annotations

import math
import numpy as np
import pytest

from midas_process_grains.compute.pass_a import (
    merge_clusters_by_spot_overlap,
    _candidate_pairs_from_spot_overlap,
)


def test_candidate_pairs_basic():
    # cluster 0: spots {1, 2, 3}; cluster 1: {2, 3, 4}; cluster 2: {10, 11}.
    arr = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 3, 4], dtype=np.int64),
        np.array([10, 11], dtype=np.int64),
    ]
    pairs = _candidate_pairs_from_spot_overlap(arr)
    # Only (0, 1) shares spots (2 and 3); cluster 2 is isolated.
    assert pairs.tolist() == [[0, 1]]


def test_candidate_pairs_three_way_overlap():
    arr = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 3, 4], dtype=np.int64),
        np.array([3, 5, 6], dtype=np.int64),
    ]
    pairs = _candidate_pairs_from_spot_overlap(arr)
    # All 3 pairs share at least one spot via spot 3 / 2.
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_pass_a_merges_high_jaccard_pair():
    # Two cluster reps with very similar quats and 80% spot overlap → merge.
    n_seeds = 4
    cluster_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    members_by_label = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([2, 3], dtype=np.int64),
    }
    # Identity-ish quats (both at FZ identity, so misori = 0).
    quats = np.zeros((n_seeds, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    ias = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
    # Build IBF col 0: 4 seeds × 5 hkl slots. Cluster 0 rep is seed 0,
    # cluster 1 rep is seed 2. Make their spot lists overlap heavily.
    ibf_alive_col0 = np.array([
        [101, 102, 103, 104, 105],   # seed 0 (cluster 0 rep)
        [201, 202, 203, 204, 205],   # seed 1
        [101, 102, 103, 104, 999],   # seed 2 (cluster 1 rep) — 4/5 spots match seed 0
        [301, 302, 303, 304, 305],   # seed 3
    ], dtype=np.int64)
    ibf_global_to_local = np.arange(n_seeds, dtype=np.int64)

    new_labels, n_super, new_members = merge_clusters_by_spot_overlap(
        cluster_labels=cluster_labels,
        n_phase1_clusters=2,
        members_by_label=members_by_label,
        quats_per_seed=quats,
        ias_per_seed=ias,
        ibf_alive_col0=ibf_alive_col0,
        ibf_global_to_local=ibf_global_to_local,
        space_group=225,
        misori_tol_rad=math.radians(1.0),
        jaccard_tol=0.5,
        progress=False,
    )
    # 4 of 6 spots overlap → Jaccard = 4/6 = 0.667 ≥ 0.5 → merge.
    assert n_super == 1
    assert new_labels.tolist() == [0, 0, 0, 0]
    assert sorted(new_members[0].tolist()) == [0, 1, 2, 3]


def test_pass_a_does_not_merge_low_jaccard():
    n_seeds = 4
    cluster_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    members_by_label = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([2, 3], dtype=np.int64),
    }
    quats = np.zeros((n_seeds, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    ias = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
    # Only 1 of 5 spots overlap → Jaccard = 1/9 ≈ 0.11 < 0.5 → no merge.
    ibf_alive_col0 = np.array([
        [101, 102, 103, 104, 105],
        [201, 202, 203, 204, 205],
        [101, 999, 998, 997, 996],
        [301, 302, 303, 304, 305],
    ], dtype=np.int64)
    ibf_global_to_local = np.arange(n_seeds, dtype=np.int64)

    new_labels, n_super, _ = merge_clusters_by_spot_overlap(
        cluster_labels=cluster_labels,
        n_phase1_clusters=2,
        members_by_label=members_by_label,
        quats_per_seed=quats,
        ias_per_seed=ias,
        ibf_alive_col0=ibf_alive_col0,
        ibf_global_to_local=ibf_global_to_local,
        space_group=225,
        misori_tol_rad=math.radians(1.0),
        jaccard_tol=0.5,
        progress=False,
    )
    assert n_super == 2
    assert new_labels.tolist() == [0, 0, 1, 1]


def test_pass_a_does_not_merge_when_misori_too_large():
    # High spot overlap but orientations 30° apart → misori filter rejects.
    n_seeds = 4
    cluster_labels = np.array([0, 0, 1, 1], dtype=np.int64)
    members_by_label = {
        0: np.array([0, 1], dtype=np.int64),
        1: np.array([2, 3], dtype=np.int64),
    }
    quats = np.zeros((n_seeds, 4), dtype=np.float64)
    # Cluster 0 rep at identity.
    quats[:, 0] = 1.0
    # Cluster 1 rep at 30° rotation about x-axis: q = (cos15°, sin15°, 0, 0).
    half = math.radians(15.0)
    quats[2] = [math.cos(half), math.sin(half), 0.0, 0.0]
    quats[3] = quats[2]
    ias = np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
    ibf_alive_col0 = np.array([
        [101, 102, 103, 104, 105],
        [101, 102, 103, 104, 105],
        [101, 102, 103, 104, 105],
        [101, 102, 103, 104, 105],
    ], dtype=np.int64)
    ibf_global_to_local = np.arange(n_seeds, dtype=np.int64)

    new_labels, n_super, _ = merge_clusters_by_spot_overlap(
        cluster_labels=cluster_labels,
        n_phase1_clusters=2,
        members_by_label=members_by_label,
        quats_per_seed=quats,
        ias_per_seed=ias,
        ibf_alive_col0=ibf_alive_col0,
        ibf_global_to_local=ibf_global_to_local,
        space_group=225,
        misori_tol_rad=math.radians(1.0),
        jaccard_tol=0.5,
        progress=False,
    )
    # 30° misori >> 1° tol → no merge despite full Jaccard overlap.
    assert n_super == 2


def test_pass_a_zero_seeds_passthrough():
    # No clusters: must be a no-op.
    cluster_labels = np.array([], dtype=np.int64)
    new_labels, n_super, new_members = merge_clusters_by_spot_overlap(
        cluster_labels=cluster_labels,
        n_phase1_clusters=0,
        members_by_label={},
        quats_per_seed=np.empty((0, 4), dtype=np.float64),
        ias_per_seed=np.empty(0, dtype=np.float64),
        ibf_alive_col0=np.empty((0, 0), dtype=np.int64),
        ibf_global_to_local=np.empty(0, dtype=np.int64),
        space_group=225,
        misori_tol_rad=math.radians(1.0),
        jaccard_tol=0.5,
        progress=False,
    )
    assert n_super == 0
    assert new_members == {}
