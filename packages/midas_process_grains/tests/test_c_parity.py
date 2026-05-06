"""Synthetic-data tests for C-parity Stage 1 + Pass A.

Smoke tests we can iterate on in milliseconds. Once these pass we run on
the real peakfit_hard data and compare to C ProcessGrains' output.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from midas_process_grains.compute.c_parity import (
    OPF_OM, OPF_POS, OPF_LATTICE, OPF_IA, OPF_RADIUS, OPF_CONFIDENCE,
    _build_pos_by_id,
    pass_a_position_dedup,
    stage1_find_internal_angles,
    build_kept_list,
)


# --------------------------------------------------------------------------
# Helpers to build synthetic OPF / Key / ProcessKey
# --------------------------------------------------------------------------

def _make_opf(orient_mats, positions, lattices=None, ia=None,
              radius=None, conf=None, ids=None):
    n = orient_mats.shape[0]
    opf = np.zeros((n, 27), dtype=np.float64)
    if ids is None:
        ids = np.arange(1, n + 1, dtype=np.int64)
    opf[:, 0] = ids
    opf[:, OPF_OM] = orient_mats.reshape(n, 9)
    opf[:, OPF_POS] = positions
    if lattices is None:
        lattices = np.tile([3.6, 3.6, 3.6, 90, 90, 90], (n, 1))
    opf[:, OPF_LATTICE] = lattices
    opf[:, OPF_IA] = ia if ia is not None else np.linspace(0.1, 0.5, n)
    opf[:, OPF_RADIUS] = radius if radius is not None else np.full(n, 1.0)
    opf[:, OPF_CONFIDENCE] = conf if conf is not None else np.full(n, 1.0)
    return opf, ids


def _process_key_from_overlap(n_seeds, overlap_lists, max_per_grain=20):
    """overlap_lists[i] = list of candidate IDs (1-indexed SpotIDs) for seed i."""
    pk = np.zeros((n_seeds, max_per_grain), dtype=np.int32)
    nr = np.zeros(n_seeds, dtype=np.int32)
    for i, ol in enumerate(overlap_lists):
        n = len(ol)
        if n > max_per_grain:
            raise ValueError(f"seed {i}: {n} candidates exceeds {max_per_grain}")
        pk[i, :n] = ol
        nr[i] = n
    return pk, nr


# --------------------------------------------------------------------------
# pos_by_id
# --------------------------------------------------------------------------

def test_pos_by_id_basic():
    ids = np.array([10, 20, 30, 40], dtype=np.int64)
    pbi, max_id = _build_pos_by_id(ids)
    assert max_id == 40
    assert pbi[10] == 0
    assert pbi[20] == 1
    assert pbi[30] == 2
    assert pbi[40] == 3
    # Gaps stay -1
    assert pbi[15] == -1
    assert pbi[5] == -1


def test_pos_by_id_first_wins_on_dupes():
    # Duplicate ID 10 at positions 0 and 2; pos 0 should win (matches C's
    # `if pos_by_id[id] == -1` first-write semantics).
    ids = np.array([10, 20, 10, 30], dtype=np.int64)
    pbi, _ = _build_pos_by_id(ids)
    assert pbi[10] == 0


# --------------------------------------------------------------------------
# Stage 1: synthetic clusters
# --------------------------------------------------------------------------

def _identity_om():
    return np.eye(3).astype(np.float64)


def _rot_om(axis, theta_rad):
    """Rotation matrix about ``axis`` by ``theta_rad``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta_rad) * K + (1 - math.cos(theta_rad)) * K @ K


def test_stage1_two_seeds_low_misori_merge():
    # Two seeds, one identity, one rotated 0.2° about z. Both share at
    # least one spot in ProcessKey. Should merge into one cluster.
    om0 = _identity_om()
    om1 = _rot_om([0, 0, 1], math.radians(0.2))
    om = np.stack([om0, om1], axis=0)
    pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos, ia=np.array([0.1, 0.2]))
    keep = np.ones(2, dtype=bool)
    pk, nr = _process_key_from_overlap(2, [
        [2],   # seed 0 has spot ID 2 as candidate
        [1],   # seed 1 has spot ID 1 as candidate
    ])
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    assert len(res.clusters) == 1
    assert sorted(res.clusters[0].member_positions.tolist()) == [0, 1]
    assert res.clusters[0].rep_pos == 0   # min IA was 0.1 at pos 0


def test_stage1_two_seeds_high_misori_split():
    # Two seeds, 1.0° apart — should NOT merge at 0.4° tol.
    om0 = _identity_om()
    om1 = _rot_om([0, 0, 1], math.radians(1.0))
    om = np.stack([om0, om1], axis=0)
    pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos)
    keep = np.ones(2, dtype=bool)
    pk, nr = _process_key_from_overlap(2, [[2], [1]])
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    assert len(res.clusters) == 2


def test_stage1_no_spot_overlap_no_merge():
    # Two seeds with identical orientation but NO shared candidate spots
    # → no merge, even though misori = 0.
    om = np.stack([_identity_om(), _identity_om()], axis=0)
    pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos)
    keep = np.ones(2, dtype=bool)
    pk, nr = _process_key_from_overlap(2, [[], []])  # no candidates
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    assert len(res.clusters) == 2


def test_stage1_chain_through_transitivity():
    # Three seeds A, B, C. A→B has misori 0.3° (passes), B→C has misori
    # 0.3° (passes), but A→C has misori 0.6° (would fail directly). With
    # transitive closure via B, all three merge into one cluster.
    om = np.stack([
        _identity_om(),
        _rot_om([0, 0, 1], math.radians(0.3)),
        _rot_om([0, 0, 1], math.radians(0.6)),
    ], axis=0)
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos, ia=np.array([0.1, 0.2, 0.3]))
    keep = np.ones(3, dtype=bool)
    # ProcessKey: A's candidates are [B], B's candidates are [A, C], C's are [B]
    pk, nr = _process_key_from_overlap(3, [
        [2],         # A → B
        [1, 3],      # B → A, C
        [2],         # C → B
    ])
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    assert len(res.clusters) == 1
    assert sorted(res.clusters[0].member_positions.tolist()) == [0, 1, 2]
    assert res.clusters[0].rep_pos == 0   # min IA


def test_stage1_dfs_visit_order_matches_c():
    # Five seeds: 0 → [1, 2, 3, 4]. All within 0.1° of each other.
    # C's recursive DFS visits in pre-order: 0, then recurse into 1
    # (1's candidates), then into 2, etc.
    # We only check that 0 is the first member.
    om = [_identity_om()]
    for k in range(4):
        om.append(_rot_om([0, 0, 1], math.radians(0.05 * (k + 1))))
    om = np.stack(om, axis=0)
    pos = np.zeros((5, 3))
    opf, ids = _make_opf(om, pos, ia=np.linspace(0.1, 0.5, 5))
    keep = np.ones(5, dtype=bool)
    pk, nr = _process_key_from_overlap(5, [
        [2, 3, 4, 5],     # seed 0 → seeds 1, 2, 3, 4
        [1], [1], [1], [1],
    ])
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    assert len(res.clusters) == 1
    assert sorted(res.clusters[0].member_positions.tolist()) == [0, 1, 2, 3, 4]
    # First visited member should be 0 (the cluster root we started from).
    assert res.clusters[0].member_positions[0] == 0


def test_stage1_dead_seeds_skipped():
    # Three seeds, but seed 1 is dead (keep_flag False). Should not appear
    # in any cluster.
    om = np.stack([_identity_om(), _identity_om(), _identity_om()], axis=0)
    pos = np.zeros((3, 3))
    opf, ids = _make_opf(om, pos)
    keep = np.array([True, False, True])
    pk, nr = _process_key_from_overlap(3, [[2, 3], [1, 3], [1, 2]])
    res = stage1_find_internal_angles(
        opf=opf, ids=ids, keep_flag=keep, nr_ids_per_id=nr,
        process_key=pk, space_group=225,
        misori_tol_rad=math.radians(0.4),
    )
    # Only seeds 0 and 2 are alive and connected → one cluster.
    assert len(res.clusters) == 1
    assert sorted(res.clusters[0].member_positions.tolist()) == [0, 2]
    # Seed 1 is non-alive, label = -1.
    assert res.cluster_label_per_pos[1] == -1


# --------------------------------------------------------------------------
# Pass A: position + misori dedup
# --------------------------------------------------------------------------

def test_pass_a_close_pair_marks_dup():
    # Two grain reps within 5 µm AND 0.1° apart → mark j as dup.
    om = np.stack([
        _identity_om(),
        _rot_om([0, 0, 1], math.radians(0.05)),
    ], axis=0)
    pos = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos)
    grain_positions = np.array([0, 1], dtype=np.int64)
    is_dup = pass_a_position_dedup(
        grain_positions=grain_positions,
        opf=opf, space_group=225,
        misori_tol_rad=math.radians(0.1),
        pos_tol_um=5.0,
    )
    assert is_dup.tolist() == [False, True]


def test_pass_a_far_pair_kept():
    om = np.stack([_identity_om(), _identity_om()], axis=0)
    pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)  # 10 µm apart
    opf, ids = _make_opf(om, pos)
    grain_positions = np.array([0, 1], dtype=np.int64)
    is_dup = pass_a_position_dedup(
        grain_positions=grain_positions,
        opf=opf, space_group=225,
        misori_tol_rad=math.radians(0.1),
        pos_tol_um=5.0,
    )
    assert is_dup.tolist() == [False, False]


def test_pass_a_high_misori_kept():
    # Same position, but 0.5° apart → misori filter rejects merge.
    om = np.stack([
        _identity_om(),
        _rot_om([0, 0, 1], math.radians(0.5)),
    ], axis=0)
    pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos)
    grain_positions = np.array([0, 1], dtype=np.int64)
    is_dup = pass_a_position_dedup(
        grain_positions=grain_positions,
        opf=opf, space_group=225,
        misori_tol_rad=math.radians(0.1),
        pos_tol_um=5.0,
    )
    assert is_dup.tolist() == [False, False]


def test_pass_a_chain_three_keeps_first_marks_others():
    # Three reps at (0,0,0), (3,0,0), (6,0,0) — pairs: (0,1) within 5 µm
    # (close), (0,2) NOT within 5 (6 µm > 5), (1,2) within 5.
    # Expected: process pair (0,1) → mark 1 as dup; then pair (1,2) but 1
    # is already dup, so skip; pair (0,2) fails position. So is_dup = [F,T,F].
    om = np.stack([_identity_om()] * 3, axis=0)
    pos = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]], dtype=np.float64)
    opf, ids = _make_opf(om, pos)
    grain_positions = np.arange(3, dtype=np.int64)
    is_dup = pass_a_position_dedup(
        grain_positions=grain_positions,
        opf=opf, space_group=225,
        misori_tol_rad=math.radians(0.1),
        pos_tol_um=5.0,
    )
    assert is_dup.tolist() == [False, True, False]


# --------------------------------------------------------------------------
# build_kept_list: confidence filter + dedup
# --------------------------------------------------------------------------

def test_build_kept_list_drops_low_confidence():
    om = np.stack([_identity_om()] * 3, axis=0)
    pos = np.zeros((3, 3))
    conf = np.array([0.04, 0.10, 1.00])   # only first below 0.05 floor
    opf, ids = _make_opf(om, pos, conf=conf)
    grain_positions = np.arange(3, dtype=np.int64)
    is_dup = np.array([False, False, False])
    kept = build_kept_list(grain_positions=grain_positions,
                            is_dup=is_dup, opf=opf, confidence_min=0.05)
    assert kept.tolist() == [1, 2]


def test_build_kept_list_drops_dup_and_low_conf():
    om = np.stack([_identity_om()] * 3, axis=0)
    pos = np.zeros((3, 3))
    conf = np.array([0.01, 1.0, 1.0])
    opf, ids = _make_opf(om, pos, conf=conf)
    grain_positions = np.arange(3, dtype=np.int64)
    is_dup = np.array([False, True, False])
    kept = build_kept_list(grain_positions=grain_positions,
                            is_dup=is_dup, opf=opf, confidence_min=0.05)
    # 0 dropped for low conf, 1 dropped for dup, 2 kept.
    assert kept.tolist() == [2]
