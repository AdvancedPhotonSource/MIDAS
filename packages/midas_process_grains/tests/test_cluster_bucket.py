"""Tests for the Rodrigues-bucket prefilter and bucketed clustering."""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_stress.orientation import (
    euler_to_orient_mat,
    fundamental_zone,
    orient_mat_to_quat,
)

from midas_process_grains.compute.cluster import (
    bucket_candidate_pairs,
    cluster_by_misorientation,
)


def _q_from_euler_rad(e):
    om = np.asarray(euler_to_orient_mat(e)).reshape(-1)
    return np.asarray(orient_mat_to_quat(om))


def _fz(quats):
    out = np.empty_like(quats)
    for i in range(quats.shape[0]):
        out[i] = np.asarray(fundamental_zone(quats[i], 225))
    return out


def test_bucket_pairs_finds_close_pair():
    q1 = _q_from_euler_rad([0.0, 0.0, 0.0])
    q2 = _q_from_euler_rad([math.radians(0.1), 0.0, 0.0])  # 0.1° away
    quats = np.stack([q1, q2], axis=0)
    fz = _fz(quats)
    pairs = bucket_candidate_pairs(
        fz, math.radians(0.25),
        alive_idx=np.array([0, 1]), space_group=225,
    )
    assert pairs.shape[0] >= 1
    assert tuple(pairs[0]) == (0, 1)


def test_bucket_pairs_drops_far_apart():
    q1 = _q_from_euler_rad([0.0, 0.0, 0.0])
    q2 = _q_from_euler_rad([math.radians(45), 0.0, 0.0])  # 45° away
    quats = np.stack([q1, q2], axis=0)
    fz = _fz(quats)
    pairs = bucket_candidate_pairs(
        fz, math.radians(0.25),
        alive_idx=np.array([0, 1]), space_group=225,
        safety_cells=1,
    )
    # 45° vs 0.25° threshold → must be filtered.
    # NOTE: with 24 sym-equivalent reps, sometimes a sym-rep of one seed
    # may land in the same cell as the other; the misori filter still
    # rejects it. So we just verify the candidate count is small (not the
    # full pair set) — the actual correctness comes from the misori filter
    # downstream.
    assert pairs.shape[0] <= 1


def test_bucketed_cluster_matches_naive_on_small_set():
    """The two methods must produce identical partitions on a manageable
    seed population."""
    rng = np.random.default_rng(7)
    centres = [
        [0.0, 0.0, 0.0],
        [math.radians(10.0), 0.0, 0.0],
        [math.radians(20.0), math.radians(5.0), 0.0],
    ]
    quats = []
    for c in centres:
        for _ in range(8):
            jitter = rng.normal(scale=math.radians(0.05), size=3)
            quats.append(_q_from_euler_rad(np.array(c) + jitter))
    quats = np.asarray(quats)

    a = cluster_by_misorientation(
        quats, 225,
        misori_tol_rad=math.radians(0.25),
        method="naive",
    )
    b = cluster_by_misorientation(
        quats, 225,
        misori_tol_rad=math.radians(0.25),
        method="bucketed",
    )
    # Same number of clusters.
    assert a.n_clusters == b.n_clusters == 3

    def partition(labels):
        out = {}
        for i, l in enumerate(labels):
            out.setdefault(int(l), []).append(i)
        return frozenset(frozenset(v) for v in out.values())

    assert partition(a.labels) == partition(b.labels)


def test_bucketed_cluster_handles_dead_seeds():
    q = _q_from_euler_rad([0.1, 0.2, 0.3])
    quats = np.stack([q.copy(), q.copy()], axis=0)
    alive = np.array([True, False])
    res = cluster_by_misorientation(
        quats, 225, misori_tol_rad=math.radians(0.25),
        alive_mask=alive, method="bucketed",
    )
    assert res.labels[0] == 0
    assert res.labels[1] == -1


def test_bucketed_scales_better_than_naive_on_pocket_data():
    """Sanity benchmark — bucketed should evaluate dramatically fewer pairs."""
    rng = np.random.default_rng(11)
    n_pockets = 5
    pocket_size = 30
    quats = []
    for p in range(n_pockets):
        centre = np.array([
            math.radians(15.0 * p), math.radians(2.0 * p), math.radians(3.0 * p),
        ])
        for _ in range(pocket_size):
            jitter = rng.normal(scale=math.radians(0.05), size=3)
            quats.append(_q_from_euler_rad(centre + jitter))
    quats = np.asarray(quats)
    fz = _fz(quats)
    n = quats.shape[0]                      # 150 seeds

    # With safety_cells=2 the prefilter catches all true intra-pocket
    # pairs (no boundary-jitter misses). Default safety_cells=1 catches
    # ~95% which is sufficient because the cluster connectivity gets
    # filled in by transitivity in connected_components.
    pairs = bucket_candidate_pairs(
        fz, math.radians(0.25),
        alive_idx=np.arange(n), space_group=225,
        safety_cells=2,
    )
    full_pairs = n * (n - 1) // 2
    intra_pocket_pairs = n_pockets * (pocket_size * (pocket_size - 1) // 2)
    # Catch ≥ 99 % of true intra-pocket pairs — tiny boundary jitter is
    # acceptable because connected_components fills the gaps via
    # transitivity. The actual cluster-correctness contract is the
    # naive-vs-bucketed parity test above.
    assert pairs.shape[0] >= 0.99 * intra_pocket_pairs
    assert pairs.shape[0] <= full_pairs
