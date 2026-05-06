"""Phase 1 cluster tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from midas_stress.orientation import (
    euler_to_orient_mat,
    orient_mat_to_quat,
    make_symmetries,
)

from midas_process_grains.compute.canonicalize import _quat_mul
from midas_process_grains.compute.cluster import (
    ClusterResult,
    cluster_by_misorientation,
    pairwise_misorientation,
)


def _q_from_euler_rad(e):
    return np.asarray(orient_mat_to_quat(np.asarray(euler_to_orient_mat(e)).reshape(-1)))


def test_two_identical_seeds_form_one_cluster():
    q = _q_from_euler_rad([0.1, 0.2, 0.3])
    quats = np.stack([q, q.copy(), q.copy()], axis=0)
    res = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25),
    )
    assert isinstance(res, ClusterResult)
    assert res.n_clusters == 1
    assert (res.labels == 0).all()


def test_two_far_apart_seeds_form_two_clusters():
    q1 = _q_from_euler_rad([0.0, 0.0, 0.0])
    q2 = _q_from_euler_rad([math.radians(45), 0.0, 0.0])  # 45° away
    quats = np.stack([q1, q2], axis=0)
    res = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25),
    )
    assert res.n_clusters == 2
    assert res.labels[0] != res.labels[1]


def test_symmetry_equivalent_variants_cluster_together():
    """q_b = q_a · S for some non-trivial cubic op should form ONE cluster."""
    n_sym, syms = make_symmetries(225)
    sym_quats = np.asarray(syms, dtype=np.float64)

    q_a = _q_from_euler_rad([0.5, 0.3, 0.7])
    import torch
    q_a_t = torch.from_numpy(q_a)
    q_b_t = _quat_mul(q_a_t, torch.from_numpy(sym_quats[5]))   # apply op 5
    q_b_t = q_b_t / q_b_t.norm()
    q_b = q_b_t.numpy()

    quats = np.stack([q_a, q_b], axis=0)
    res = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25),
    )
    assert res.n_clusters == 1, (
        "two equivalent variants must cluster despite having raw quaternions "
        "that differ by ~degrees-to-tens-of-degrees"
    )


def test_dead_seeds_get_label_minus_one():
    q = _q_from_euler_rad([0.1, 0.2, 0.3])
    quats = np.stack([q.copy(), q.copy()], axis=0)
    alive = np.array([True, False])
    res = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25),
        alive_mask=alive,
    )
    assert res.labels[0] == 0
    assert res.labels[1] == -1
    assert res.n_clusters == 1


def test_three_seeds_two_close_one_far():
    q1 = _q_from_euler_rad([0.0, 0.0, 0.0])
    q2 = _q_from_euler_rad([math.radians(0.1), 0.0, 0.0])
    q3 = _q_from_euler_rad([math.radians(20.0), 0.0, 0.0])
    quats = np.stack([q1, q2, q3], axis=0)
    res = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25),
    )
    assert res.n_clusters == 2
    assert res.labels[0] == res.labels[1]
    assert res.labels[0] != res.labels[2]


def test_pairwise_misorientation_returns_zero_for_identical():
    q = _q_from_euler_rad([0.1, 0.2, 0.3])
    quats = np.stack([q, q.copy()], axis=0)
    pairs = np.array([[0, 1]])
    misori = pairwise_misorientation(quats, 225, pair_indices=pairs)
    assert float(misori[0]) < 1e-9


def test_block_size_does_not_change_result():
    """Smoke: clustering with a tiny block size produces the same labels."""
    rng = np.random.default_rng(7)
    n = 30
    base = _q_from_euler_rad([0.5, 0.3, 0.7])
    # make 3 tight pockets of 10 seeds each, well-separated
    quats_list = []
    for centre in [
        [0.0, 0.0, 0.0],
        [math.radians(20.0), 0.0, 0.0],
        [math.radians(40.0), 0.0, 0.0],
    ]:
        for _ in range(10):
            jitter = rng.normal(scale=math.radians(0.05), size=3)
            quats_list.append(_q_from_euler_rad(np.array(centre) + jitter))
    quats = np.asarray(quats_list)

    a = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25), block_size=64,
    )
    b = cluster_by_misorientation(
        quats, space_group=225, misori_tol_rad=math.radians(0.25), block_size=4,
    )
    assert a.n_clusters == b.n_clusters == 3
    # Labels may be re-numbered between calls, but the partition must match.
    def partition(labels):
        out = {}
        for i, l in enumerate(labels):
            out.setdefault(int(l), []).append(i)
        return frozenset(frozenset(v) for v in out.values())
    assert partition(a.labels) == partition(b.labels)
