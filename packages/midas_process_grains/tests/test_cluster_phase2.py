"""Phase 2 spot-aware sub-clustering tests."""

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
from midas_process_grains.compute.refine_cluster import (
    GrainCandidate,
    refine_cluster_spot_aware,
)
from midas_process_grains.compute.symmetry import build_symmetry_table


def _q_from_euler_rad(e):
    return np.asarray(orient_mat_to_quat(np.asarray(euler_to_orient_mat(e)).reshape(-1)))


@pytest.fixture
def fcc_hkl_complete_orbit(tmp_path):
    """Full {111} orbit so every cubic op maps inside the table."""
    p = tmp_path / "hkls.csv"
    text = "h k l D-spacing RingNr g1 g2 g3 Theta(deg) 2Theta(deg) Radius\n"
    for h in (-1, 1):
        for k in (-1, 1):
            for l in (-1, 1):
                text += f"{h} {k} {l} 2.0754 1 0 0 0 2.39 4.78 60000.0\n"
    p.write_text(text)
    from midas_process_grains.io.hkls import load_hkl_table
    return load_hkl_table(p)


def test_two_identical_seeds_with_identical_spots_form_one_grain(fcc_hkl_complete_orbit):
    """Two seeds, same orientation, same SpotIDs at every hkl → one grain."""
    sym_table = build_symmetry_table(225, fcc_hkl_complete_orbit, warn_missing=False)
    n_hkls = sym_table.n_hkls
    rep_q = _q_from_euler_rad([0.0, 0.0, 0.0])

    spot_table = np.zeros(n_hkls, dtype=np.int64)
    spot_table[:4] = [101, 102, 103, 104]

    member_positions = [0, 1]
    member_quats = np.stack([rep_q, rep_q.copy()], axis=0)
    member_col0 = np.stack([spot_table, spot_table.copy()], axis=0)
    ring_radii = np.full(n_hkls, 60000.0)

    grains = refine_cluster_spot_aware(
        member_positions, 0, member_quats, rep_q, member_col0, sym_table,
        ring_radii_per_hkl=ring_radii, pixel_size_um=200.0,
        edge_weight_threshold=0.5,
    )
    assert len(grains) == 1
    g = grains[0]
    assert g.rep_pos == 0
    assert sorted(g.member_positions) == [0, 1]


def test_two_seeds_disagreeing_at_high_resolution_split(fcc_hkl_complete_orbit):
    """Two seeds, very small but nonzero misorientation, disagree on SpotIDs at
    every informative hkl → split into two grains."""
    sym_table = build_symmetry_table(225, fcc_hkl_complete_orbit, warn_missing=False)
    n_hkls = sym_table.n_hkls
    q_a = _q_from_euler_rad([0.0, 0.0, 0.0])
    q_b = _q_from_euler_rad([math.radians(0.20), 0.0, 0.0])  # 0.2° apart

    # Different SpotIDs at every row (well above noise).
    table_a = np.arange(1, n_hkls + 1, dtype=np.int64) * 1000
    table_b = np.arange(1, n_hkls + 1, dtype=np.int64) * 2000

    member_positions = [0, 1]
    member_quats = np.stack([q_a, q_b], axis=0)
    member_col0 = np.stack([table_a, table_b], axis=0)
    ring_radii = np.full(n_hkls, 60000.0)

    grains = refine_cluster_spot_aware(
        member_positions, 0, member_quats, q_a, member_col0, sym_table,
        ring_radii_per_hkl=ring_radii, pixel_size_um=200.0,
        pixel_tol=0.5,                         # informative everywhere at 60k µm × 0.0035 rad ≈ 200 µm
        edge_weight_threshold=0.7,
    )
    assert len(grains) == 2
    rep_positions = sorted(g.rep_pos for g in grains)
    assert rep_positions == [0, 1]


def test_symmetry_equivalent_seeds_with_aligned_spots_form_one_grain(
    fcc_hkl_complete_orbit,
):
    """q_b = q_a · S, both seeds explain the same spots after symmetry alignment."""
    sym_table = build_symmetry_table(225, fcc_hkl_complete_orbit, warn_missing=False)
    n_hkls = sym_table.n_hkls
    n_sym, sym_quats_np = make_symmetries(225)

    q_a = _q_from_euler_rad([0.5, 0.3, 0.7])

    # Apply op s=5 (a non-identity cubic op) to get q_b.
    import torch
    sym_t = torch.from_numpy(np.asarray(sym_quats_np))
    s_chosen = 5
    q_a_t = torch.from_numpy(q_a)
    q_b_t = _quat_mul(q_a_t, sym_t[s_chosen])
    q_b_t = q_b_t / q_b_t.norm()
    q_b = q_b_t.numpy()

    # In the rep (q_a) frame, both seeds should claim the SAME SpotID at each
    # row index. We construct seed-A's table directly. Seed-B's table is
    # related by the inverse permutation: aligned_B[k] = raw_B[π[k]] should
    # equal A. So raw_B[π[k]] = A[k], i.e. raw_B[i] = A[π^{-1}[i]].
    perm = sym_table.hkl_perm[s_chosen].cpu().numpy()
    table_a = np.zeros(n_hkls, dtype=np.int64)
    table_a[:n_hkls] = np.arange(1001, 1001 + n_hkls)

    # Build raw_B such that raw_B[perm[k]] == table_a[k] for every k where perm[k] >= 0.
    raw_b = np.zeros(n_hkls, dtype=np.int64)
    for k in range(n_hkls):
        if perm[k] >= 0:
            raw_b[perm[k]] = table_a[k]

    member_positions = [0, 1]
    member_quats = np.stack([q_a, q_b], axis=0)
    member_col0 = np.stack([table_a, raw_b], axis=0)
    ring_radii = np.full(n_hkls, 60000.0)

    grains = refine_cluster_spot_aware(
        member_positions, 0, member_quats, q_a, member_col0, sym_table,
        ring_radii_per_hkl=ring_radii, pixel_size_um=200.0,
        edge_weight_threshold=0.5,
    )
    assert len(grains) == 1, (
        "two equivalent variants explaining the same observed spots must "
        "merge into a single grain after symmetry-aware row alignment"
    )


def test_singleton_below_min_nr_spots_is_dropped(fcc_hkl_complete_orbit):
    """A 1-member sub-cluster with fewer matched SpotIDs than min_nr_spots
    should be filtered out."""
    sym_table = build_symmetry_table(225, fcc_hkl_complete_orbit, warn_missing=False)
    n_hkls = sym_table.n_hkls

    q = _q_from_euler_rad([0.0, 0.0, 0.0])
    table = np.zeros(n_hkls, dtype=np.int64)
    table[0] = 42                              # only one matched SpotID

    grains = refine_cluster_spot_aware(
        [0], 0, q[None, :], q, table[None, :], sym_table,
        ring_radii_per_hkl=np.full(n_hkls, 60000.0),
        pixel_size_um=200.0,
        edge_weight_threshold=0.5,
        min_nr_spots=2,
    )
    assert grains == []


def test_singleton_above_min_nr_spots_passes_through(fcc_hkl_complete_orbit):
    """A 1-member cluster with enough matched spots survives — the new
    pipeline rescues these (the C MinNrSpots=3 used to drop them)."""
    sym_table = build_symmetry_table(225, fcc_hkl_complete_orbit, warn_missing=False)
    n_hkls = sym_table.n_hkls

    q = _q_from_euler_rad([0.0, 0.0, 0.0])
    table = np.zeros(n_hkls, dtype=np.int64)
    table[:5] = [101, 102, 103, 104, 105]

    grains = refine_cluster_spot_aware(
        [7], 7, q[None, :], q, table[None, :], sym_table,
        ring_radii_per_hkl=np.full(n_hkls, 60000.0),
        pixel_size_um=200.0,
        edge_weight_threshold=0.5,
        min_nr_spots=2,
    )
    assert len(grains) == 1
    assert grains[0].rep_pos == 7
    assert grains[0].member_positions == [7]
