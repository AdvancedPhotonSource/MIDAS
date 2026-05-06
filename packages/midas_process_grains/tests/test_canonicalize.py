"""Canonicalisation tests: pick the right symmetry op + apply the permutation."""

from __future__ import annotations

import numpy as np
import torch

from midas_stress.orientation import (
    euler_to_orient_mat,
    orient_mat_to_quat,
    make_symmetries,
)

from midas_process_grains.compute.canonicalize import (
    align_member_to_rep,
    pick_best_sym_op,
)
from midas_process_grains.compute.symmetry import build_symmetry_table


def _quat_from_euler(euler_rad):
    # midas_stress.orient_mat_to_quat expects a flat 9-vector (matches the C ABI).
    om_flat = np.asarray(euler_to_orient_mat(euler_rad)).reshape(-1)
    return np.asarray(orient_mat_to_quat(om_flat))


def test_identity_picks_op_zero():
    """Two identical orientations → s_idx=0 (identity), residual ≈ 0."""
    n_sym, syms = make_symmetries(225)
    sym_quats = torch.from_numpy(np.asarray(syms, dtype=np.float64))
    q = torch.from_numpy(_quat_from_euler([0.1, 0.2, 0.3]))
    s_idx, angle = pick_best_sym_op(q, q.clone(), sym_quats)
    assert s_idx == 0
    assert float(angle) < 1e-10


def test_equivalent_variant_aligns_to_zero_residual():
    """q_member = q_rep · S_k for some non-trivial k → pick_best_sym_op
    returns the same k (or its inverse) and residual angle ≈ 0."""
    n_sym, syms = make_symmetries(225)
    sym_quats_np = np.asarray(syms, dtype=np.float64)
    sym_quats = torch.from_numpy(sym_quats_np)

    q_rep = torch.from_numpy(_quat_from_euler([0.5, 0.3, 0.7]))

    # Construct q_member = q_rep · S_k for a few k and verify residual ≈ 0.
    # We multiply via the helpers in canonicalize.
    from midas_process_grains.compute.canonicalize import _quat_mul

    for k in (1, 5, 10, 17):
        q_member = _quat_mul(q_rep, sym_quats[k])
        # Normalize (Hamilton products of unit quats stay unit, but defensively)
        q_member = q_member / q_member.norm()
        s_idx, angle = pick_best_sym_op(q_rep, q_member, sym_quats)
        assert float(angle) < 1e-9, (
            f"could not align via op {k}: best s_idx={s_idx}, residual={angle}°"
        )


def test_align_member_to_rep_permutes_rows(cubic_hkl_table):
    """If π_s sends row 2 → 4, then aligned_table[4] == member_table[2]."""
    tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    n_hkls = tbl.n_hkls

    # Synthetic per-hkl table: row k holds [k, k * 10].
    member = torch.tensor(
        [[k, k * 10] for k in range(n_hkls)], dtype=torch.float64,
    )

    # Pick op 0 (identity) -- aligned should be identical to member.
    aligned0 = align_member_to_rep(member, 0, tbl.hkl_perm)
    np.testing.assert_array_equal(aligned0.numpy(), member.numpy())

    # Pick op 1 -- whatever its π is, the test is that the inverse mapping
    # holds: aligned[k] == member[π[k]] (or zero if π[k] == -1).
    s = 1
    perm = tbl.hkl_perm[s]
    aligned = align_member_to_rep(member, s, tbl.hkl_perm)
    for k in range(n_hkls):
        if perm[k] >= 0:
            np.testing.assert_array_equal(
                aligned[k].numpy(), member[int(perm[k])].numpy()
            )
        else:
            np.testing.assert_array_equal(aligned[k].numpy(), [0.0, 0.0])


def test_align_member_to_rep_preserves_shape(cubic_hkl_table):
    tbl = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    n_hkls = tbl.n_hkls

    # Mimic the FitBest row layout (n_hkls, 22)
    member = torch.arange(n_hkls * 22, dtype=torch.float64).reshape(n_hkls, 22)
    aligned = align_member_to_rep(member, 1, tbl.hkl_perm)
    assert aligned.shape == member.shape
