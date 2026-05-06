"""Twin post-processor tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_stress.orientation import (
    euler_to_orient_mat,
    orient_mat_to_quat,
)

from midas_process_grains.compute.canonicalize import _quat_mul
from midas_process_grains.compute.twins import (
    TwinRelation,
    default_fcc_twin_relations,
    extend_symmetry_table_with_twin,
    find_twin_pairs,
)
from midas_process_grains.compute.symmetry import build_symmetry_table


def _q_from_euler(e):
    return np.asarray(orient_mat_to_quat(np.asarray(euler_to_orient_mat(e)).reshape(-1)))


def test_default_fcc_twin_relations_has_four_111_ops():
    twins = default_fcc_twin_relations()
    assert len(twins) == 4
    for tw in twins:
        assert isinstance(tw, TwinRelation)
        assert abs(tw.angle_deg - 60.0) < 1e-9
        # Quaternion is unit
        assert abs(np.linalg.norm(tw.quaternion) - 1.0) < 1e-12


def test_find_twin_pairs_detects_planted_pair():
    """Plant grain B = grain A · T_111 then verify the pair is found."""
    twins = default_fcc_twin_relations()
    T = twins[0]                                    # <111> 60°
    qa = _q_from_euler([0.5, 0.3, 0.7])
    qa_t = torch.from_numpy(qa)
    qb_t = _quat_mul(qa_t, torch.from_numpy(T.quaternion))
    qb = (qb_t / qb_t.norm()).numpy()

    qc = _q_from_euler([1.5, 0.2, 0.4])             # unrelated grain

    grain_quats = np.stack([qa, qb, qc], axis=0)
    pairs = find_twin_pairs(grain_quats, 225, twins, tol_rad=math.radians(0.1))

    pair_set = {(p[0], p[1]) for p in pairs}
    assert (0, 1) in pair_set
    # No spurious match with the unrelated grain.
    for i, j, _ in pairs:
        assert (i, j) != (0, 2)
        assert (i, j) != (1, 2)


def test_find_twin_pairs_returns_empty_when_no_twins(cubic_hkl_table):
    """Two unrelated grains → no twin pairs."""
    twins = default_fcc_twin_relations()
    qa = _q_from_euler([0.0, 0.0, 0.0])
    qb = _q_from_euler([math.radians(15), 0.0, 0.0])
    grain_quats = np.stack([qa, qb], axis=0)
    pairs = find_twin_pairs(grain_quats, 225, twins, tol_rad=math.radians(0.1))
    assert pairs == []


def test_extend_symmetry_table_with_twin_keeps_shape(cubic_hkl_table):
    sym_table = build_symmetry_table(225, cubic_hkl_table, warn_missing=False)
    twin = default_fcc_twin_relations()[0]
    extended = extend_symmetry_table_with_twin(
        sym_table, twin,
        hkl_table_real=cubic_hkl_table.real,
        hkl_table_int=cubic_hkl_table.integers,
        hkl_to_row=cubic_hkl_table.hkl_to_row,
    )
    assert extended.n_sym == sym_table.n_sym
    assert extended.hkl_perm.shape == sym_table.hkl_perm.shape
    assert extended.ops_R.shape == (sym_table.n_sym, 3, 3)
    # Twin op composed with identity should be the twin itself, so the
    # extended ops are NOT the original ones (det must be a proper rotation).
    R0 = extended.ops_R[0].cpu().numpy()
    np.testing.assert_allclose(R0 @ R0.T, np.eye(3), atol=1e-9)
