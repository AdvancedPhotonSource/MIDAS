"""Cross-voxel global clustering."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    global_cluster,
    write_unique_orientations_csv,
)


def test_4vox_2grains_dedup_to_2_uniques(axis_angle_om, tmp_path):
    """4 voxels, 2 underlying grains → global_cluster yields 2 uniques."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    per_vox_OMs = np.vstack([g1, g1, g2, g2])
    per_vox_confs = np.array([0.9, 0.7, 0.8, 0.6])
    # Layout: [SpotID, nMatches, nIDs_best_solution, bestSolIdx] per voxel.
    per_vox_keys = np.array([
        [10, 5, 4, 0],
        [11, 5, 4, 1],
        [20, 6, 5, 0],
        [21, 6, 5, 1],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    assert glob.n_uniques == 2
    # The representative voxel of each cluster should be the highest-conf one.
    # Voxel 0 wins for g1 (conf 0.9), voxel 2 wins for g2 (conf 0.8).
    rep_voxels = sorted(int(r[0]) for r in glob.unique_key_arr)
    assert rep_voxels == [0, 2]


def test_global_cluster_skips_invalid_voxels(axis_angle_om):
    """Voxels with sentinel ``per_vox_keys[v, 0] == (uint64)-1`` are skipped."""
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    per_vox_OMs = np.vstack([g1, g1, g1])
    per_vox_confs = np.array([0.5, 0.5, 0.5])
    per_vox_keys = np.array([
        [2**64 - 1, 0, 0, 0],   # invalid
        [11, 5, 4, 0],
        [12, 5, 4, 0],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    assert glob.n_uniques == 1
    # The rep voxel is 1 (first non-invalid; ties broken by lower index).
    assert int(glob.unique_key_arr[0, 0]) == 1


def test_unique_orientations_csv_14col_format(axis_angle_om, tmp_path):
    g1 = axis_angle_om(np.array([0, 0, 1]), 0.0)
    g2 = axis_angle_om(np.array([0, 0, 1]), 30.0)
    per_vox_OMs = np.vstack([g1, g2])
    per_vox_confs = np.array([0.9, 0.7])
    per_vox_keys = np.array([
        [10, 5, 4, 0],
        [20, 6, 5, 1],
    ], dtype=np.uint64)
    glob = global_cluster(
        per_vox_OMs, per_vox_confs, per_vox_keys,
        space_group=225, max_ang_deg=1.0,
    )
    csv_path = tmp_path / "UniqueOrientations.csv"
    write_unique_orientations_csv(csv_path, glob.unique_key_arr, glob.unique_OM_arr)
    text = csv_path.read_text().strip().splitlines()
    assert text[0].startswith("#")
    # Header tokens after '#' should mention 5 key cols + 9 OM cols = 14.
    header_tokens = [t for t in text[0].lstrip("#").split() if t]
    assert len(header_tokens) == 14, header_tokens
    # Body: 2 rows, each row has 14 columns.
    for body_line in text[1:]:
        toks = body_line.split()
        assert len(toks) == 14
