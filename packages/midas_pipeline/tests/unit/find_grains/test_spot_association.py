"""Tolerance-mode spot association (process_spots equivalent)."""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import (
    open_all_three,
    process_spots,
    write_ids_bin,
    write_keys_bin,
    write_vals_bin,
)


def _build_fixture(tmp_path, *, n_voxels=2, n_spots=5):
    """Synthesize a minimal consolidated-file fixture for process_spots.

    Layout:
      - 2 voxels. Voxel 0 has 1 candidate; voxel 1 has 1 candidate.
      - Each candidate matches 3 spots (IDs 1,2,3 for voxel 0; 3,4,5 for voxel 1).
      - Note ID 3 is shared — that triggers the within-grain dedup logic.
    """
    out_dir = tmp_path / "Output"
    out_dir.mkdir()

    keys_per_vox = [
        np.array([[100, 3, 3, 0]], dtype=np.uint64),
        np.array([[200, 3, 3, 0]], dtype=np.uint64),
    ]
    vals_per_vox = [
        np.zeros((1, 16), dtype=np.float64),
        np.zeros((1, 16), dtype=np.float64),
    ]
    ids_per_vox = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([3, 4, 5], dtype=np.int32),
    ]
    write_vals_bin(out_dir / "IndexBest_all.bin", vals_per_vox)
    write_keys_bin(out_dir / "IndexKey_all.bin", keys_per_vox)
    write_ids_bin(out_dir / "IndexBest_IDs_all.bin", ids_per_vox)

    # Spots.bin: 5 rows, 10 cols.
    spots = np.zeros((n_spots, 10), dtype=np.float64)
    for i in range(n_spots):
        sid = i + 1
        spots[i, 0] = 100.0 + i
        spots[i, 1] = 200.0 + i
        spots[i, 2] = 30.0 + i * 0.001  # unique omega per spot
        spots[i, 3] = 1000.0
        spots[i, 4] = sid
        spots[i, 5] = 1
        spots[i, 6] = 15.0 + i * 0.001
        spots[i, 7] = 5.0
        spots[i, 9] = 0  # all in scan 0
    return out_dir, spots


def test_process_spots_dedup_shared_ring_omega_eta(tmp_path):
    out_dir, spots = _build_fixture(tmp_path)
    _, keys_r, ids_r = open_all_three(out_dir)
    # unique_key_arr layout: [voxNr, SpotID, nMatches, nIDs, bestSolIdx].
    # Two unique grains, each pointing at its voxel's only candidate.
    unique_key_arr = np.array([
        [0, 100, 3, 3, 0],
        [1, 200, 3, 3, 0],
    ], dtype=np.uint64)
    sl = process_spots(
        unique_key_arr=unique_key_arr,
        all_spots=spots,
        keys_reader=keys_r,
        ids_reader=ids_r,
        tol_ome=0.5,
        tol_eta=0.5,
    )
    # Shared spot 3 between grains 0 and 1 with the same omega/eta within tolerance
    # → it's a duplicate, dropped from both grains.
    spot_ids = sorted(s.merged_id for s in sl.spots)
    assert 3 not in spot_ids, f"shared spot 3 should be dropped, got {spot_ids}"


def test_process_spots_returns_empty_on_invalid_ids(tmp_path):
    """Spots referencing out-of-range IDs are skipped (logged as invalid)."""
    out_dir = tmp_path / "Output"
    out_dir.mkdir()
    keys_per_vox = [np.array([[1, 2, 2, 0]], dtype=np.uint64)]
    vals_per_vox = [np.zeros((1, 16), dtype=np.float64)]
    ids_per_vox = [np.array([99, 100], dtype=np.int32)]  # both out of range
    write_vals_bin(out_dir / "IndexBest_all.bin", vals_per_vox)
    write_keys_bin(out_dir / "IndexKey_all.bin", keys_per_vox)
    write_ids_bin(out_dir / "IndexBest_IDs_all.bin", ids_per_vox)
    spots = np.zeros((3, 10), dtype=np.float64)
    spots[:, 4] = np.arange(1, 4)  # IDs 1, 2, 3
    spots[:, 5] = 1

    _, keys_r, ids_r = open_all_three(out_dir)
    unique_key_arr = np.array([[0, 1, 2, 2, 0]], dtype=np.uint64)
    sl = process_spots(
        unique_key_arr=unique_key_arr,
        all_spots=spots,
        keys_reader=keys_r,
        ids_reader=ids_r,
        tol_ome=0.5,
        tol_eta=0.5,
    )
    assert sl.n_invalid_ids == 2
    assert len(sl.spots) == 0
