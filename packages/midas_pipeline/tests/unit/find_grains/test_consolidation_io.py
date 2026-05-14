"""Tests for the consolidated file I/O readers + writers."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    CONSOLIDATED_KEY_COLS,
    CONSOLIDATED_VALS_COLS,
    open_ids,
    open_keys,
    open_vals,
    write_ids_bin,
    write_keys_bin,
    write_vals_bin,
)


def test_write_read_vals_bin_3voxels_2cands(tmp_path):
    """Synthesize a 3-voxel × 2-candidate file and assert exact float64 byte equality."""
    rng = np.random.default_rng(123)
    vals_per_vox = [rng.normal(size=(2, CONSOLIDATED_VALS_COLS)).astype(np.float64) for _ in range(3)]
    path = tmp_path / "IndexBest_all.bin"
    write_vals_bin(path, vals_per_vox)
    reader = open_vals(path)
    assert reader.n_voxels == 3
    np.testing.assert_array_equal(reader.n_sol_arr, np.array([2, 2, 2], dtype=np.int32))
    for v in range(3):
        got = reader.get_vals(v)
        assert got is not None
        assert got.shape == (2, CONSOLIDATED_VALS_COLS)
        np.testing.assert_array_equal(got.tobytes(), vals_per_vox[v].tobytes())


def test_write_read_keys_bin(tmp_path):
    keys_per_vox = [
        np.array([[10, 5, 0, 0]], dtype=np.uint64),
        np.array([], dtype=np.uint64).reshape(0, 0),
        np.array([[7, 3, 1, 0], [8, 4, 1, 0]], dtype=np.uint64),
    ]
    path = tmp_path / "IndexKey_all.bin"
    write_keys_bin(path, keys_per_vox)
    r = open_keys(path)
    assert r.n_voxels == 3
    np.testing.assert_array_equal(r.n_sol_arr, np.array([1, 0, 2], dtype=np.int32))
    g0 = r.get_keys(0)
    assert g0.shape == (1, CONSOLIDATED_KEY_COLS)
    assert int(g0[0, 0]) == 10
    assert r.get_keys(1) is None
    g2 = r.get_keys(2)
    assert g2.shape == (2, CONSOLIDATED_KEY_COLS)


def test_write_read_ids_bin(tmp_path):
    ids_per_vox = [
        np.array([1, 5, 8], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([42], dtype=np.int32),
    ]
    path = tmp_path / "IndexBest_IDs_all.bin"
    write_ids_bin(path, ids_per_vox)
    r = open_ids(path)
    assert r.n_voxels == 3
    np.testing.assert_array_equal(r.n_sol_arr, np.array([3, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(r.get_ids(0), ids_per_vox[0])
    assert r.get_ids(1) is None
    np.testing.assert_array_equal(r.get_ids(2), ids_per_vox[2])


def test_vals_bin_header_layout_is_int32_int32_int64(tmp_path):
    """Match the C ``IndexerConsolidatedIO.h`` layout byte-for-byte."""
    arr = np.arange(16, dtype=np.float64).reshape(1, 16)
    path = tmp_path / "IndexBest_all.bin"
    write_vals_bin(path, [arr])
    data = path.read_bytes()
    # Header: int32 nVoxels = 1; int32 nSolArr[1] = [1]; int64 offArr[1] = [16].
    n_vox = np.frombuffer(data[:4], dtype=np.int32)[0]
    assert n_vox == 1
    n_sol = np.frombuffer(data[4:8], dtype=np.int32)[0]
    assert n_sol == 1
    off = np.frombuffer(data[8:16], dtype=np.int64)[0]
    # header_size = 4 + 4 + 8 = 16
    assert int(off) == 16
    payload = np.frombuffer(data[16:], dtype=np.float64)
    np.testing.assert_array_equal(payload, arr.ravel())


def test_kind_mismatch_raises(tmp_path):
    path = tmp_path / "x.bin"
    write_vals_bin(path, [np.zeros((1, 16))])
    r = open_vals(path)
    with pytest.raises(ValueError):
        r.get_keys(0)
    with pytest.raises(ValueError):
        r.get_ids(0)
