"""UniqueIndexSingleKey.bin pwrite semantics — slot-by-slot uint64 writes."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    read_unique_index_single_key,
    write_unique_index_single_key,
)


def test_pwrite_row_byte_offsets_match_C_semantics(tmp_path):
    """Each row written at byte offset = 5 * 8 * voxNr."""
    n_voxels = 7
    path = tmp_path / "UniqueIndexSingleKey.bin"
    rows = [
        (0, np.array([0, 100, 5, 4, 0], dtype=np.uint64)),
        (3, np.array([3, 300, 7, 6, 1], dtype=np.uint64)),
        (5, np.array([5, 500, 9, 8, 2], dtype=np.uint64)),
    ]
    write_unique_index_single_key(path, n_voxels, rows)
    data = path.read_bytes()
    # File size = 5 * 8 * n_voxels.
    assert len(data) == 5 * 8 * n_voxels
    arr = read_unique_index_single_key(path)
    assert arr.shape == (n_voxels, 5)
    # Voxels 0, 3, 5 have written rows; rest are zero.
    np.testing.assert_array_equal(arr[0], rows[0][1])
    np.testing.assert_array_equal(arr[3], rows[1][1])
    np.testing.assert_array_equal(arr[5], rows[2][1])
    for v in (1, 2, 4, 6):
        assert (arr[v] == 0).all(), v


def test_invalid_vox_nr_raises(tmp_path):
    path = tmp_path / "x.bin"
    with pytest.raises(IndexError):
        write_unique_index_single_key(path, 3, [(5, np.zeros(5, dtype=np.uint64))])


def test_wrong_row_shape_raises(tmp_path):
    path = tmp_path / "x.bin"
    with pytest.raises(ValueError):
        write_unique_index_single_key(path, 3, [(0, np.zeros(4, dtype=np.uint64))])
