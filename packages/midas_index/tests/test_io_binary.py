"""Tests for binary readers (Spots.bin, Data.bin, nData.bin)."""

import numpy as np
import pytest

from midas_index.io import read_bins, read_spots


def test_read_spots_roundtrip(tmp_path):
    # 50 fake spots, 9 doubles each
    arr = np.arange(50 * 9, dtype=np.float64).reshape(50, 9) + 0.5
    (tmp_path / "Spots.bin").write_bytes(arr.tobytes())

    n, obs = read_spots(tmp_path)
    assert n == 50
    assert obs.shape == (50, 9)
    assert obs.dtype == np.float64
    np.testing.assert_array_equal(np.asarray(obs), arr)


def test_read_spots_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_spots(tmp_path)


def test_read_spots_bad_size(tmp_path):
    # 8 doubles is not a multiple of 9
    (tmp_path / "Spots.bin").write_bytes(np.zeros(8, dtype=np.float64).tobytes())
    with pytest.raises(ValueError, match="multiple of 9"):
        read_spots(tmp_path)


def test_read_bins_roundtrip(tmp_path):
    data = np.arange(100, dtype=np.int32)
    ndata = np.arange(40, dtype=np.int32)  # 20 bins x 2 ints (count, offset)
    (tmp_path / "Data.bin").write_bytes(data.tobytes())
    (tmp_path / "nData.bin").write_bytes(ndata.tobytes())

    d, nd = read_bins(tmp_path)
    assert d.dtype == np.int32
    assert nd.dtype == np.int32
    np.testing.assert_array_equal(np.asarray(d), data)
    np.testing.assert_array_equal(np.asarray(nd), ndata)


def test_read_bins_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data.bin"):
        read_bins(tmp_path)
    (tmp_path / "Data.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="nData.bin"):
        read_bins(tmp_path)


def test_read_bins_odd_ndata_size(tmp_path):
    (tmp_path / "Data.bin").write_bytes(np.zeros(4, dtype=np.int32).tobytes())
    (tmp_path / "nData.bin").write_bytes(np.zeros(3, dtype=np.int32).tobytes())
    with pytest.raises(ValueError, match="multiple of 2"):
        read_bins(tmp_path)
