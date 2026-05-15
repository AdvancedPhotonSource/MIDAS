"""Tests for binary readers (Spots.bin, Data.bin, nData.bin)."""

import numpy as np
import pytest

from midas_index.io import read_bins, read_spots
from midas_index.io.binary import read_bins_scanning


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


# ---------------------------------------------------------------------------
# Scanning-mode readers — SaveBinDataScanning.c writes int64 (size_t) with
# (count, offset) in nData.bin and (spot_id, scan_nr) in Data.bin.
# ---------------------------------------------------------------------------


def _write_scanning_fixture(tmp_path, *, spots, ndata_pairs):
    """Lay out a small int64 (Data.bin, nData.bin) pair the way
    SaveBinDataScanning.c does — (spot_id, scan_nr) per spot in Data.bin and
    (count, offset_in_spot_units) per bin in nData.bin."""
    data64 = np.asarray(spots, dtype=np.int64).reshape(-1)
    ndata64 = np.asarray(ndata_pairs, dtype=np.int64).reshape(-1)
    (tmp_path / "Data.bin").write_bytes(data64.tobytes())
    (tmp_path / "nData.bin").write_bytes(ndata64.tobytes())


def test_read_bins_scanning_projects_spot_ids_and_ndata(tmp_path):
    # 3 bins, occupancies (2, 0, 1); spot pairs are (spot_id, scan_nr).
    #   bin 0 spots: (100, 3), (200, 7)
    #   bin 1 spots: none
    #   bin 2 spots: (300, 11)
    # Offsets are in "spot units": 0, 2, 2.
    _write_scanning_fixture(
        tmp_path,
        spots=[(100, 3), (200, 7), (300, 11)],
        ndata_pairs=[(2, 0), (0, 2), (1, 2)],
    )
    data, ndata = read_bins_scanning(tmp_path)
    assert data.dtype == np.int32
    assert ndata.dtype == np.int32
    np.testing.assert_array_equal(data, [100, 200, 300])
    np.testing.assert_array_equal(ndata, [2, 0, 0, 2, 1, 2])


def test_read_bins_scanning_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data.bin"):
        read_bins_scanning(tmp_path)
    (tmp_path / "Data.bin").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="nData.bin"):
        read_bins_scanning(tmp_path)


def test_read_bins_scanning_rejects_odd_pair_counts(tmp_path):
    # Data.bin with 3 int64s (odd, not pair-aligned)
    (tmp_path / "Data.bin").write_bytes(np.zeros(3, dtype=np.int64).tobytes())
    (tmp_path / "nData.bin").write_bytes(np.zeros(2, dtype=np.int64).tobytes())
    with pytest.raises(ValueError, match="Data.bin"):
        read_bins_scanning(tmp_path)
    # Now make Data even but break nData's pair count.
    (tmp_path / "Data.bin").write_bytes(np.zeros(2, dtype=np.int64).tobytes())
    (tmp_path / "nData.bin").write_bytes(np.zeros(3, dtype=np.int64).tobytes())
    with pytest.raises(ValueError, match="nData.bin"):
        read_bins_scanning(tmp_path)
