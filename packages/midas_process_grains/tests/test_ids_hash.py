"""IDsHash.csv reader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_process_grains.io.ids_hash import IDsHash, load_ids_hash


def test_load_ids_hash(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text(
        "3 1 738333 1.270918\n"
        "4 738333 1702106 1.083843\n"
        "5 1702106 2287190 1.037701\n"
    )
    h = load_ids_hash(p)
    assert isinstance(h, IDsHash)
    np.testing.assert_array_equal(h.ring_nrs, [3, 4, 5])
    np.testing.assert_array_equal(h.id_starts, [1, 738333, 1702106])
    np.testing.assert_allclose(h.d_spacings, [1.270918, 1.083843, 1.037701])


def test_d_for_spot_id_in_range(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text(
        "3 1 1000 1.27\n"
        "4 1000 2000 1.08\n"
    )
    h = load_ids_hash(p)
    assert h.d_for_spot_id(1) == 1.27
    assert h.d_for_spot_id(999) == 1.27
    assert h.d_for_spot_id(1000) == 1.08
    assert h.d_for_spot_id(1999) == 1.08


def test_ring_for_spot_id_out_of_range(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text("3 1 1000 1.27\n")
    h = load_ids_hash(p)
    assert h.ring_for_spot_id(0) == -1
    assert h.ring_for_spot_id(2000) == -1
    assert h.ring_for_spot_id(500) == 3


def test_d_for_spot_ids_vectorised(tmp_path: Path):
    p = tmp_path / "IDsHash.csv"
    p.write_text("3 1 1000 1.27\n4 1000 2000 1.08\n")
    h = load_ids_hash(p)
    sids = np.array([100, 1500, 2500, 0])
    d = h.d_for_spot_ids(sids)
    np.testing.assert_allclose(d, [1.27, 1.08, 0.0, 0.0])
