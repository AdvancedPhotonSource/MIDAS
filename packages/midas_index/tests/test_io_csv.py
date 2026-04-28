"""Tests for CSV / text readers (hkls.csv, SpotsToIndex.csv, Grains.csv)."""

import textwrap

import numpy as np
import pytest

from midas_index.io import (
    read_grains_csv,
    read_hkls_csv,
    read_spots_to_index_csv,
    write_spots_to_index_csv,
)


# ---------------------------------------------------------------------- hkls

_HKLS_BODY = """\
h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius
1 -1 -1 3.124 1 0.184 -0.184 -0.184 1.585 3.171 36445.0
1 1 1 3.124 1 0.184 0.184 0.184 1.585 3.171 36445.0
2 0 0 2.706 2 0.213 0.0 0.0 1.832 3.665 42100.0
2 0 0 2.706 2 0.213 0.0 0.0 1.832 3.665 42100.0
1 1 0 3.605 99 0.150 0.150 0.0 1.370 2.741 30000.0
"""


def test_read_hkls_keep_all(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p)
    assert real.shape == (5, 7)
    assert ints.shape == (5, 4)
    # Layout check: cols [g1, g2, g3, ring_nr, d_spacing, theta, radius]
    assert real[0, 3] == 1.0   # ring_nr
    assert real[0, 4] == pytest.approx(3.124, rel=1e-6)
    assert ints[0, 3] == 1     # ring_nr in int form too


def test_read_hkls_filter_by_ring(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p, ring_numbers=[1, 2])
    assert real.shape == (4, 7)        # 5th row (ring 99) dropped
    assert ints.shape == (4, 4)
    assert set(np.unique(ints[:, 3].tolist())) == {1, 2}


def test_read_hkls_empty_ring_filter_returns_empty(tmp_path):
    p = tmp_path / "hkls.csv"
    p.write_text(_HKLS_BODY)
    real, ints = read_hkls_csv(p, ring_numbers=[])
    assert real.shape == (0, 7)
    assert ints.shape == (0, 4)


# ---------------------------------------------------------------------- spots-to-index


def test_read_spots_to_index_one_int_per_line(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    p.write_text("17\n42\n8\n")
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [17, 42, 8]


def test_read_spots_to_index_two_int_per_line_first_only(tmp_path):
    # Mode A writes "newID origID" — IndexerOMP.c:2313 reads only first %d
    p = tmp_path / "SpotsToIndex.csv"
    p.write_text("100 1\n200 2\n300 3\n")
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [100, 200, 300]


def test_write_spots_to_index_roundtrip_pairs(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(p, [(100, 1), (200, 2)])
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [100, 200]
    # Verify the on-disk format matches mode-A two-int layout
    text = p.read_text().splitlines()
    assert text == ["100 1", "200 2"]


def test_write_spots_to_index_roundtrip_singles(tmp_path):
    p = tmp_path / "SpotsToIndex.csv"
    write_spots_to_index_csv(p, [10, 20, 30])
    ids = read_spots_to_index_csv(p)
    assert ids.tolist() == [10, 20, 30]


# ---------------------------------------------------------------------- grains


_GRAINS_BODY = """\
%NumGrains 2
%BeamCenter 0.000000
%BeamThickness 200.000000
%GlobalPosition 0.000000
%NumPhases 1
%PhaseInfo
%	SpaceGroup:225
%	Lattice Parameter: 4.080000 4.080000 4.080000 90.000000 90.000000 90.000000
%GrainID	O11	O12	O13	O21	O22	O23	O31	O32	O33	X	Y	Z	a	b	c	alpha	beta	gamma	DiffPos	DiffOme	DiffAngle	GrainRadius
1\t1.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t1.0\t10.0\t20.0\t30.0\tx\ty\tz\tw\tv\tu\tt\ts\tr\t50.0
2\t0.0\t-1.0\t0.0\t1.0\t0.0\t0.0\t0.0\t0.0\t1.0\t-15.0\t5.0\t-2.5\tx\ty\tz\tw\tv\tu\tt\ts\tr\t75.0
"""


def test_read_grains_csv(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text(_GRAINS_BODY)
    g = read_grains_csv(p)

    assert g["ids"].tolist() == [1, 2]
    np.testing.assert_array_equal(
        g["orient_mat"][0],
        np.eye(3),
    )
    np.testing.assert_array_equal(
        g["orient_mat"][1],
        np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]]),
    )
    np.testing.assert_array_equal(g["positions"][0], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(g["positions"][1], [-15.0, 5.0, -2.5])
    assert g["radii"].tolist() == [50.0, 75.0]


def test_read_grains_csv_bad_header(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text("not a numgrains line\n")
    with pytest.raises(ValueError, match="NumGrains"):
        read_grains_csv(p)
