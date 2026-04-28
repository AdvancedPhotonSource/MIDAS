"""Tests for read_params (paramstest.txt parser)."""

import textwrap

import pytest

from midas_index.io import read_params


def _write(tmp_path, body):
    p = tmp_path / "paramstest.txt"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


def test_minimal(tmp_path):
    p = read_params(_write(tmp_path, """
        Wavelength 0.172979
        Distance 1000000
        SpaceGroup 225
        LatticeConstant 4.08 4.08 4.08 90 90 90
        StepsizePos 5
        StepsizeOrient 0.5
        MarginOme 0.5
        MarginRadius 200
        MarginRadial 200
        MarginEta 1
        EtaBinSize 0.1
        OmeBinSize 0.1
        ExcludePoleAngle 1
        MinMatchesToAcceptFrac 0.6
        OmegaRange -180 180
        BoxSize -1500000 1500000 -1500000 1500000
        OutputFolder /tmp/out
    """))
    assert p.Wavelength == pytest.approx(0.172979)
    assert p.Distance == pytest.approx(1_000_000.0)
    assert p.SpaceGroup == 225
    assert p.LatticeConstant == (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    assert p.StepsizeOrient == 0.5
    assert p.MarginRad == 200.0
    assert p.MarginRadial == 200.0
    assert p.OmegaRanges == [(-180.0, 180.0)]
    assert p.BoxSizes == [(-1_500_000.0, 1_500_000.0, -1_500_000.0, 1_500_000.0)]
    assert p.UseFriedelPairs == 0
    assert p.OutputFolder == "/tmp/out"
    assert p.isGrainsInput is False


def test_aliases_collapse_to_canonical_field(tmp_path):
    p = read_params(_write(tmp_path, """
        Lsd 999000
        LatticeParameter 3.6 3.6 3.6 90 90 90
        MarginRadius 150
        StepSizeOrient 0.25
        Completeness 0.75
        MinEta 1.5
    """))
    assert p.Distance == 999_000.0           # Lsd  -> Distance
    assert p.LatticeConstant[0] == 3.6        # LatticeParameter -> LatticeConstant
    assert p.MarginRad == 150.0               # MarginRadius -> MarginRad
    assert p.StepsizeOrient == 0.25           # StepSizeOrient -> StepsizeOrient
    assert p.MinMatchesToAcceptFrac == 0.75   # Completeness -> MinMatchesToAcceptFrac
    assert p.ExcludePoleAngle == 1.5          # MinEta -> ExcludePoleAngle


def test_repeated_keys_accumulate(tmp_path):
    p = read_params(_write(tmp_path, """
        RingNumbers 1
        RingNumbers 2
        RingNumbers 5
        RingsToExcludeFraction 3
        RingsToExcludeFraction 7
        RingRadii 56000
        RingRadii 81000
        RingRadii 142000
        OmegaRange -180 0
        OmegaRange 0 180
        BoxSize -1 1 -1 1
        BoxSize -2 2 -2 2
    """))
    assert p.RingNumbers == [1, 2, 5]
    assert p.RingsToReject == [3, 7]
    # RingRadii is sparse-by-ring-index per IndexerOMP.c:1535-1538
    assert p.RingRadii == {1: 56000.0, 2: 81000.0, 5: 142000.0}
    assert p.get_ring_radius(2) == 81000.0
    assert p.get_ring_radius(99) == 0.0
    assert p.highest_ring_nr() == 5
    assert len(p.OmegaRanges) == 2
    assert len(p.BoxSizes) == 2


def test_grains_file_sets_mode_a(tmp_path):
    p = read_params(_write(tmp_path, """
        GrainsFile Grains.csv
        UseFriedelPairs 1
    """))
    assert p.isGrainsInput is True
    assert p.GrainsFileName == "Grains.csv"
    assert p.UseFriedelPairs == 1


def test_unknown_key_emits_warning(tmp_path):
    path = _write(tmp_path, """
        Wavelength 0.17
        Bogus 42
    """)
    with pytest.warns(UserWarning, match="Bogus"):
        read_params(path)


def test_blank_and_comment_lines_skipped(tmp_path):
    p = read_params(_write(tmp_path, """

        # this is a comment
        Wavelength 0.17

        # another
        Distance 999
    """))
    assert p.Wavelength == 0.17
    assert p.Distance == 999.0


def test_big_det_size_silently_ignored(tmp_path):
    p = read_params(_write(tmp_path, """
        BigDetSize 8192
        Wavelength 0.17
    """))
    assert p.Wavelength == 0.17
    # No attribute leak for BigDet
    assert not hasattr(p, "BigDetSize")
