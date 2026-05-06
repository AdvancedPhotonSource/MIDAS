"""Parameter-reader unit tests."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from midas_process_grains.params import ProcessGrainsParams, read_paramstest_pg


def test_read_basic(tiny_run_dir: Path):
    p = read_paramstest_pg(tiny_run_dir / "paramstest.txt")
    assert isinstance(p, ProcessGrainsParams)
    assert p.SGNr == 225
    assert p.MinNrSpots == 2
    assert p.MisoriTol == 0.25
    assert p.RingNumbers == [1, 2]
    assert p.RingRadii == [60000.0, 70000.0]
    assert p.LatticeConstant == (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)


def test_minnrspots_clamps_to_two_with_warning(tmp_path: Path):
    p = tmp_path / "paramstest.txt"
    p.write_text(
        "LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
        "Wavelength 0.17;\n"
        "Distance 800000;\n"
        "px 200;\n"
        "RingNumbers 1;\n"
        "RingRadii 60000;\n"
        "MinNrSpots 1;\n"
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prm = read_paramstest_pg(p)
    assert any("MinNrSpots" in str(warning.message) for warning in w)
    assert prm.MinNrSpots == 2


def test_invalid_jaccard_tol_raises(tmp_path: Path):
    p = tmp_path / "paramstest.txt"
    p.write_text(
        "LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
        "Wavelength 0.17;\n"
        "Distance 800000;\n"
        "px 200;\n"
        "RingNumbers 1;\n"
        "RingRadii 60000;\n"
        "JaccardTol 1.5;\n"
    )
    with pytest.raises(ValueError, match="JaccardTol"):
        read_paramstest_pg(p)


def test_invalid_strain_method_raises(tmp_path: Path):
    p = tmp_path / "paramstest.txt"
    p.write_text(
        "LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
        "Wavelength 0.17;\n"
        "Distance 800000;\n"
        "px 200;\n"
        "RingNumbers 1;\n"
        "RingRadii 60000;\n"
        "StrainMethod nelder_mead;\n"
    )
    with pytest.raises(ValueError, match="StrainMethod"):
        read_paramstest_pg(p)


def test_paramstest_passes_through_unknown_keys(tiny_run_dir: Path):
    """Unknown keys land in p.raw without raising."""
    extra = tiny_run_dir / "paramstest.txt"
    text = extra.read_text() + "ExperimentalKey 42 abc;\n"
    extra.write_text(text)
    p = read_paramstest_pg(extra)
    assert "ExperimentalKey" in p.raw
