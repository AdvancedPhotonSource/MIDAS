"""Tests for paramstest.txt parser + FitConfig.from_param_file."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from midas_fit_grain import FitConfig


def _write_param_file(tmp_path: Path) -> Path:
    """Realistic paramstest.txt slice (semi-colon terminators allowed)."""
    body = textwrap.dedent(
        """
        % a sample paramstest.txt
        LatticeParameter 4.04 4.04 4.04 90 90 90;
        SpaceGroup 225;
        Wavelength 0.1729;
        Distance 1000000.0;
        px 200.0;
        Wedge 0.0;
        MaxRingRad 200000.0;
        MarginRadius 800.0;
        MarginEta 1.5;
        MarginOme 0.5;
        MarginRadial 800.0;
        EtaBinSize 2.0;
        OmeBinSize 2.0;
        ExcludePoleAngle 6.0;
        MargABC 0.05;
        MargABG 0.05;
        DoDynamicReassignment 1;
        FitAllAtOnce 0;
        RingNumbers 1;
        RingRadii 1827.5;
        RingNumbers 2;
        RingRadii 2110.5;
        RingNumbers 4;
        RingRadii 2987.0;
        OmegaRange -180 180;
        BoxSize -1000 1000 -1000 1000;
        OutputFolder /tmp/midas_out/Output;
        ResultFolder /tmp/midas_out/Results;
        SpotsFileName Spots.bin
        IDsFileName SpotsToIndex.csv
        RefinementFileName InputAllExtraInfoFittingAll.csv
        StepsizePos 100;
        StepsizeOrient 0.2;
        BeamSize 1000;
        UseFriedelPairs 1;
        # comment
        """
    ).strip()
    p = tmp_path / "paramstest.txt"
    p.write_text(body)
    return p


def test_from_param_file_basic(tmp_path):
    p = _write_param_file(tmp_path)
    cfg = FitConfig.from_param_file(p)
    assert cfg.Lsd == 1_000_000.0
    assert cfg.px == 200.0
    assert cfg.Wavelength == pytest.approx(0.1729)
    assert cfg.LatticeConstant == (4.04, 4.04, 4.04, 90.0, 90.0, 90.0)
    assert cfg.SpaceGroup == 225
    assert cfg.MarginRadius == 800.0
    assert cfg.MinEta == 6.0
    assert cfg.RhoD == 200_000.0
    assert cfg.RingNumbers == [1, 2, 4]
    assert cfg.RingRadii == [1827.5, 2110.5, 2987.0]
    assert cfg.ring_radius(2) == 2110.5
    assert cfg.ring_radius(99) == 0.0
    assert cfg.OmegaRanges == [(-180.0, 180.0)]
    assert cfg.BoxSizes == [(-1000.0, 1000.0, -1000.0, 1000.0)]
    assert cfg.OutputFolder.endswith("Output")
    assert cfg.ResultFolder.endswith("Results")
    # Mode follows FitAllAtOnce when not overridden.
    assert cfg.mode == "iterative"


def test_mode_override_wins(tmp_path):
    p = _write_param_file(tmp_path)
    cfg = FitConfig.from_param_file(p, mode="all_at_once")
    assert cfg.mode == "all_at_once"


def test_mode_from_fitallatone_flag(tmp_path):
    body = (tmp_path / "paramstest.txt")
    body.write_text("FitAllAtOnce 1\nRingNumbers 1\nRingRadii 1.0\n")
    cfg = FitConfig.from_param_file(body)
    assert cfg.FitAllAtOnce == 1
    assert cfg.mode == "all_at_once"


def test_solver_loss_overrides(tmp_path):
    p = _write_param_file(tmp_path)
    cfg = FitConfig.from_param_file(
        p, solver="lm", loss="internal_angle", max_iter=42,
    )
    assert cfg.solver == "lm"
    assert cfg.loss == "internal_angle"
    assert cfg.max_iter == 42


def test_unknown_field_kwargs_raise(tmp_path):
    p = _write_param_file(tmp_path)
    with pytest.raises(AttributeError):
        FitConfig.from_param_file(p, not_a_real_field=1)


def test_distance_lsd_alias(tmp_path):
    p = tmp_path / "paramstest.txt"
    p.write_text("Lsd 999.0\nRingNumbers 1\nRingRadii 1.0\n")
    cfg = FitConfig.from_param_file(p)
    assert cfg.Lsd == 999.0


def test_warns_on_unknown_key(tmp_path):
    p = tmp_path / "paramstest.txt"
    p.write_text("ThisIsNotAKey 1\n")
    with pytest.warns(UserWarning, match="unrecognized"):
        FitConfig.from_param_file(p)
