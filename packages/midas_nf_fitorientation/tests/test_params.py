"""Smoke + functional tests for the paramfile parser."""
from __future__ import annotations

import textwrap

import pytest

from midas_nf_fitorientation.params import FitParams, parse_paramfile


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "params.txt"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_parses_minimal_single_distance(tmp_path):
    path = _write(tmp_path, """
        nDistances 1
        Lsd 1000000.0
        BC 1024 1024
        px 200.0
        OmegaStart 0.0
        OmegaStep 0.25
        StartNr 1
        EndNr 1440
        Wavelength 0.172979
        ExcludePoleAngle 6.0
        SpaceGroup 225
        LatticeParameter 4.08 4.08 4.08 90 90 90
        OrientTol 1.0
        MinFracAccept 0.6
        OutputDirectory /tmp/scratch
        DataDirectory /tmp/scratch
        ReducedFileName scratch.bin
        MicFileBinary mic.bin
        OmegaRange -180.0 180.0
        BoxSize 0 2048 0 2048
    """)
    p = parse_paramfile(path)
    assert p.n_distances == 1
    assert p.Lsd == [1000000.0]
    assert p.ybc == [1024.0]
    assert p.zbc == [1024.0]
    assert p.px == 200.0
    assert p.start_nr == 1 and p.end_nr == 1440
    assert p.lattice_constant == (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    assert p.orient_tol == 1.0
    assert p.n_frames_per_distance == 1440
    assert p.out_dir == "/tmp/scratch"


def test_multi_distance_lists_match(tmp_path):
    path = _write(tmp_path, """
        nDistances 3
        Lsd 1000000
        Lsd 1500000
        Lsd 2000000
        BC 1024 1024
        BC 1023 1024
        BC 1022 1024
        px 200
    """)
    p = parse_paramfile(path)
    assert p.n_distances == 3
    assert len(p.Lsd) == 3
    assert len(p.ybc) == 3
    assert len(p.zbc) == 3


def test_rings_to_use_and_box_size_stack(tmp_path):
    path = _write(tmp_path, """
        nDistances 1
        Lsd 1000000
        BC 0 0
        RingsToUse 1
        RingsToUse 2
        RingsToUse 4
        BoxSize 0 100 0 100
        BoxSize 200 300 200 300
        OmegaRange -90 90
    """)
    p = parse_paramfile(path)
    assert p.rings_to_use == [1, 2, 4]
    assert len(p.box_sizes) == 2
    assert len(p.omega_ranges) == 1


def test_new_keys_recognised(tmp_path):
    path = _write(tmp_path, """
        nDistances 1
        Lsd 1000000
        BC 0 0
        RefineWedge 1
        WedgeTol 0.1
        TikhonovCalibration 0.01
        TikhonovSigmaLsd 50.0
        GaussianSplatSigmaPx 1.5
    """)
    p = parse_paramfile(path)
    assert p.refine_wedge is True
    assert p.wedge_tol == pytest.approx(0.1)
    assert p.tikhonov_calibration == pytest.approx(0.01)
    assert p.tikhonov_sigma_lsd == pytest.approx(50.0)
    assert p.gaussian_splat_sigma_px == pytest.approx(1.5)


def test_n_distances_mismatch_raises(tmp_path):
    path = _write(tmp_path, """
        nDistances 2
        Lsd 1000000
        BC 0 0
    """)
    with pytest.raises(ValueError, match="nDistances=2"):
        parse_paramfile(path)
