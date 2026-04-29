"""Shared fixtures."""

import math
from pathlib import Path

import numpy as np
import pytest

from midas_transforms.params import ParamsTest, ZarrParams


@pytest.fixture
def tiny_paramstest(tmp_path: Path) -> ParamsTest:
    p = ParamsTest()
    p.Wavelength = 0.18
    p.Lsd = 1_000_000.0      # 1 m in µm
    p.px = 200.0
    p.MarginOme = 1.0
    p.MarginEta = 500.0
    p.EtaBinSize = 5.0
    p.OmeBinSize = 5.0
    p.StepSizeOrient = 0.2
    p.NoSaveAll = 0
    p.RingNumbers = [1, 2, 3]
    p.RingRadii = [500.0, 700.0, 900.0]
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    p.SpaceGroup = 225
    return p


@pytest.fixture
def tiny_inputall(tiny_paramstest: ParamsTest):
    """A small, deterministic synthetic InputAll matrix.

    8 cols: YLab, ZLab, Omega, GrainRadius, SpotID, RingNumber, Eta, Ttheta.
    """
    rng = np.random.default_rng(seed=0)
    parts = []
    sid = 1
    # Choose eta values that avoid 0 and ±180 (where omemargin would blow up
    # via 1/|sin(eta)|). These match real-world peakfit output behaviour.
    eta_grid = [-150.0, -120.0, -60.0, -30.0, 30.0, 60.0, 120.0, 150.0]
    for ring_idx, (rn, rr) in enumerate(zip(tiny_paramstest.RingNumbers, tiny_paramstest.RingRadii)):
        for eta in eta_grid:
            yl = -rr * math.sin(eta * math.pi / 180.0) * tiny_paramstest.px
            zl = rr * math.cos(eta * math.pi / 180.0) * tiny_paramstest.px
            omega = -90 + 180 * rng.random()
            ttheta = math.degrees(math.atan2(rr * tiny_paramstest.px, tiny_paramstest.Lsd))
            parts.append([yl, zl, omega, 5.0, float(sid), float(rn), eta, ttheta])
            sid += 1
    return np.array(parts, dtype=np.float64)


@pytest.fixture
def tiny_inputall_extra(tiny_inputall: np.ndarray):
    """16-col extra info — zeros except the SpotID/Ring/Eta etc. carried from cols 0-7."""
    n = tiny_inputall.shape[0]
    out = np.zeros((n, 18), dtype=np.float64)
    out[:, :8] = tiny_inputall
    out[:, 8] = tiny_inputall[:, 2]   # OmegaIni
    out[:, 14] = 1.0                  # IntegratedIntensity
    return out


@pytest.fixture
def tmp_inputall_dir(
    tmp_path: Path, tiny_paramstest: ParamsTest,
    tiny_inputall: np.ndarray, tiny_inputall_extra: np.ndarray,
) -> Path:
    """Write the synthetic inputs to a tmp dir for stage-CLI tests."""
    from midas_transforms.io import csv as csv_io
    from midas_transforms.params import write_paramstest
    csv_io.write_inputall_csv(tmp_path / "InputAll.csv", tiny_inputall)
    csv_io.write_inputall_extra_csv(tmp_path / "InputAllExtraInfoFittingAll.csv", tiny_inputall_extra)
    write_paramstest(tiny_paramstest, tmp_path / "paramstest.txt")
    return tmp_path
