"""voxel_scan_pos.bin sidecar tests.

Tightens the contract from §11c of the plan: the sidecar is float64 with
length equal to ``n_scans`` (the 1-D Y array, not the 2-D voxel grid).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from midas_transforms.bin_data.voxel_binner import bin_data_scanning
from midas_transforms.io import binary as bio
from midas_transforms.io import csv as csv_io
from midas_transforms.params import ParamsTest, write_paramstest


def _make_paramstest() -> ParamsTest:
    p = ParamsTest()
    p.Wavelength = 0.18
    p.Lsd = 1_000_000.0
    p.px = 200.0
    p.MarginOme = 1.0
    p.MarginEta = 500.0
    p.EtaBinSize = 5.0
    p.OmeBinSize = 5.0
    p.StepSizeOrient = 0.2
    p.NoSaveAll = 0
    p.RingNumbers = [1, 2]
    p.RingRadii = [500.0, 700.0]
    p.LatticeConstant = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    p.SpaceGroup = 225
    return p


def _make_minimal_scan_csv(path: Path, scan_idx: int):
    """One spot per scan, just to exercise the sidecar path."""
    rows = np.zeros((1, 18), dtype=np.float64)
    rows[0, 0] = 100.0 * (scan_idx + 1)             # YLab
    rows[0, 1] = 200.0 * (scan_idx + 1)             # ZLab
    rows[0, 2] = -30.0 + 5.0 * scan_idx              # Omega
    rows[0, 3] = 5.0                                 # GrainRadius
    rows[0, 4] = scan_idx + 1                        # SpotID
    rows[0, 5] = 1                                   # RingNumber
    rows[0, 6] = -30.0                               # Eta
    rows[0, 7] = 0.1                                 # Ttheta
    rows[0, 8] = rows[0, 2]                          # OmegaIni
    csv_io.write_inputall_extra_csv(path, rows)


@pytest.fixture
def tmp_min_pf_dir(tmp_path: Path) -> Path:
    write_paramstest(_make_paramstest(), tmp_path / "paramstest.txt")
    for s in range(5):
        _make_minimal_scan_csv(tmp_path / f"InputAllExtraInfoFittingAll{s}.csv", s)
    return tmp_path


def test_voxel_scan_pos_bin_dtype_and_length(tmp_min_pf_dir: Path):
    sp = np.linspace(-10.0, 10.0, 5).astype(np.float64)
    bin_data_scanning(
        result_folder=tmp_min_pf_dir,
        n_scans=5,
        scan_positions=sp,
    )
    arr = np.fromfile(tmp_min_pf_dir / "voxel_scan_pos.bin", dtype=np.float64)
    assert arr.shape == (5,)
    np.testing.assert_allclose(arr, sp, rtol=0, atol=0)


def test_voxel_scan_pos_bin_reader_roundtrip(tmp_min_pf_dir: Path):
    sp = np.array([-3.0, 1.5, 4.0, 8.0, 11.5])
    bin_data_scanning(
        result_folder=tmp_min_pf_dir,
        n_scans=5,
        scan_positions=sp,
    )
    arr = bio.read_voxel_scan_pos_bin(tmp_min_pf_dir / "voxel_scan_pos.bin")
    np.testing.assert_allclose(arr, sp, rtol=0, atol=0)


def test_positions_csv_matches_scan_positions(tmp_min_pf_dir: Path):
    sp = np.array([-3.0, 1.5, 4.0, 8.0, 11.5])
    bin_data_scanning(
        result_folder=tmp_min_pf_dir,
        n_scans=5,
        scan_positions=sp,
    )
    text = (tmp_min_pf_dir / "positions.csv").read_text().strip().split("\n")
    parsed = np.array([float(line.strip()) for line in text])
    np.testing.assert_allclose(parsed, sp, rtol=0, atol=1e-9)


def test_positions_csv_optional(tmp_min_pf_dir: Path):
    sp = np.array([-3.0, 1.5, 4.0, 8.0, 11.5])
    # Clean any pre-existing positions.csv from prior tests.
    (tmp_min_pf_dir / "positions.csv").unlink(missing_ok=True)
    bin_data_scanning(
        result_folder=tmp_min_pf_dir,
        n_scans=5,
        scan_positions=sp,
        write_positions_csv=False,
    )
    assert not (tmp_min_pf_dir / "positions.csv").exists()
    # voxel_scan_pos.bin should still be written.
    assert (tmp_min_pf_dir / "voxel_scan_pos.bin").exists()


def test_voxel_binner_n_scans_mismatch(tmp_min_pf_dir: Path):
    with pytest.raises(ValueError, match="scan_positions has .* entries"):
        bin_data_scanning(
            result_folder=tmp_min_pf_dir,
            n_scans=5,
            scan_positions=np.array([0.0, 1.0, 2.0]),   # 3 != 5
        )
