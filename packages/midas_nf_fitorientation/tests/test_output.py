"""Tests for the MicFileBinary / AllMatches writers (byte layout)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from midas_nf_fitorientation.output import (
    MicRecord,
    MicWriter,
    MIC_RECORD_BYTES,
    MIC_RECORD_DOUBLES,
)


def test_mic_record_size():
    assert MIC_RECORD_DOUBLES == 11
    assert MIC_RECORD_BYTES == 88


def test_writer_pre_allocates_files(tmp_path):
    out = tmp_path / "mic.bin"
    with MicWriter(out, n_voxels=10, n_saves=2, block_nr=3) as w:
        rec = MicRecord(
            best_row_nr=42, n_winners=5, fit_time_s=1.23,
            xs=10.0, ys=20.0, grid_size=5.0, ud=1.0,
            euler_a=0.1, euler_b=0.2, euler_c=0.3, frac_overlap=0.99,
        )
        w.write_mic(7, rec)
        sols = np.array([[0.1, 0.2, 0.3, 0.99]], dtype=np.float64)
        w.write_all_matches(7, n_winners=5, xs=10.0, ys=20.0,
                             grid_size=5.0, ud=1.0, sols=sols)

    # Mic file is exactly 10 records of 88 bytes
    assert out.stat().st_size == 10 * 88

    # Read voxel 7 back and check fields
    with open(out, "rb") as f:
        f.seek(7 * 88)
        data = f.read(88)
    fields = struct.unpack("11d", data)
    assert fields[0] == 42.0
    assert fields[1] == 5.0
    assert fields[2] == pytest.approx(1.23)
    assert fields[10] == pytest.approx(0.99)

    # AllMatches file: 10 records, each 7 + 4*2 = 15 doubles
    am = Path(str(out) + ".AllMatches")
    assert am.stat().st_size == 10 * 15 * 8
    with open(am, "rb") as f:
        f.seek(7 * 15 * 8)
        data = f.read(15 * 8)
    fields = struct.unpack("15d", data)
    assert fields[0] == 3.0          # block_nr
    assert fields[1] == 5.0          # n_winners
    assert fields[3] == 10.0         # xs
    assert fields[7 + 0] == pytest.approx(0.1)
    assert fields[7 + 3] == pytest.approx(0.99)
    # Slot 1 zero-filled
    assert fields[7 + 4] == 0.0
    assert fields[7 + 7] == 0.0


def test_writer_other_voxels_zero(tmp_path):
    out = tmp_path / "mic.bin"
    with MicWriter(out, n_voxels=4, n_saves=1) as w:
        rec = MicRecord(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        w.write_mic(0, rec)
    with open(out, "rb") as f:
        f.seek(2 * 88)  # voxel 2
        data = f.read(88)
    fields = struct.unpack("11d", data)
    assert all(x == 0.0 for x in fields)
