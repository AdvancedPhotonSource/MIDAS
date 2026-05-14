"""FF regression gate for the voxel_binner: ``bin_data_unified`` with
``scan_positions=None`` must produce a byte-identical ``Spots.bin`` to
today's ``bin_data``.

This is the **single most important test** for stream B: extending
``midas-transforms`` with PF semantics must not change FF behaviour.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from midas_transforms import bin_data, bin_data_unified
from midas_transforms.io import binary as bio


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def test_unified_ff_mode_byte_identical(tmp_inputall_dir: Path, tmp_path: Path):
    """``bin_data_unified(scan_positions=None)`` writes identical bytes to
    ``bin_data``."""
    # Baseline run (FF bin_data).
    ff_dir = tmp_path / "ff"
    ff_dir.mkdir()
    import shutil
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy(tmp_inputall_dir / fn, ff_dir / fn)
    bin_data(result_folder=ff_dir)

    # Unified run with scan_positions=None.
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy(tmp_inputall_dir / fn, unified_dir / fn)
    bin_data_unified(result_folder=unified_dir, scan_positions=None)

    # Every output file must be byte-identical.
    for fn in ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin"):
        assert (ff_dir / fn).exists(), f"baseline missing {fn}"
        assert (unified_dir / fn).exists(), f"unified missing {fn}"
        assert _hash_file(ff_dir / fn) == _hash_file(unified_dir / fn), (
            f"{fn} differs between baseline ``bin_data`` and "
            f"``bin_data_unified(scan_positions=None)``"
        )


def test_unified_ff_mode_no_voxel_sidecar(tmp_inputall_dir: Path):
    """FF-mode call must NOT emit ``voxel_scan_pos.bin`` or
    ``positions.csv`` sidecar files."""
    # Note: ``tmp_inputall_dir`` already contains InputAll.csv et al.
    # Pre-clean any sidecars in case a previous run left them.
    for fn in ("voxel_scan_pos.bin", "positions.csv"):
        (tmp_inputall_dir / fn).unlink(missing_ok=True)
    bin_data_unified(result_folder=tmp_inputall_dir, scan_positions=None)
    assert not (tmp_inputall_dir / "voxel_scan_pos.bin").exists(), (
        "FF mode must not emit voxel_scan_pos.bin"
    )
    # positions.csv is the legacy C indexer's per-scan Y table — FF mode
    # has no scan, so no positions.csv is emitted.
    assert not (tmp_inputall_dir / "positions.csv").exists(), (
        "FF mode must not emit positions.csv"
    )
