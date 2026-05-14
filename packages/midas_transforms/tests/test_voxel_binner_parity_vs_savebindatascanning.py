"""Bit-exact parity vs ``SaveBinDataScanning`` C output.

Gated on a frozen-fixture path under
``packages/midas_pipeline/dev/golden_data/test_pf_5grain/``. When the
user later freezes that fixture (per the plan §11c), this test
auto-activates and exercises the gate.

Layout expected once fixture lands:
    test_pf_5grain/
    ├── inputs/
    │   ├── InputAllExtraInfoFittingAll0.csv
    │   ├── InputAllExtraInfoFittingAll1.csv
    │   ├── ...
    │   ├── paramstest.txt
    │   └── scan_positions.txt    # one Y per line
    └── golden/
        ├── Spots.bin             # C-produced reference
        ├── ExtraInfo.bin
        ├── IDsMergedScanning.csv
        ├── Data.bin
        ├── nData.bin
        └── positions.csv         # produced by mergeScansScanning (if relevant)
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest


def _golden_root() -> Path | None:
    # Search relative to this test file, walking up to find the repo root.
    # Located alongside packages/midas_pipeline/dev/golden_data/.
    here = Path(__file__).resolve()
    for parent in (here.parents[3], here.parents[4], here.parents[5]):
        candidate = parent / "packages" / "midas_pipeline" / "dev" / "golden_data" / "test_pf_5grain"
        if candidate.exists():
            return candidate
    return None


GOLDEN = _golden_root()


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.skipif(
    GOLDEN is None or not (GOLDEN / "golden" / "Spots.bin").exists(),
    reason="Golden fixture for SaveBinDataScanning not yet frozen.",
)
def test_voxel_binner_parity_spots_bin(tmp_path: Path):
    """``bin_data_scanning`` Spots.bin must be byte-identical to the C
    output."""
    from midas_transforms.bin_data.voxel_binner import bin_data_scanning

    # Copy inputs into the tmp dir (so we don't pollute the fixture).
    inputs = GOLDEN / "inputs"
    for child in inputs.iterdir():
        shutil.copy(child, tmp_path / child.name)

    # Read scan_positions.txt.
    sp_path = tmp_path / "scan_positions.txt"
    scan_positions = np.array(
        [float(x) for x in sp_path.read_text().split()],
        dtype=np.float64,
    )
    n_scans = scan_positions.shape[0]

    bin_data_scanning(
        result_folder=tmp_path,
        n_scans=n_scans,
        scan_positions=scan_positions,
    )

    # Byte-exact match against the golden fixture.
    for fn in ("Spots.bin", "ExtraInfo.bin"):
        gold = GOLDEN / "golden" / fn
        if not gold.exists():
            pytest.skip(f"golden {fn} not present")
        assert _hash_file(tmp_path / fn) == _hash_file(gold), (
            f"{fn} bytes differ from golden"
        )


@pytest.mark.skipif(
    GOLDEN is None or not (GOLDEN / "golden" / "Data.bin").exists(),
    reason="Golden Data.bin / nData.bin not yet frozen.",
)
def test_voxel_binner_parity_data_bin(tmp_path: Path):
    """Data.bin + nData.bin must be byte-identical to the C output."""
    from midas_transforms.bin_data.voxel_binner import bin_data_scanning

    inputs = GOLDEN / "inputs"
    for child in inputs.iterdir():
        shutil.copy(child, tmp_path / child.name)

    sp_path = tmp_path / "scan_positions.txt"
    scan_positions = np.array(
        [float(x) for x in sp_path.read_text().split()],
        dtype=np.float64,
    )
    n_scans = scan_positions.shape[0]

    bin_data_scanning(
        result_folder=tmp_path,
        n_scans=n_scans,
        scan_positions=scan_positions,
    )

    for fn in ("Data.bin", "nData.bin"):
        gold = GOLDEN / "golden" / fn
        if not gold.exists():
            pytest.skip(f"golden {fn} not present")
        assert _hash_file(tmp_path / fn) == _hash_file(gold), (
            f"{fn} bytes differ from golden"
        )
