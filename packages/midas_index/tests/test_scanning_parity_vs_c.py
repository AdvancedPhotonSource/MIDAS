"""P5c parity gate: Indexer.run_scanning vs IndexerScanningOMP frozen output.

Loads the frozen C ``IndexBest_all.bin`` produced by running
``IndexerScanningOMP`` on the 5-grain × 15-scan synthetic fixture under
``tests/data/scanning_5grain_golden/``. Re-runs the Python
``Indexer.run_scanning`` on the same Spots.bin / paramstest /
positions.csv inputs with the C-parity options (single-sided Friedel
filter, OMP_NUM_THREADS=1 for deterministic ordering on the C side).

Pass criterion (per plan §12b + locked tolerances from §16):
- Bit-exact byte layout (header_size_bytes formula, nVoxels, byte
  offset table) — see ``test_io_consolidated.py`` for the format pins.
- Per-voxel solution count matches exactly.
- For each voxel with at least one solution, the orientation matrices
  agree at fp64 ``1e-12`` and positions at ``1e-9`` µm (relative ~
  picometer over a ~100 µm sample). Completeness counts (cols 14, 15)
  match bit-exactly.

When the golden fixture is absent on disk, the test skips with a clear
message — checked into the repo as data, generated via
``tests/data/scanning_5grain_golden/build.py`` (driver script).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_index.io.consolidated import (
    VALS_COLS,
    read_index_best_all,
    split_records_by_voxel,
)
from midas_index.indexer import Indexer

DATA_DIR = Path(__file__).parent / "data" / "scanning_5grain_golden"
GOLDEN_PATH = DATA_DIR / "golden" / "IndexBest_all.bin"
DATA_BIN_PATH = DATA_DIR / "Data.bin"
NDATA_BIN_PATH = DATA_DIR / "nData.bin"


# ---------------------------------------------------------------------------
# Skip when the fixture isn't on disk
# ---------------------------------------------------------------------------


def _fixture_complete() -> bool:
    return (
        GOLDEN_PATH.exists()
        and DATA_BIN_PATH.exists()
        and NDATA_BIN_PATH.exists()
    )


pytestmark = pytest.mark.skipif(
    not _fixture_complete(),
    reason=(
        "Golden fixture incomplete. Regenerate locally:\n"
        "  1. python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup --doTomo 0\n"
        "  2. python packages/midas_index/tests/data/scanning_5grain_golden/build.py\n"
        "Data.bin / nData.bin are gitignored (too big for git); the build "
        "script copies them locally from FF_HEDM/Example/pfhedm_test/."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_python_indexer(out_path: Path) -> None:
    """Run midas-index in scan-aware C-parity mode on the frozen fixture.

    Reads the frozen Data.bin / nData.bin (gitignored locally; copied
    by build.py from the C run) via the normal ``read_bins`` path.
    Forces ``friedel_symmetric_scan_filter=False`` so the scan filter
    matches C ``IndexerScanningOMP`` byte-for-byte (the production
    Friedel-symmetric form is wider and would alter completeness).
    """
    import os
    cwd0 = Path.cwd()
    os.chdir(DATA_DIR)            # load_observations uses relative paths for hkls.csv etc.
    try:
        ind = Indexer.from_param_file(DATA_DIR / "paramstest.txt", device="cpu",
                                      dtype="float64")
        # Single-sided filter form to match C IndexerScanningOMP (the
        # production default is Friedel-symmetric — see plan §1b).
        ind.params.friedel_symmetric_scan_filter = False
        ind.params.multi_solution_output = True
        # ``paramstest`` has ``OutputFolder = .`` etc.; point them at the
        # fixture so sibling-file lookups (load_observations) resolve.
        ind.params.OutputFolder = str(DATA_DIR)
        ind.load_observations(cwd=DATA_DIR)
        scan_positions = np.loadtxt(DATA_DIR / "positions.csv")
        ind.run_scanning(
            scan_positions=scan_positions,
            out_path=out_path,
            num_procs=1,
            seed_group_size=4,
        )
    finally:
        os.chdir(cwd0)


# ---------------------------------------------------------------------------
# Parity gates
# ---------------------------------------------------------------------------


def test_header_bit_exact(tmp_path: Path):
    """Header (nVoxels + nSolArr + offArr) must be byte-identical."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out)
    py = read_index_best_all(out)
    c = read_index_best_all(GOLDEN_PATH)
    assert py.n_voxels == c.n_voxels
    np.testing.assert_array_equal(py.n_sol_arr, c.n_sol_arr)
    np.testing.assert_array_equal(py.off_arr, c.off_arr)


def test_orientation_matrices_match(tmp_path: Path):
    """Per-voxel OM columns (cols 2-10) within 1e-12."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out)
    py = split_records_by_voxel(read_index_best_all(out))
    c = split_records_by_voxel(read_index_best_all(GOLDEN_PATH))
    assert len(py) == len(c)
    for v, (py_block, c_block) in enumerate(zip(py, c)):
        if c_block.shape[0] == 0:
            assert py_block.shape[0] == 0, (
                f"voxel {v}: C has 0 solutions, Python has {py_block.shape[0]}"
            )
            continue
        assert py_block.shape == c_block.shape, (
            f"voxel {v}: shape py={py_block.shape} vs c={c_block.shape}"
        )
        np.testing.assert_allclose(
            py_block[:, 2:11], c_block[:, 2:11], atol=1e-12, rtol=0.0,
            err_msg=f"voxel {v}: OM mismatch beyond 1e-12",
        )


def test_completeness_counts_bit_exact(tmp_path: Path):
    """Cols 14 (nExpected) + 15 (nMatches) must be integer-equal."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out)
    py = split_records_by_voxel(read_index_best_all(out))
    c = split_records_by_voxel(read_index_best_all(GOLDEN_PATH))
    for v, (py_block, c_block) in enumerate(zip(py, c)):
        if c_block.shape[0] == 0:
            continue
        np.testing.assert_array_equal(
            py_block[:, 14:16].astype(np.int64),
            c_block[:, 14:16].astype(np.int64),
            err_msg=f"voxel {v}: completeness counts diverged",
        )


def test_positions_within_picometer(tmp_path: Path):
    """Per-voxel position cols (11-13) within 1e-9 µm."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out)
    py = split_records_by_voxel(read_index_best_all(out))
    c = split_records_by_voxel(read_index_best_all(GOLDEN_PATH))
    for v, (py_block, c_block) in enumerate(zip(py, c)):
        if c_block.shape[0] == 0:
            continue
        np.testing.assert_allclose(
            py_block[:, 11:14], c_block[:, 11:14], atol=1e-9, rtol=0.0,
            err_msg=f"voxel {v}: position mismatch beyond 1e-9 µm",
        )
