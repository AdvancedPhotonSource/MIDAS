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


# The parity tests are gated on ``slow`` (pyproject markers) because the
# per-voxel Python runtime + the ~1 GB nData.bin mmap make it too heavy
# for routine CI. Production runs target chiltepin (64 cores, ample
# RAM) per project handoff. Enable locally with ``-m slow``.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _fixture_complete(),
        reason=(
            "Golden fixture incomplete. Regenerate locally:\n"
            "  1. python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup --doTomo 0\n"
            "  2. python packages/midas_index/tests/data/scanning_5grain_golden/build.py\n"
            "Data.bin / nData.bin are gitignored (too big for git); the build "
            "script copies them locally from FF_HEDM/Example/pfhedm_test/."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


VOXEL_SHARDS = 25       # 225 voxels / 25 = 9 voxels per smoke shard


def _run_python_indexer(out_path: Path, *, voxel_block_nr: int = 0) -> None:
    """Run midas-index in scan-aware C-parity mode on ONE voxel shard.

    Splitting into ``VOXEL_SHARDS`` chunks keeps peak memory bounded
    on the smoke run; the full 225-voxel run with the ~1 GB bin index
    OOMs on a 16 GB Mac. The C reference (IndexerScanningOMP) is
    inherently per-voxel-independent, so shard-by-shard parity is the
    same gate as full-grid parity.

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
        ind.params.friedel_symmetric_scan_filter = False
        ind.params.multi_solution_output = True
        ind.params.OutputFolder = str(DATA_DIR)
        ind.load_observations(cwd=DATA_DIR)
        scan_positions = np.loadtxt(DATA_DIR / "positions.csv")
        ind.run_scanning(
            scan_positions=scan_positions,
            out_path=out_path,
            num_procs=1,
            seed_group_size=4,
            voxel_block_nr=voxel_block_nr,
            voxel_n_blocks=VOXEL_SHARDS,
        )
    finally:
        os.chdir(cwd0)


def _slice_golden_to_shard(voxel_block_nr: int) -> tuple[int, int, np.ndarray]:
    """Return (v_start, v_end, voxel_records[v_start:v_end]) from the golden."""
    c = read_index_best_all(GOLDEN_PATH)
    n_vox = c.n_voxels
    block_size = (n_vox + VOXEL_SHARDS - 1) // VOXEL_SHARDS
    v_start = voxel_block_nr * block_size
    v_end = min(v_start + block_size, n_vox)
    records = split_records_by_voxel(c)
    return v_start, v_end, records[v_start:v_end]


def _slice_python_to_shard(out_path: Path,
                           voxel_block_nr: int) -> list[np.ndarray]:
    """Return Python per-voxel records for the shard (drops empty
    voxels outside the shard's range, which were never run)."""
    py = read_index_best_all(out_path)
    n_vox = py.n_voxels
    block_size = (n_vox + VOXEL_SHARDS - 1) // VOXEL_SHARDS
    v_start = voxel_block_nr * block_size
    v_end = min(v_start + block_size, n_vox)
    records = split_records_by_voxel(py)
    return records[v_start:v_end]


# ---------------------------------------------------------------------------
# Parity gates
# ---------------------------------------------------------------------------


def test_smoke_shard0_solution_count_matches(tmp_path: Path):
    """First voxel shard: per-voxel solution counts agree with C."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out, voxel_block_nr=0)
    py_blocks = _slice_python_to_shard(out, voxel_block_nr=0)
    _, _, c_blocks = _slice_golden_to_shard(voxel_block_nr=0)
    assert len(py_blocks) == len(c_blocks), (
        f"shard length py={len(py_blocks)} vs c={len(c_blocks)}"
    )
    counts_py = np.array([b.shape[0] for b in py_blocks])
    counts_c = np.array([b.shape[0] for b in c_blocks])
    np.testing.assert_array_equal(counts_py, counts_c)


def test_smoke_shard0_orientation_matrices_match(tmp_path: Path):
    """First voxel shard: per-voxel OM (cols 2-10) within 1e-12."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out, voxel_block_nr=0)
    py_blocks = _slice_python_to_shard(out, voxel_block_nr=0)
    _, _, c_blocks = _slice_golden_to_shard(voxel_block_nr=0)
    for v, (py_b, c_b) in enumerate(zip(py_blocks, c_blocks)):
        if c_b.shape[0] == 0:
            assert py_b.shape[0] == 0, (
                f"voxel {v}: C empty, Python has {py_b.shape[0]} solutions"
            )
            continue
        assert py_b.shape == c_b.shape
        np.testing.assert_allclose(
            py_b[:, 2:11], c_b[:, 2:11], atol=1e-12, rtol=0.0,
            err_msg=f"voxel {v}: OM mismatch beyond 1e-12",
        )


def test_smoke_shard0_completeness_counts_bit_exact(tmp_path: Path):
    """First voxel shard: cols 14 (nExpected) + 15 (nMatches) integer-equal."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out, voxel_block_nr=0)
    py_blocks = _slice_python_to_shard(out, voxel_block_nr=0)
    _, _, c_blocks = _slice_golden_to_shard(voxel_block_nr=0)
    for v, (py_b, c_b) in enumerate(zip(py_blocks, c_blocks)):
        if c_b.shape[0] == 0:
            continue
        np.testing.assert_array_equal(
            py_b[:, 14:16].astype(np.int64),
            c_b[:, 14:16].astype(np.int64),
            err_msg=f"voxel {v}: completeness counts diverged",
        )


def test_smoke_shard0_positions_within_picometer(tmp_path: Path):
    """First voxel shard: position cols (11-13) within 1e-9 µm."""
    out = tmp_path / "IndexBest_all.bin"
    _run_python_indexer(out, voxel_block_nr=0)
    py_blocks = _slice_python_to_shard(out, voxel_block_nr=0)
    _, _, c_blocks = _slice_golden_to_shard(voxel_block_nr=0)
    for v, (py_b, c_b) in enumerate(zip(py_blocks, c_blocks)):
        if c_b.shape[0] == 0:
            continue
        np.testing.assert_allclose(
            py_b[:, 11:14], c_b[:, 11:14], atol=1e-9, rtol=0.0,
            err_msg=f"voxel {v}: position mismatch beyond 1e-9 µm",
        )
