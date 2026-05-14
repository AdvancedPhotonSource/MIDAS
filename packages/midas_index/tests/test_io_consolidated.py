"""Tests for the consolidated IndexBest_all.bin writer / reader (P5).

Verifies:
- Roundtrip writer→reader equality.
- Byte layout matches the pf_MIDAS.py:_read_indexbest parser semantics:
  ``int32 nVox`` + ``int32×nVox nSolArr`` + ``int64×nVox offArr`` + float64 records.
- Empty-voxel (n_sol=0) entries serialize correctly.
- offArr values are byte offsets from the file start, pointing at the
  first record of each voxel.
- split_records_by_voxel returns per-voxel arrays of the right shape.

These tests pin the on-disk format so when the per-voxel batch loop in
indexer.py is wired up next, the find_grains consumer reads the same
bytes the C scanner wrote.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_index.io.consolidated import (
    VALS_COLS,
    ConsolidatedReadResult,
    header_size_bytes,
    read_index_best_all,
    split_records_by_voxel,
    write_index_best_all,
)


def _make_records(n_solutions: int, *, seed: int = 0) -> np.ndarray:
    """Build a fixture (n_solutions, 16) float64 candidate block."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_solutions, VALS_COLS)).astype(np.float64)


class TestRoundtrip:
    def test_three_voxel_uniform(self, tmp_path):
        per_voxel = [_make_records(2, seed=1),
                     _make_records(3, seed=2),
                     _make_records(1, seed=3)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        result = read_index_best_all(p)
        assert result.n_voxels == 3
        np.testing.assert_array_equal(result.n_sol_arr, [2, 3, 1])
        np.testing.assert_array_equal(
            np.vstack(per_voxel), result.vals,
        )

    def test_with_empty_voxel(self, tmp_path):
        """A voxel with no solutions should serialize as 0 records but
        still have a valid offArr entry."""
        per_voxel = [_make_records(2, seed=10),
                     np.zeros((0, VALS_COLS), dtype=np.float64),
                     _make_records(1, seed=11)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        result = read_index_best_all(p)
        np.testing.assert_array_equal(result.n_sol_arr, [2, 0, 1])
        assert result.vals.shape == (3, VALS_COLS)

    def test_all_empty(self, tmp_path):
        per_voxel = [np.zeros((0, VALS_COLS), dtype=np.float64) for _ in range(4)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        result = read_index_best_all(p)
        assert result.n_voxels == 4
        np.testing.assert_array_equal(result.n_sol_arr, [0, 0, 0, 0])
        assert result.vals.shape == (0, VALS_COLS)


class TestByteLayout:
    def test_offarr_is_absolute_byte_offsets(self, tmp_path):
        """offArr[v] = header_size + cumulative_record_bytes_before_v."""
        per_voxel = [_make_records(2, seed=42),
                     _make_records(3, seed=43),
                     _make_records(1, seed=44)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        result = read_index_best_all(p)
        header = header_size_bytes(3)
        expected = np.array([
            header,
            header + 2 * VALS_COLS * 8,
            header + (2 + 3) * VALS_COLS * 8,
        ], dtype=np.int64)
        np.testing.assert_array_equal(result.off_arr, expected)

    def test_header_size_formula(self):
        """Pins the parser-side header offset to match pf_MIDAS.py:_read_indexbest."""
        assert header_size_bytes(0) == 4
        assert header_size_bytes(1) == 4 + 4 + 8
        assert header_size_bytes(10) == 4 + 40 + 80
        assert header_size_bytes(225) == 4 + 4 * 225 + 8 * 225

    def test_pf_midas_parser_roundtrip(self, tmp_path):
        """Match the exact byte interpretation in pf_MIDAS.py:_read_indexbest."""
        per_voxel = [_make_records(2, seed=100), _make_records(3, seed=200)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        # Replay the pf_MIDAS.py parser logic locally:
        with p.open("rb") as f:
            n_vox = np.frombuffer(f.read(4), dtype=np.int32)[0]
            n_sol = np.frombuffer(f.read(4 * n_vox), dtype=np.int32)
            off = np.frombuffer(f.read(8 * n_vox), dtype=np.int64)
            header = 4 + 4 * n_vox + 8 * n_vox
            all_vals = np.frombuffer(f.read(), dtype=np.double)
        # Reconstruct per-voxel:
        cursor = 0
        for v in range(n_vox):
            n = n_sol[v]
            block = all_vals[cursor * VALS_COLS:(cursor + n) * VALS_COLS].reshape(
                n, VALS_COLS,
            )
            np.testing.assert_allclose(block, per_voxel[v])
            cursor += n
        # offArr should match the byte-cursor view.
        assert off[0] == header
        assert off[1] == header + n_sol[0] * VALS_COLS * 8


class TestSplit:
    def test_split_recovers_per_voxel_arrays(self, tmp_path):
        per_voxel = [_make_records(2, seed=1),
                     np.zeros((0, VALS_COLS), dtype=np.float64),
                     _make_records(4, seed=2)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        result = read_index_best_all(p)
        recovered = split_records_by_voxel(result)
        assert len(recovered) == 3
        np.testing.assert_array_equal(recovered[0], per_voxel[0])
        assert recovered[1].shape == (0, VALS_COLS)
        np.testing.assert_array_equal(recovered[2], per_voxel[2])


class TestValidation:
    def test_wrong_record_shape_raises(self, tmp_path):
        per_voxel = [np.zeros((2, 15), dtype=np.float64)]   # wrong cols
        p = tmp_path / "IndexBest_all.bin"
        with pytest.raises(ValueError, match="16"):
            write_index_best_all(p, per_voxel)

    def test_short_file_raises(self, tmp_path):
        p = tmp_path / "too_short.bin"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="too short"):
            read_index_best_all(p)

    def test_truncated_records_raises(self, tmp_path):
        per_voxel = [_make_records(2, seed=0)]
        p = tmp_path / "IndexBest_all.bin"
        write_index_best_all(p, per_voxel)
        raw = p.read_bytes()
        # Chop off the last 8 bytes (truncate one float64 in the records).
        p.write_bytes(raw[:-8])
        with pytest.raises(ValueError, match="byte count mismatch"):
            read_index_best_all(p)
