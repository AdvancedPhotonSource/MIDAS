"""Tests for from_cache: loading pre-computed seed-orientation files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_preprocess.seed_orientations import (
    DEFAULT_SEED_DIR,
    SeedCacheNotFound,
    load_seeds_for_lookup_type,
    load_seeds_for_space_group,
)


# ----- Helpers --------------------------------------------------------------


def _write_csv(path: Path, quats: np.ndarray) -> None:
    np.savetxt(path, quats, delimiter=",")


def _write_master_lookup(seed_dir: Path, lookup_type: str, master: np.ndarray, indices: np.ndarray) -> None:
    master.astype(np.float64).tofile(seed_dir / "orientations_master.bin")
    indices.astype(np.int32).tofile(seed_dir / f"lookup_{lookup_type}.bin")


# ----- CSV path -------------------------------------------------------------


def test_load_from_csv(tmp_path):
    quats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.from_numpy(quats))


def test_load_from_csv_matches_space_group_route(tmp_path):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out_lt = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    out_sg = load_seeds_for_space_group(225, seed_dir=tmp_path)
    assert torch.equal(out_lt, out_sg)


def test_load_from_csv_dtype_float32(tmp_path):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    out = load_seeds_for_lookup_type(
        "cubic_high", seed_dir=tmp_path, dtype="fp32"
    )
    assert out.dtype == torch.float32


# ----- Master + lookup binary path ------------------------------------------


def test_load_from_master_lookup(tmp_path):
    """When CSV is absent, fall back to orientations_master.bin + lookup_*.bin."""
    master = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    indices = np.array([0, 2, 3], dtype=np.int32)  # pick rows 0, 2, 3
    _write_master_lookup(tmp_path, "cubic_high", master, indices)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    expected = master[indices]
    assert out.shape == (3, 4)
    assert torch.allclose(out, torch.from_numpy(expected))


def test_csv_takes_precedence_over_master(tmp_path):
    """If both forms are present, the CSV should win (it's faster to load)."""
    csv_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", csv_quats)
    # Master with different content
    master = np.array([[0.5, 0.5, 0.5, 0.5]])
    indices = np.array([0], dtype=np.int32)
    _write_master_lookup(tmp_path, "cubic_high", master, indices)
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)
    assert torch.allclose(out, torch.from_numpy(csv_quats))


# ----- Errors ---------------------------------------------------------------


def test_missing_seed_dir_raises(tmp_path):
    with pytest.raises(SeedCacheNotFound):
        load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path / "missing")


def test_missing_files_in_seed_dir_raises(tmp_path):
    with pytest.raises(SeedCacheNotFound, match="Run NF_HEDM"):
        load_seeds_for_lookup_type("cubic_high", seed_dir=tmp_path)


def test_env_var_seed_dir(tmp_path, monkeypatch):
    quats = np.array([[1.0, 0.0, 0.0, 0.0]])
    _write_csv(tmp_path / "seed_cubic_high.csv", quats)
    monkeypatch.setenv("MIDAS_NF_SEED_DIR", str(tmp_path))
    out = load_seeds_for_lookup_type("cubic_high", seed_dir=None)
    assert out.shape == (1, 4)


# ----- Real cached files (skipped when not present) -------------------------


@pytest.mark.skipif(
    not (DEFAULT_SEED_DIR / "seed_cubic_high.csv").resolve().exists(),
    reason="bundled seed cache not present",
)
def test_load_real_cubic_high_cache():
    out = load_seeds_for_lookup_type("cubic_high")
    # The bundled cubicSeed.txt has ~243k entries.
    assert out.shape[0] > 100_000
    assert out.shape[1] == 4
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
