"""Tests for seed_orientations.io: CSV reader/writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_preprocess.seed_orientations import (
    read_seeds_csv,
    write_seeds_csv,
)


def test_write_then_read_roundtrip(tmp_path):
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7071067811865476, 0.0, 0.7071067811865476, 0.0],
        ],
        dtype=torch.float64,
    )
    path = tmp_path / "seeds.csv"
    write_seeds_csv(quats, path)
    back = read_seeds_csv(path)
    assert back.shape == quats.shape
    assert torch.allclose(back, quats, atol=1e-6)


def test_write_creates_parent_dir(tmp_path):
    quats = torch.zeros(2, 4)
    path = tmp_path / "subdir" / "seeds.csv"
    write_seeds_csv(quats, path)
    assert path.exists()


def test_write_wrong_shape_raises():
    with pytest.raises(ValueError, match="\\(N, 4\\)"):
        write_seeds_csv(torch.zeros(3, 3), Path("/tmp/foo.csv"))


def test_read_single_row(tmp_path):
    """A 1-row file should still produce a (1, 4) tensor."""
    path = tmp_path / "single.csv"
    path.write_text("1.0,0.0,0.0,0.0\n")
    out = read_seeds_csv(path)
    assert out.shape == (1, 4)


def test_read_wrong_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="4 columns"):
        read_seeds_csv(path)


def test_csv_dtype_is_float64(tmp_path):
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    path = tmp_path / "seeds.csv"
    write_seeds_csv(quats, path)
    back = read_seeds_csv(path)
    assert back.dtype == torch.float64
