"""Tests for from_grains: parsing FF Grains.csv into seed orientations."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from midas_nf_preprocess.seed_orientations import (
    GrainOrientation,
    read_grains_orientations,
    write_seeds_with_lattice_csv,
)
from midas_nf_preprocess.diffr_spots import quat_to_orient_matrix


def _write_grains_file(path: Path, rows: list[dict]) -> None:
    """Build a synthetic FF Grains.csv that satisfies the C parser's column layout."""
    with open(path, "w") as f:
        f.write("% header\n")
        for row in rows:
            om = row["om"]
            latc = row["latc"]
            grain_id = row["grain_id"]
            f.write(
                f"{grain_id} "
                + " ".join(f"{v:.6f}" for v in om)
                + " 0 0 0 "  # 3 dummies (positions)
                + " ".join(f"{v:.6f}" for v in latc)
                + " 0 0 0 0\n"  # 4 trailing dummies
            )


def _identity_grain(gid: int = 1):
    return {
        "grain_id": gid,
        "om": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "latc": [3.6, 3.6, 3.6, 90.0, 90.0, 90.0],
    }


def test_read_grains_basic(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p, [_identity_grain(7)])
    grains = read_grains_orientations(p)
    assert len(grains) == 1
    g = grains[0]
    assert isinstance(g, GrainOrientation)
    assert g.grain_id == 7
    assert g.lattice == (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    # Identity OM -> identity quat
    expected_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(g.quat, expected_quat, atol=1e-12)


def test_read_grains_quat_inverse_consistent(tmp_path):
    """OM -> quat -> OM should round-trip to the original OM."""
    p = tmp_path / "Grains.csv"
    om = [
        0.0, -1.0,  0.0,
        1.0,  0.0,  0.0,
        0.0,  0.0,  1.0,
    ]  # 90deg rotation about +z
    rows = [{"grain_id": 1, "om": om, "latc": [3.6] * 3 + [90.0] * 3}]
    _write_grains_file(p, rows)
    grains = read_grains_orientations(p)
    om_in = grains[0].orient_matrix
    om_back = quat_to_orient_matrix(grains[0].quat.unsqueeze(0)).squeeze(0)
    assert torch.allclose(om_in, om_back, atol=1e-10)


def test_read_grains_skips_comments(tmp_path):
    p = tmp_path / "Grains.csv"
    with open(p, "w") as f:
        f.write("% comment\n")
        f.write("% another comment line\n")
    p_data = tmp_path / "Grains2.csv"
    _write_grains_file(p_data, [_identity_grain()])
    grains = read_grains_orientations(p_data)
    assert len(grains) == 1


def test_read_grains_skips_short_rows(tmp_path):
    p = tmp_path / "Grains.csv"
    with open(p, "w") as f:
        f.write("% header\n")
        f.write("1 2 3\n")  # too few columns
        f.write(" ".join(["1"] + ["0"] * 8 + ["0"] * 9) + "\n")  # exactly 18 cols < 19
    grains = read_grains_orientations(p)
    assert grains == []


def test_read_grains_empty_file(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text("% only header\n")
    assert read_grains_orientations(p) == []


def test_write_seeds_with_lattice(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p, [_identity_grain(1), _identity_grain(2)])
    grains = read_grains_orientations(p)
    out = tmp_path / "seeds.csv"
    n = write_seeds_with_lattice_csv(grains, out)
    assert n == 2
    # Each row should have 11 comma-separated fields.
    rows = out.read_text().strip().split("\n")
    assert len(rows) == 2
    for row in rows:
        cols = row.split(",")
        assert len(cols) == 11


def test_write_seeds_with_lattice_grain_id_preserved(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p, [_identity_grain(42)])
    grains = read_grains_orientations(p)
    out = tmp_path / "seeds.csv"
    write_seeds_with_lattice_csv(grains, out)
    last_field = out.read_text().strip().split(",")[-1]
    assert last_field == "42"
