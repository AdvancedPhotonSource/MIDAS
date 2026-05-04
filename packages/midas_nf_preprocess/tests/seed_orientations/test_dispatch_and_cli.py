"""Tests for the high-level dispatch + CLI entry point."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_preprocess.cli import main as umbrella_main
from midas_nf_preprocess.seed_orientations import generate_seeds, read_seeds_csv
from midas_nf_preprocess.seed_orientations.cli import main as so_main


# ----- Helpers --------------------------------------------------------------


def _write_cache_csv(seed_dir: Path, lookup_type: str, n: int = 4) -> Path:
    seed_dir.mkdir(parents=True, exist_ok=True)
    quats = np.tile([[1.0, 0.0, 0.0, 0.0]], (n, 1))
    p = seed_dir / f"seed_{lookup_type}.csv"
    np.savetxt(p, quats, delimiter=",")
    return p


def _write_grains_file(path: Path) -> None:
    with open(path, "w") as f:
        f.write("% header\n")
        f.write(
            "1 1 0 0 0 1 0 0 0 1 0 0 0 3.6 3.6 3.6 90 90 90 0 0 0 0\n"
        )


# ----- generate_seeds dispatcher -------------------------------------------


def test_dispatch_cache(tmp_path):
    _write_cache_csv(tmp_path, "cubic_high", n=3)
    out = generate_seeds(method="cache", space_group=225, seed_dir=tmp_path)
    assert out.shape == (3, 4)


def test_dispatch_cache_via_crystal_system(tmp_path):
    _write_cache_csv(tmp_path, "cubic_high", n=3)
    out = generate_seeds(method="cache", crystal_system="cubic", seed_dir=tmp_path)
    assert out.shape == (3, 4)


def test_dispatch_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        generate_seeds(method="banana", space_group=225)


def test_dispatch_both_sg_and_crystal_raises():
    with pytest.raises(ValueError, match="not both"):
        generate_seeds(
            method="cache", space_group=225, crystal_system="cubic"
        )


def test_dispatch_no_sg_raises_for_cache():
    with pytest.raises(ValueError, match="required"):
        generate_seeds(method="cache")


def test_dispatch_from_grains_requires_file():
    with pytest.raises(ValueError, match="grains_file"):
        generate_seeds(method="from_grains")


def test_dispatch_from_grains(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p)
    out = generate_seeds(method="from_grains", grains_file=p)
    assert out.shape == (1, 4)


@pytest.mark.slow
def test_dispatch_from_scratch(tmp_path):
    out = generate_seeds(
        method="from_scratch",
        space_group=225,
        resolution_deg=10.0,  # coarse for speed
        seed=0,
    )
    assert out.shape[0] > 50
    assert out.shape[1] == 4


# ----- CLI: standalone main --------------------------------------------------


def test_cli_cache(tmp_path):
    _write_cache_csv(tmp_path / "cache", "cubic_high", n=5)
    out = tmp_path / "seeds.csv"
    rc = so_main(
        [
            "--method", "cache",
            "--space-group", "225",
            "--seed-dir", str(tmp_path / "cache"),
            "--output", str(out),
        ]
    )
    assert rc == 0
    seeds = read_seeds_csv(out)
    assert seeds.shape == (5, 4)


def test_cli_from_grains(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p)
    out = tmp_path / "seeds.csv"
    rc = so_main(
        [
            "--method", "from-grains",
            "--grains-file", str(p),
            "--output", str(out),
        ]
    )
    assert rc == 0
    # Output uses the 11-column format.
    line = out.read_text().strip()
    assert len(line.split(",")) == 11


def test_cli_missing_grains_file_raises(tmp_path):
    out = tmp_path / "seeds.csv"
    with pytest.raises(SystemExit, match="grains-file"):
        so_main(["--method", "from-grains", "--output", str(out)])


def test_cli_missing_sg_for_cache_raises(tmp_path):
    out = tmp_path / "seeds.csv"
    with pytest.raises(SystemExit, match="--space-group"):
        so_main(
            [
                "--method", "cache",
                "--seed-dir", str(tmp_path),
                "--output", str(out),
            ]
        )


def test_cli_uses_crystal_system(tmp_path):
    _write_cache_csv(tmp_path / "cache", "cubic_high", n=2)
    out = tmp_path / "seeds.csv"
    rc = so_main(
        [
            "--method", "cache",
            "--crystal-system", "cubic",
            "--seed-dir", str(tmp_path / "cache"),
            "--output", str(out),
        ]
    )
    assert rc == 0
    assert read_seeds_csv(out).shape == (2, 4)


# ----- CLI: through umbrella -------------------------------------------------


def test_umbrella_seed_orientations_cache(tmp_path):
    _write_cache_csv(tmp_path / "cache", "cubic_high", n=2)
    out = tmp_path / "seeds.csv"
    rc = umbrella_main(
        [
            "seed-orientations",
            "--method", "cache",
            "--space-group", "225",
            "--seed-dir", str(tmp_path / "cache"),
            "--output", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()


def test_umbrella_seed_orientations_from_grains(tmp_path):
    p = tmp_path / "Grains.csv"
    _write_grains_file(p)
    out = tmp_path / "seeds.csv"
    rc = umbrella_main(
        [
            "seed-orientations",
            "--method", "from-grains",
            "--grains-file", str(p),
            "--output", str(out),
        ]
    )
    assert rc == 0


@pytest.mark.slow
def test_umbrella_seed_orientations_from_scratch(tmp_path):
    out = tmp_path / "seeds.csv"
    rc = umbrella_main(
        [
            "seed-orientations",
            "--method", "from-scratch",
            "--crystal-system", "cubic",
            "--resolution-deg", "10.0",
            "--output", str(out),
        ]
    )
    assert rc == 0
    seeds = read_seeds_csv(out)
    assert seeds.shape[0] > 0
