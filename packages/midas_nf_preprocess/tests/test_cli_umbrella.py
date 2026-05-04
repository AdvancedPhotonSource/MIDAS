"""Tests for the umbrella ``midas-nf-preprocess`` CLI."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from midas_nf_preprocess.cli import main as umbrella_main
from midas_nf_preprocess.hex_grid import read_grid_txt


def test_umbrella_no_args_exits():
    with pytest.raises(SystemExit):
        umbrella_main([])


def test_umbrella_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        umbrella_main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "midas-nf-preprocess" in captured.out


def test_umbrella_hex_grid_subcommand(tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text(
        f"GridSize 1.0\nRsample 4.0\nDataDirectory {tmp_path}\n"
    )
    rc = umbrella_main(["hex-grid", str(pf), "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "grid.txt").exists()
    grid = read_grid_txt(tmp_path / "grid.txt")
    assert grid.shape[0] > 0


def test_umbrella_tomo_filter_subcommand(tmp_path):
    # Build a grid first.
    pf = tmp_path / "p.txt"
    pf.write_text(f"GridSize 1.0\nRsample 4.0\nDataDirectory {tmp_path}\n")
    umbrella_main(["hex-grid", str(pf), "--device", "cpu"])
    out = tmp_path / "filtered.txt"
    rc = umbrella_main(
        [
            "tomo-filter",
            str(tmp_path / "grid.txt"),
            str(out),
            "--bbox",
            "-1.0",
            "1.0",
            "-1.0",
            "1.0",
        ]
    )
    assert rc == 0
    assert out.exists()


def test_umbrella_process_images_subcommand(tmp_path):
    """Smoke test: write a tiny TIFF stack and run process-images via the umbrella."""
    H, W = 16, 16
    n_frames = 3
    for j in range(n_frames):
        arr = np.zeros((H, W), dtype=np.uint16)
        arr[5 + j, 6 + j] = 1000
        tifffile.imwrite(tmp_path / f"img_{j:06d}.tif", arr)
    pf = tmp_path / "params.txt"
    pf.write_text(
        f"DataDirectory {tmp_path}\n"
        f"OutputDirectory {tmp_path}\n"
        f"OrigFileName img\n"
        f"ReducedFileName proc\n"
        f"extOrig tif\n"
        f"extReduced bin\n"
        f"NrPixelsY {W}\n"
        f"NrPixelsZ {H}\n"
        f"NrFilesPerDistance {n_frames}\n"
        f"nDistances 1\n"
        f"LoGMaskRadius 4\n"
        f"GaussFiltRadius 1.0\n"
        f"MedFiltRadius 1\n"
    )
    rc = umbrella_main(["process-images", str(pf), "1", "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "SpotsInfo.bin").exists()
