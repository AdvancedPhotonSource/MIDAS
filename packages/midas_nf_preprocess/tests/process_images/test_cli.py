"""Tests for the midas-nf-process-images CLI."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from midas_nf_preprocess.process_images.cli import main


def _write_tiny_dataset(tmp_path, n_frames=3, H=32, W=32, sigma=2.0):
    """Write a single layer's worth of TIFFs and return the param-file path."""
    z = np.arange(H).reshape(-1, 1)
    y = np.arange(W).reshape(1, -1)
    for j in range(n_frames):
        cz, cy = 10 + j * 3, 12 + j * 2
        blob = 800 * np.exp(-((z - cz) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        arr = (100 + blob).astype(np.uint16)
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
        f"GaussFiltRadius 2.0\n"
        f"MedFiltRadius 1\n"
    )
    return pf


def test_cli_runs_one_layer(tmp_path, capsys):
    pf = _write_tiny_dataset(tmp_path)
    rc = main([str(pf), "1", "--device", "cpu"])
    assert rc == 0
    assert (tmp_path / "SpotsInfo.bin").exists()
    captured = capsys.readouterr()
    assert "Wrote" in captured.out
    assert "SpotsInfo.bin" in captured.out


def test_cli_all_layers(tmp_path):
    pf = _write_tiny_dataset(tmp_path)
    rc = main([str(pf), "1", "--device", "cpu", "--all-layers"])
    assert rc == 0
    assert (tmp_path / "SpotsInfo.bin").exists()


def test_cli_custom_output_path(tmp_path):
    pf = _write_tiny_dataset(tmp_path)
    custom = tmp_path / "subdir" / "Custom.bin"
    rc = main([str(pf), "1", "--device", "cpu", "--output", str(custom)])
    assert rc == 0
    assert custom.exists()
    assert not (tmp_path / "SpotsInfo.bin").exists()


def test_cli_dtype_fp32(tmp_path):
    pf = _write_tiny_dataset(tmp_path)
    rc = main([str(pf), "1", "--device", "cpu", "--dtype", "fp32"])
    assert rc == 0


def test_cli_n_cpus(tmp_path):
    pf = _write_tiny_dataset(tmp_path)
    rc = main([str(pf), "1", "--device", "cpu", "--n-cpus", "2"])
    assert rc == 0


def test_cli_missing_args_exits():
    with pytest.raises(SystemExit):
        main([])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "midas-nf-preprocess" in captured.out
