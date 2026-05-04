"""Tests for ProcessParams parsing and defaults."""

from __future__ import annotations

import pytest

from midas_nf_preprocess.process_images import ProcessParams


def test_defaults_match_c():
    p = ProcessParams()
    # C defaults from L644-L649
    assert p.raw_start_nr == 0
    assert p.nr_pixels == 2048
    assert p.nr_pixels_y == 2048
    assert p.nr_pixels_z == 2048
    assert p.blanket_subtraction == 0
    assert p.mean_filt_radius == 1
    assert p.do_log_filter == 1
    assert p.log_mask_radius == 4
    assert p.sigma == 1.0
    assert p.n_distances == 1


def test_pixel_resolution_y_only():
    p = ProcessParams(nr_pixels=0, nr_pixels_y=512)
    assert p.nr_pixels_y == 512
    assert p.nr_pixels_z == 512  # falls back to Y


def test_pixel_resolution_z_only():
    p = ProcessParams(nr_pixels=0, nr_pixels_z=400)
    assert p.nr_pixels_z == 400
    assert p.nr_pixels_y == 400


def test_pixel_resolution_both():
    p = ProcessParams(nr_pixels_y=300, nr_pixels_z=200)
    assert p.nr_pixels_y == 300
    assert p.nr_pixels_z == 200


def test_deblur_forces_write_fin_image():
    p = ProcessParams(do_deblur=1, write_fin_image=0)
    assert p.write_fin_image == 1


def test_output_dir_defaults_to_data_dir():
    p = ProcessParams(data_directory="/tmp/data")
    assert p.output_directory == "/tmp/data"


def test_output_dir_explicit():
    p = ProcessParams(data_directory="/tmp/data", output_directory="/tmp/out")
    assert p.output_directory == "/tmp/out"


def test_from_paramfile(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text(
        "RawStartNr 100\n"
        "DataDirectory /tmp/foo\n"
        "OutputDirectory /tmp/bar\n"
        "NrPixels 1024\n"
        "NrPixelsY 512\n"
        "NrPixelsZ 256\n"
        "WFImages 360\n"
        "NrFilesPerDistance 720\n"
        "OrigFileName scan_001\n"
        "ReducedFileName proc_001\n"
        "extOrig tif\n"
        "extReduced bin\n"
        "BlanketSubtraction 5\n"
        "MedFiltRadius 2\n"
        "DoLoGFilter 1\n"
        "LoGMaskRadius 3\n"
        "GaussFiltRadius 1.5\n"
        "WriteFinImage 0\n"
        "Deblur 0\n"
        "nDistances 4\n"
        "WriteLegacyBin 1\n"
        "SoftTemperature 2.0\n"
    )
    p = ProcessParams.from_paramfile(pf)
    assert p.raw_start_nr == 100
    assert p.data_directory == "/tmp/foo"
    assert p.output_directory == "/tmp/bar"
    assert p.nr_pixels == 1024
    assert p.nr_pixels_y == 512
    assert p.nr_pixels_z == 256
    assert p.wf_images == 360
    assert p.nr_files_per_distance == 720
    assert p.orig_filename == "scan_001"
    assert p.reduced_filename == "proc_001"
    assert p.ext_orig == "tif"
    assert p.ext_reduced == "bin"
    assert p.blanket_subtraction == 5
    assert p.mean_filt_radius == 2
    assert p.do_log_filter == 1
    assert p.log_mask_radius == 3
    assert p.sigma == 1.5
    assert p.write_fin_image == 0
    assert p.do_deblur == 0
    assert p.n_distances == 4
    assert p.write_legacy_bin == 1
    assert p.soft_temperature == 2.0


def test_from_paramfile_ignores_unknown_keys(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text("RawStartNr 5\nUnknownKey 99\nNonsense\n# comment\n")
    p = ProcessParams.from_paramfile(pf)
    assert p.raw_start_nr == 5


def test_from_paramfile_minimal(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text("NrPixels 256\n")
    p = ProcessParams.from_paramfile(pf)
    assert p.nr_pixels == 256
    assert p.nr_pixels_y == 256
    assert p.nr_pixels_z == 256


def test_with_overrides_runs_post_init():
    p = ProcessParams(nr_pixels=128).with_overrides(nr_pixels_y=64, nr_pixels_z=0)
    # After override Z=0 should fall back to Y=64
    assert p.nr_pixels_y == 64
    assert p.nr_pixels_z == 64


def test_soft_temperature_default_auto():
    p = ProcessParams()
    assert p.soft_temperature == "auto"


def test_from_paramfile_soft_temperature_auto(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text("SoftTemperature auto\n")
    p = ProcessParams.from_paramfile(pf)
    assert p.soft_temperature == "auto"


def test_from_paramfile_soft_temperature_numeric(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text("SoftTemperature 5.5\n")
    p = ProcessParams.from_paramfile(pf)
    assert p.soft_temperature == 5.5


def test_from_paramfile_soft_temperature_case_insensitive(tmp_path):
    pf = tmp_path / "ps.txt"
    pf.write_text("SoftTemperature AUTO\n")
    p = ProcessParams.from_paramfile(pf)
    assert p.soft_temperature == "auto"
