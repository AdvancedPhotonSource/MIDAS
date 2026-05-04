"""Tests for TIFF stack I/O and frame path generation."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch

from midas_nf_preprocess.process_images import ProcessParams, frame_paths, from_tensor, load_tiff_stack


def test_frame_paths_basic_layer1():
    p = ProcessParams(
        data_directory="/data",
        orig_filename="scan",
        ext_orig="tif",
        raw_start_nr=10,
        wf_images=0,
        nr_files_per_distance=3,
    )
    paths = frame_paths(p, layer_nr=1)
    # layer 1: base_idx=0, start=10+0=10 -> 10, 11, 12
    assert paths == [
        "/data/scan_000010.tif",
        "/data/scan_000011.tif",
        "/data/scan_000012.tif",
    ]


def test_frame_paths_layer2_with_wf():
    p = ProcessParams(
        data_directory="/data",
        orig_filename="scan",
        ext_orig="tif",
        raw_start_nr=10,
        wf_images=100,
        nr_files_per_distance=3,
    )
    paths = frame_paths(p, layer_nr=2)
    # layer 2: base_idx=3, start=10+1*100=110 -> 113, 114, 115
    assert paths == [
        "/data/scan_000113.tif",
        "/data/scan_000114.tif",
        "/data/scan_000115.tif",
    ]


def test_frame_paths_invalid_layer():
    p = ProcessParams(orig_filename="x", nr_files_per_distance=3)
    with pytest.raises(ValueError, match="layer_nr"):
        frame_paths(p, layer_nr=0)


def test_load_tiff_stack_roundtrip(tmp_path):
    """Write a stack, load it back, verify shape and contents."""
    p = ProcessParams(
        data_directory=str(tmp_path),
        orig_filename="img",
        ext_orig="tif",
        raw_start_nr=0,
        wf_images=0,
        nr_files_per_distance=4,
        nr_pixels=8,
    )
    rng = np.random.default_rng(42)
    expected = rng.integers(0, 65535, size=(4, 8, 8), dtype=np.uint16)
    for j in range(4):
        tifffile.imwrite(tmp_path / f"img_{j:06d}.tif", expected[j])

    stack = load_tiff_stack(p, 1, device="cpu", dtype=torch.float64)
    assert stack.shape == (4, 8, 8)
    assert stack.dtype == torch.float64
    np.testing.assert_array_equal(stack.numpy().astype(np.uint16), expected)


def test_load_tiff_stack_missing_file_raises(tmp_path):
    p = ProcessParams(
        data_directory=str(tmp_path),
        orig_filename="img",
        ext_orig="tif",
        nr_files_per_distance=2,
        nr_pixels=4,
    )
    # Only one file present; loader should fail loading the second.
    tifffile.imwrite(tmp_path / "img_000000.tif", np.zeros((4, 4), dtype=np.uint16))
    with pytest.raises(Exception):
        load_tiff_stack(p, 1)


def test_load_tiff_stack_wrong_shape_raises(tmp_path):
    p = ProcessParams(
        data_directory=str(tmp_path),
        orig_filename="img",
        ext_orig="tif",
        nr_files_per_distance=1,
        nr_pixels_y=4,
        nr_pixels_z=4,
    )
    tifffile.imwrite(tmp_path / "img_000000.tif", np.zeros((8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match="shape"):
        load_tiff_stack(p, 1)


def test_from_tensor_validates_shape():
    t = torch.zeros(3, 8, 8)
    out = from_tensor(t, nr_pixels_y=8, nr_pixels_z=8)
    assert out is t  # no-copy


def test_from_tensor_rejects_wrong_y():
    t = torch.zeros(3, 8, 8)
    with pytest.raises(ValueError, match="Y mismatch"):
        from_tensor(t, nr_pixels_y=16)


def test_from_tensor_rejects_wrong_z():
    t = torch.zeros(3, 8, 8)
    with pytest.raises(ValueError, match="Z mismatch"):
        from_tensor(t, nr_pixels_z=16)


def test_from_tensor_rejects_wrong_ndim():
    t = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="3D tensor"):
        from_tensor(t)
