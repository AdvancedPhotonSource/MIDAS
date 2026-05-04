"""Tests for the tomo_filter submodule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_preprocess.hex_grid import make_hex_grid
from midas_nf_preprocess.tomo_filter import (
    bbox_mask,
    filter_grid_by_bbox,
    filter_grid_by_tomo,
    load_square_tomo,
    sample_tomo,
)


# -----------------------------------------------------------------------------
# load_square_tomo
# -----------------------------------------------------------------------------


def test_load_square_tomo_roundtrip(tmp_path):
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    p = tmp_path / "tomo.bin"
    arr.tofile(p)
    out = load_square_tomo(p)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_load_square_tomo_non_square_raises(tmp_path):
    arr = np.zeros(60, dtype=np.uint8)  # 60 is not a perfect square
    p = tmp_path / "tomo.bin"
    arr.tofile(p)
    with pytest.raises(ValueError, match="not a perfect square"):
        load_square_tomo(p)


# -----------------------------------------------------------------------------
# sample_tomo: coordinate convention matches the C
# -----------------------------------------------------------------------------


def test_sample_tomo_center_pixel():
    """Origin (0, 0) maps to the image center pixel."""
    nr_px = 9
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    tomo[nr_px - (nr_px // 2), nr_px // 2] = 99  # see C row index nrPxTomo - yPos
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 99


def test_sample_tomo_out_of_bounds_returns_zero():
    """Points beyond the image return zero."""
    tomo = np.full((4, 4), 7, dtype=np.uint8)
    points = torch.tensor([[100.0, 100.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 0


def test_sample_tomo_y_flip():
    """+y in um maps UP in the image, not down (matches C's nrPxTomo - yPos)."""
    nr_px = 5
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    # Mark a pixel with positive y in the lab frame -> upper rows of the image.
    # y_um = +1, px = 1 -> y_pos = 1 + 2 = 3; row = 5 - 3 = 2 (upper half).
    tomo[2, 2] = 42  # row 2 is "above" center
    points = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 42


def test_sample_tomo_torch_tensor_accepted():
    tomo = torch.zeros((4, 4), dtype=torch.uint8)
    tomo[2, 2] = 5
    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    values = sample_tomo(points, tomo, px_tomo_um=1.0)
    assert int(values[0]) == 5


def test_sample_tomo_non_square_raises():
    tomo = np.zeros((4, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="square tomo"):
        sample_tomo(torch.zeros(1, 2), tomo, px_tomo_um=1.0)


def test_sample_tomo_wrong_point_shape_raises():
    tomo = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="\\(N, 2\\)"):
        sample_tomo(torch.zeros(4), tomo, px_tomo_um=1.0)


# -----------------------------------------------------------------------------
# filter_grid_by_tomo
# -----------------------------------------------------------------------------


def test_filter_grid_by_tomo_keeps_inside_blob():
    """A grid covering the origin should keep only points landing on the blob.

    The 3x3 blob sits at tomo[4:7, 4:7] of an 11x11 image with px=1um. The
    coordinate convention (filterGridfromTomo.c L39-L43) is:
        col = int(x/px) + 5, row = 11 - (int(y/px) + 5)
    So the blob covers cells where ``int(x)`` in {-1, 0, 1} and ``int(y)`` in
    {0, 1, 2} -- accounting for ``int()`` truncation toward zero, kept points
    have ``-1.99 <= x < 2.0`` and ``0.0 <= y < 3.0``.
    """
    grid = make_hex_grid(grid_size=1.0, r_sample=4.0)
    nr_px = 11
    tomo = np.zeros((nr_px, nr_px), dtype=np.uint8)
    tomo[4:7, 4:7] = 1  # 3x3 blob at the center
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == int(mask.sum())
    assert filtered.shape[0] > 0  # at least one grid point landed on the blob
    assert filtered.shape[0] < grid.shape[0]  # the rest were filtered out
    # int(x_um) must be in {-1, 0, 1}: equivalent to -1 <= x_um and x_um < 2.
    x_int = filtered[:, 2].to(torch.int64)
    y_int = filtered[:, 3].to(torch.int64)
    assert torch.all((x_int >= -1) & (x_int <= 1))
    assert torch.all((y_int >= 0) & (y_int <= 2))


def test_filter_grid_by_tomo_all_zero_keeps_nothing():
    grid = make_hex_grid(grid_size=1.0, r_sample=3.0)
    tomo = np.zeros((9, 9), dtype=np.uint8)
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == 0
    assert int(mask.sum()) == 0


def test_filter_grid_by_tomo_all_one_keeps_within_image():
    grid = make_hex_grid(grid_size=1.0, r_sample=3.0)
    nr_px = 11  # large enough to cover the grid
    tomo = np.ones((nr_px, nr_px), dtype=np.uint8)
    filtered, mask = filter_grid_by_tomo(grid, tomo, px_tomo_um=1.0)
    assert filtered.shape[0] == grid.shape[0]


def test_filter_grid_wrong_shape_raises():
    bad = torch.zeros(10, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="\\(N, 5\\)"):
        filter_grid_by_tomo(bad, np.zeros((4, 4), dtype=np.uint8), 1.0)


# -----------------------------------------------------------------------------
# bbox_mask + filter_grid_by_bbox
# -----------------------------------------------------------------------------


def test_bbox_mask_geometry():
    grid = make_hex_grid(grid_size=1.0, r_sample=4.0)
    mask = bbox_mask(grid, [-1.0, 1.0, -1.0, 1.0])
    # All True positions must be within the bbox.
    kept = grid[mask]
    assert torch.all(kept[:, 2] >= -1.0)
    assert torch.all(kept[:, 2] <= 1.0)
    assert torch.all(kept[:, 3] >= -1.0)
    assert torch.all(kept[:, 3] <= 1.0)


def test_bbox_mask_invalid_length_raises():
    with pytest.raises(ValueError, match="length"):
        bbox_mask(torch.zeros(3, 5), [0, 1, 2])


def test_bbox_mask_reversed_corners_raises():
    with pytest.raises(ValueError, match="reversed"):
        bbox_mask(torch.zeros(3, 5), [1, 0, 0, 1])


def test_filter_grid_by_bbox():
    grid = make_hex_grid(grid_size=1.0, r_sample=5.0)
    f, m = filter_grid_by_bbox(grid, [-2.0, 2.0, -2.0, 2.0])
    assert f.shape[0] == int(m.sum())
