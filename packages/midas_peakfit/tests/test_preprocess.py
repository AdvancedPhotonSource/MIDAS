"""Test pre-processing pipeline (square-pad, transforms, dark/flood)."""
import numpy as np

from midas_peakfit.preprocess import (
    apply_image_transformations,
    make_square_image,
    transpose_square,
    preprocess_frame,
)


def test_square_pad_y_gt_z():
    """Z=3, Y=5 input → NrPixels=5; out[:3, :5] = in, rest zero."""
    a = np.arange(15, dtype=np.float64).reshape(3, 5)
    out = make_square_image(a, NrPixels=5, NrPixelsY=5, NrPixelsZ=3)
    np.testing.assert_array_equal(out[:3, :5], a)
    np.testing.assert_array_equal(out[3:, :], 0.0)


def test_square_pad_z_gt_y():
    """Z=5, Y=3 input → NrPixels=5; out[:5, :3] = in, out[:, 3:] = 0."""
    a = np.arange(15, dtype=np.float64).reshape(5, 3)
    out = make_square_image(a, NrPixels=5, NrPixelsY=3, NrPixelsZ=5)
    np.testing.assert_array_equal(out[:5, :3], a)
    np.testing.assert_array_equal(out[:, 3:], 0.0)


def test_apply_transforms_flip_h():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [1])  # flip horizontal (Y)
    np.testing.assert_array_equal(out, np.array([[3, 2, 1], [6, 5, 4]]))


def test_apply_transforms_flip_v():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [2])  # flip vertical (Z)
    np.testing.assert_array_equal(out, np.array([[4, 5, 6], [1, 2, 3]]))


def test_apply_transforms_transpose():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    out = apply_image_transformations(a, [3])  # transpose
    np.testing.assert_array_equal(out, a.T)


def test_apply_transforms_compose():
    a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    # Sequence: flip-H then transpose
    out = apply_image_transformations(a, [1, 3])
    expected = a[:, ::-1].T
    np.testing.assert_array_equal(out, expected)


def test_preprocess_frame_smoke():
    """Per-frame pipeline runs without error and returns float64 (N, N)."""
    Y, Z = 64, 64
    raw = np.full((Z, Y), 100.0, dtype=np.float64)
    raw[20:25, 20:25] = 200.0
    dark = np.zeros((Y, Y), dtype=np.float64)
    flood = np.ones((Y, Y), dtype=np.float64)
    gc = np.full((Y, Y), 50.0, dtype=np.float64)

    out = preprocess_frame(
        raw,
        NrPixels=Y, NrPixelsY=Y, NrPixelsZ=Z,
        transform_options=[],
        dark=dark, flood=flood, good_coords=gc,
        bc=1.0, bad_px_intensity=0.0, make_map=0,
    )
    assert out.shape == (Y, Y)
    assert out.dtype == np.float64
    # Bright pixels are above threshold; transposed they end up at imgCorrBC[20:25, 20:25]
    assert (out > 0).any()
