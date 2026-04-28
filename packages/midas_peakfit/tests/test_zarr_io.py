"""Test Zarr metadata parsing."""
import numpy as np

from midas_peakfit.zarr_io import parse_zarr_params, load_corrections, read_frame


def test_parse_zarr_params_basic(synthetic_zarr):
    p = parse_zarr_params(str(synthetic_zarr))
    assert p.nFrames == 3
    # Z slow, Y fast
    assert p.NrPixelsZ == 256
    assert p.NrPixelsY == 256
    assert p.NrPixels == 256
    assert p.pixelType == "uint16"
    assert p.bytesPerPx == 2
    assert p.Ycen == 128.0
    assert p.Zcen == 128.0
    assert p.DoFullImage == 1
    assert p.doPeakFit == 1
    assert p.minNrPx == 3
    assert p.maxNrPx == 10000


def test_block_frame_range(synthetic_zarr):
    p = parse_zarr_params(str(synthetic_zarr))
    # All-in-one block
    s, e = p.block_frame_range(0, 1)
    assert (s, e) == (0, 3)
    # Two blocks, block 0 gets the first ceil(3/2)=2 frames
    s, e = p.block_frame_range(0, 2)
    assert (s, e) == (0, 2)
    s, e = p.block_frame_range(1, 2)
    assert (s, e) == (2, 3)


def test_load_corrections_defaults(synthetic_zarr):
    p = parse_zarr_params(str(synthetic_zarr))
    load_corrections(str(synthetic_zarr), p)
    # No darks/floods/masks → defaults
    assert p.dark.shape == (256, 256)
    assert (p.dark == 0).all()
    assert p.flood.shape == (256, 256)
    assert (p.flood == 1).all()
    assert p.mask.shape == (256, 256)
    assert (p.mask == 0).all()


def test_read_frame_shape(synthetic_zarr):
    p = parse_zarr_params(str(synthetic_zarr))
    frame = read_frame(str(synthetic_zarr), 0)
    assert frame.shape == (256, 256)
    assert frame.dtype == np.float64
