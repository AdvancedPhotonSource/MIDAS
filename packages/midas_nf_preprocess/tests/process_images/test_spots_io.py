"""Tests for the SpotsInfo.bin bit-packed writer/reader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images import SpotsBitMask


# -----------------------------------------------------------------------------
# Sizing
# -----------------------------------------------------------------------------


def test_size_calculation():
    sm = SpotsBitMask(n_layers=2, nr_files_per_layer=4, nr_pixels_y=8, nr_pixels_z=8)
    # 2 * 4 * 8 * 8 = 512 bits = 16 uint32 words (in-memory; disk bytes are
    # identical to int32 -- the C reader interprets as int *).
    assert sm.total_bits == 512
    assert sm.n_words == 16
    assert sm.buffer.dtype == np.uint32
    assert sm.buffer.shape == (16,)


def test_size_calculation_non_multiple_of_32():
    sm = SpotsBitMask(n_layers=1, nr_files_per_layer=1, nr_pixels_y=5, nr_pixels_z=5)
    # 25 bits = 1 word (rounded up)
    assert sm.total_bits == 25
    assert sm.n_words == 1


# -----------------------------------------------------------------------------
# Per-bit ops (round-trip)
# -----------------------------------------------------------------------------


def test_set_test_bit_single():
    sm = SpotsBitMask(1, 1, 8, 8)
    assert not sm.test_bit_yz(0, 0, 3, 4)
    sm.set_bit_yz(0, 0, 3, 4)
    assert sm.test_bit_yz(0, 0, 3, 4)


def test_set_bit_idempotent():
    sm = SpotsBitMask(1, 1, 8, 8)
    sm.set_bit_yz(0, 0, 1, 1)
    sm.set_bit_yz(0, 0, 1, 1)
    assert sm.count_bits() == 1


def test_count_bits_independent_writes():
    sm = SpotsBitMask(1, 1, 16, 16)
    sm.set_bit_yz(0, 0, 0, 0)
    sm.set_bit_yz(0, 0, 5, 7)
    sm.set_bit_yz(0, 0, 15, 15)
    assert sm.count_bits() == 3


def test_count_bits_zero_initially():
    sm = SpotsBitMask(1, 1, 8, 8)
    assert sm.count_bits() == 0


def test_set_bit_out_of_range_raises():
    sm = SpotsBitMask(1, 1, 4, 4)
    with pytest.raises(IndexError):
        sm.set_bit_yz(layer=5, frame=0, y_raw=0, z_raw=0)


# -----------------------------------------------------------------------------
# Vectorized: set_frame_from_labels
# -----------------------------------------------------------------------------


def test_set_frame_from_labels_writes_correct_bits():
    sm = SpotsBitMask(1, 1, 8, 8)
    labels = torch.zeros(8, 8, dtype=torch.int64)
    labels[3, 4] = 1
    labels[5, 5] = 2
    n = sm.set_frame_from_labels(0, 0, labels)
    assert n == 2
    # The C convention flips: y_C = 8-1-y_raw, z_C = 8-1-z_raw.
    # For (z_raw=3, y_raw=4): set_bit_yz(layer=0, frame=0, y_raw=4, z_raw=3) should be set.
    assert sm.test_bit_yz(0, 0, 4, 3)
    assert sm.test_bit_yz(0, 0, 5, 5)


def test_set_frame_from_labels_empty():
    sm = SpotsBitMask(1, 1, 4, 4)
    labels = torch.zeros(4, 4, dtype=torch.int64)
    n = sm.set_frame_from_labels(0, 0, labels)
    assert n == 0
    assert sm.count_bits() == 0


def test_set_frame_from_labels_wrong_shape():
    sm = SpotsBitMask(1, 1, 4, 4)
    labels = torch.zeros(8, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="labels shape"):
        sm.set_frame_from_labels(0, 0, labels)


def test_set_frame_from_labels_full_image():
    """Setting every pixel -> count_bits == nr_pixels."""
    sm = SpotsBitMask(1, 1, 4, 4)
    labels = torch.ones(4, 4, dtype=torch.int64)
    n = sm.set_frame_from_labels(0, 0, labels)
    assert n == 16
    assert sm.count_bits() == 16


def test_set_frame_isolation_between_layers():
    """Bits in layer 0 should not bleed into layer 1's bit window."""
    sm = SpotsBitMask(2, 1, 4, 4)
    labels = torch.ones(4, 4, dtype=torch.int64)
    sm.set_frame_from_labels(0, 0, labels)
    # Total = 16. Layer 1 should read 0 everywhere.
    assert sm.count_bits() == 16
    for y in range(4):
        for z in range(4):
            assert not sm.test_bit_yz(1, 0, y, z)


def test_set_frame_isolation_between_frames():
    sm = SpotsBitMask(1, 3, 4, 4)
    labels = torch.ones(4, 4, dtype=torch.int64)
    sm.set_frame_from_labels(0, 1, labels)  # only frame 1
    assert sm.count_bits() == 16
    for y in range(4):
        for z in range(4):
            assert not sm.test_bit_yz(0, 0, y, z)
            assert sm.test_bit_yz(0, 1, y, z)
            assert not sm.test_bit_yz(0, 2, y, z)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def test_write_and_read_roundtrip(tmp_path):
    sm = SpotsBitMask(1, 1, 8, 8)
    sm.set_bit_yz(0, 0, 1, 2)
    sm.set_bit_yz(0, 0, 7, 7)
    path = tmp_path / "SpotsInfo.bin"
    sm.write(path)

    sm2 = SpotsBitMask.open_mmap(path, 1, 1, 8, 8, mode="r")
    assert sm2.test_bit_yz(0, 0, 1, 2)
    assert sm2.test_bit_yz(0, 0, 7, 7)
    assert sm2.count_bits() == 2


def test_write_byte_count(tmp_path):
    sm = SpotsBitMask(1, 1, 16, 16)
    path = tmp_path / "SpotsInfo.bin"
    sm.write(path)
    # 256 bits = 8 int32 words = 32 bytes
    assert path.stat().st_size == 32


def test_open_mmap_creates_file(tmp_path):
    path = tmp_path / "new" / "SpotsInfo.bin"
    sm = SpotsBitMask.open_mmap(path, 1, 1, 8, 8, mode="r+")
    sm.set_bit_yz(0, 0, 0, 0)
    sm.flush()
    assert path.exists()


def test_open_mmap_accumulates_per_layer(tmp_path):
    """Two separate processes (layer1, layer2) sharing the same mmap."""
    path = tmp_path / "SpotsInfo.bin"
    # Layer 1
    sm1 = SpotsBitMask.open_mmap(path, 2, 1, 4, 4, mode="r+")
    sm1.set_bit_yz(0, 0, 1, 1)
    sm1.flush()
    del sm1

    # Layer 2 (separate handle)
    sm2 = SpotsBitMask.open_mmap(path, 2, 1, 4, 4, mode="r+")
    sm2.set_bit_yz(1, 0, 2, 2)
    sm2.flush()
    del sm2

    # Read final
    sm3 = SpotsBitMask.open_mmap(path, 2, 1, 4, 4, mode="r")
    assert sm3.test_bit_yz(0, 0, 1, 1)
    assert sm3.test_bit_yz(1, 0, 2, 2)
    assert sm3.count_bits() == 2


def test_open_mmap_missing_read_only_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SpotsBitMask.open_mmap(tmp_path / "nope.bin", 1, 1, 4, 4, mode="r")


# -----------------------------------------------------------------------------
# Bit-position layout matches C SetBit macro
# -----------------------------------------------------------------------------


def test_bit_layout_matches_c_setbit():
    """The C SetBit(A, k) macro is: A[k/32] |= (1 << (k%32)).

    For (layer=0, frame=0, y_raw=0, z_raw=0):
      y_C = NrPixelsY-1-0 = 7, z_C = 7 (for an 8x8 frame)
      BinNr = 0 + 0 + 7*8 + 7 = 63
      A[63/32] = A[1], bit 63%32 = 31
      => buffer[1] should have its top bit set: 0x80000000 as uint32, equivalent to
         -2147483648 when reinterpreted as int32 (which is what the C reader does).
    """
    sm = SpotsBitMask(1, 1, 8, 8)
    sm.set_bit_yz(0, 0, 0, 0)
    assert sm.buffer[0] == 0
    assert sm.buffer[1] == np.uint32(0x80000000)
    # Same byte pattern reinterpreted as int32 == -2^31, the C-side view.
    assert sm.buffer.view(np.int32)[1] == np.int32(-2147483648)
