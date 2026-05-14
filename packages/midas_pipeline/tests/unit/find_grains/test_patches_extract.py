"""21x21 patch extraction at a known detector coordinate."""

from __future__ import annotations

import numpy as np

from midas_pipeline.find_grains import (
    PATCH_HALF_SIZE,
    PATCH_SIZE,
    extract_patch_from_frame,
    extract_patches_from_spot_map,
)


def test_patch_size_is_21():
    assert PATCH_SIZE == 21
    assert PATCH_HALF_SIZE == 10


def test_extract_patch_centered_returns_window():
    """Patch centered on (yc, zc) in a non-overlapping grid recovers a 21x21 window."""
    rng = np.random.default_rng(0)
    frame = rng.random((100, 100)).astype(np.float32)
    yc, zc = 50, 60
    p = extract_patch_from_frame(frame, yc, zc)
    # Expected: frame[zc-10:zc+11, yc-10:yc+11]
    expected = frame[zc - PATCH_HALF_SIZE : zc + PATCH_HALF_SIZE + 1,
                     yc - PATCH_HALF_SIZE : yc + PATCH_HALF_SIZE + 1]
    np.testing.assert_array_equal(p, expected)


def test_patch_near_edge_zero_pads():
    frame = np.ones((30, 30), dtype=np.float32)
    p = extract_patch_from_frame(frame, y_center=2, z_center=2)
    # Cells at (dz, dy) where zi or yi < 0 should be 0; others = 1.
    assert p[0, 0] == 0.0   # dz=-10, zi=-8
    assert p[PATCH_HALF_SIZE, PATCH_HALF_SIZE] == 1.0   # center
    # Lower-right corner: dz=+10 → zi=12, dy=+10 → yi=12 (both valid).
    assert p[PATCH_SIZE - 1, PATCH_SIZE - 1] == 1.0


def test_extract_patches_from_spot_map_minimal_roundtrip(tmp_path):
    n_g, n_h, n_s = 1, 1, 1
    spot_id_arr = np.array([[[42]]], dtype=np.int32)
    spot_meta = np.zeros((1, 1, 1, 4), dtype=np.float64)
    spot_meta[0, 0, 0, 2] = 5.0   # yCen
    spot_meta[0, 0, 0, 3] = 6.0   # zCen

    frame = np.full((20, 20), 7.0, dtype=np.float32)

    def loader(sid):
        return frame if sid == 42 else None

    patches_path, spot_pos_path = extract_patches_from_spot_map(
        spot_id_arr=spot_id_arr,
        spot_meta=spot_meta,
        n_grains=n_g,
        max_n_hkls=n_h,
        n_scans=n_s,
        frame_loader=loader,
        output_dir=tmp_path,
    )
    patches = np.frombuffer(open(patches_path, "rb").read(), dtype=np.float32)
    patches = patches.reshape(n_g, n_h, n_s, PATCH_SIZE, PATCH_SIZE)
    # The 21x21 window around (yCen=5, zCen=6) on a 20x20 all-7 frame
    # has its central portion = 7; corners that fall outside [0, 20) stay 0.
    # Center cell (z=6, y=5) is at index (PATCH_HALF_SIZE, PATCH_HALF_SIZE).
    assert patches[0, 0, 0, PATCH_HALF_SIZE, PATCH_HALF_SIZE] == 7.0
    spot_pos = np.frombuffer(open(spot_pos_path, "rb").read(), dtype=np.float64)
    spot_pos = spot_pos.reshape(n_g, n_h, n_s, 2)
    assert spot_pos[0, 0, 0, 0] == 5.0
    assert spot_pos[0, 0, 0, 1] == 6.0
