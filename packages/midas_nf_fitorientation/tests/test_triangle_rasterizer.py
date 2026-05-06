"""Tests for the vectorised triangle rasteriser (port of CalcPixels2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_nf_fitorientation.screen import rasterize_triangles


def _python_calc_pixels2(verts):
    """Reference implementation: a faithful Python port of
    ``CalcPixels2`` (SharedFuncsFit.c:308-370). Used as a parity oracle.

    ``verts`` is shape (3, 2) of (y, z) integer pixel coords.
    Returns a list of (y, z) tuples kept by the rasteriser.
    """
    verts = [(int(round(verts[k][0])), int(round(verts[k][1]))) for k in range(3)]
    min_y = min(v[0] for v in verts); max_y = max(v[0] for v in verts)
    min_z = min(v[1] for v in verts); max_z = max(v[1] for v in verts)

    def orient2d(a, b, p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def dist_sq(a, b, p):
        num = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        denSq = (a[1] - b[1]) ** 2 + (b[0] - a[0]) ** 2
        if denSq == 0:
            return float("inf")
        return (num * num) / denSq

    pixels = []
    v0, v1, v2 = verts
    for py in range(min_y, max_y + 1):
        for pz in range(min_z, max_z + 1):
            p = (py, pz)
            w0 = orient2d(v1, v2, p)
            w1 = orient2d(v2, v0, p)
            w2 = orient2d(v0, v1, p)
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                pixels.append((py, pz))
            elif (dist_sq(v1, v2, p) < 0.9801
                  or dist_sq(v2, v0, p) < 0.9801
                  or dist_sq(v0, v1, p) < 0.9801):
                pixels.append((py, pz))
    return set(pixels)


def _vectorised_pixel_set(rel_y, rel_z, t_idx=0):
    """Run our vectorised rasteriser and pull out triangle ``t_idx``'s
    in-triangle pixel set (so we can compare to the reference).
    """
    abs_y, abs_z, valid = rasterize_triangles(rel_y, rel_z)
    keep = valid[t_idx]
    yc = abs_y[t_idx][keep].cpu().tolist()
    zc = abs_z[t_idx][keep].cpu().tolist()
    return set(zip(yc, zc))


def test_rasterizer_matches_reference_simple_triangle():
    verts = np.array([[0, 0], [4, 0], [2, 4]], dtype=np.int64)
    rel_y = torch.tensor(verts[:, 0:1].T, dtype=torch.long).reshape(1, 3)
    rel_z = torch.tensor(verts[:, 1:2].T, dtype=torch.long).reshape(1, 3)
    ours = _vectorised_pixel_set(rel_y, rel_z)
    expected = _python_calc_pixels2(verts)
    assert ours == expected


def test_rasterizer_matches_reference_thin_triangle():
    # Very acute triangle: should still pick up edge pixels via the
    # 0.9801 distance tolerance.
    verts = np.array([[0, 0], [10, 1], [5, 0]], dtype=np.int64)
    rel_y = torch.tensor(verts[:, 0:1].T, dtype=torch.long).reshape(1, 3)
    rel_z = torch.tensor(verts[:, 1:2].T, dtype=torch.long).reshape(1, 3)
    ours = _vectorised_pixel_set(rel_y, rel_z)
    expected = _python_calc_pixels2(verts)
    assert ours == expected


def test_rasterizer_handles_degenerate_single_pixel():
    """All three vertices collapse to one pixel — should return that
    pixel and nothing else."""
    rel_y = torch.tensor([[3, 3, 3]])
    rel_z = torch.tensor([[5, 5, 5]])
    abs_y, abs_z, valid = rasterize_triangles(rel_y, rel_z)
    assert valid.sum() == 1
    assert int(abs_y[0][valid[0]].item()) == 3
    assert int(abs_z[0][valid[0]].item()) == 5


def test_rasterizer_batched_independence():
    """Each triangle in a batch is rasterised against its own bbox;
    swapping rows must not change the per-row output."""
    a = np.array([[0, 0], [3, 0], [1, 2]], dtype=np.int64)
    b = np.array([[2, 5], [6, 5], [4, 8]], dtype=np.int64)
    rel_y = torch.tensor(np.stack([a[:, 0], b[:, 0]], axis=0), dtype=torch.long)
    rel_z = torch.tensor(np.stack([a[:, 1], b[:, 1]], axis=0), dtype=torch.long)
    a_set = _vectorised_pixel_set(rel_y, rel_z, t_idx=0)
    b_set = _vectorised_pixel_set(rel_y, rel_z, t_idx=1)
    assert a_set == _python_calc_pixels2(a)
    assert b_set == _python_calc_pixels2(b)
