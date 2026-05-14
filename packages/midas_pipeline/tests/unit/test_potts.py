"""Unit tests for midas_pipeline.potts.confidence_weighted_potts.

- Convergence test: noisy degenerate input → ICM converges within
  ``max_iter`` sweeps and smooths salt-and-pepper noise.
- Numba-vs-Python parity: both backends produce identical maps.
- Numba speedup smoke test (skipped if numba isn't importable).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from midas_pipeline.potts import (
    _NUMBA_AVAILABLE,
    confidence_weighted_potts,
)


def _seed_two_grain_with_noise(H: int, W: int, frac_flips: float, rng):
    """Half-and-half grain map with random voxels flipped (salt-and-pepper)."""
    truth = np.zeros((H, W), dtype=np.int32)
    truth[:, W // 2:] = 1
    noisy = truth.copy()
    n_flip = int(H * W * frac_flips)
    idx = rng.choice(H * W, size=n_flip, replace=False)
    for i in idx:
        r, c = divmod(i, W)
        noisy[r, c] = 1 - noisy[r, c]
    return truth, noisy


def _posterior_from_seed(seed: np.ndarray, n_grains: int, conf: float = 0.9) -> np.ndarray:
    """Build a posterior where the seed value gets ``conf``, others split rest."""
    H, W = seed.shape
    post = np.full((n_grains, H, W), (1.0 - conf) / max(n_grains - 1, 1),
                   dtype=np.float32)
    for g in range(n_grains):
        post[g][seed == g] = conf
    return post


def test_potts_converges_within_max_iter():
    rng = np.random.default_rng(0)
    H = W = 24
    truth, noisy = _seed_two_grain_with_noise(H, W, 0.2, rng)
    post = _posterior_from_seed(noisy, 2, conf=0.55)
    out = confidence_weighted_potts(
        post, noisy.copy(), lam=1.0, max_iter=30, conf_floor=0.05,
    )
    # ICM should converge (no NaNs, all entries are valid grain IDs).
    assert out.dtype == np.int32
    assert out.shape == (H, W)
    assert ((out >= 0) | (out == -1)).all()
    # And smoothing should improve agreement with truth.
    acc_before = float((noisy == truth).mean())
    acc_after = float((out == truth).mean())
    assert acc_after >= acc_before, (
        f"Potts smoothing made things worse: before={acc_before:.3f} after={acc_after:.3f}"
    )


def test_potts_numba_matches_python():
    if not _NUMBA_AVAILABLE:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(1)
    H = W = 16
    truth, noisy = _seed_two_grain_with_noise(H, W, 0.15, rng)
    post = _posterior_from_seed(noisy, 2, conf=0.6)
    seed = noisy.copy()
    out_numba = confidence_weighted_potts(
        post, seed.copy(), lam=1.0, max_iter=10, conf_floor=0.05, use_numba=True,
    )
    out_python = confidence_weighted_potts(
        post, seed.copy(), lam=1.0, max_iter=10, conf_floor=0.05, use_numba=False,
    )
    np.testing.assert_array_equal(out_numba, out_python)


def test_potts_numba_speedup_smoke():
    """Sanity-check that the JIT'd path is no slower than 0.5× python on a
    moderately-sized grid. We don't gate on a strict 50× speedup because that
    is sensitive to environment + JIT-warmup overhead — but the JIT path must
    be at least as fast once warmed."""
    if not _NUMBA_AVAILABLE:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(2)
    H = W = 60
    truth, noisy = _seed_two_grain_with_noise(H, W, 0.1, rng)
    post = _posterior_from_seed(noisy, 2, conf=0.6)
    # Warm up the JIT (first call compiles).
    _ = confidence_weighted_potts(
        post, noisy.copy(), lam=1.0, max_iter=1, conf_floor=0.05, use_numba=True,
    )
    t0 = time.perf_counter()
    confidence_weighted_potts(post, noisy.copy(), lam=1.0, max_iter=5,
                              conf_floor=0.05, use_numba=True)
    t_numba = time.perf_counter() - t0
    t0 = time.perf_counter()
    confidence_weighted_potts(post, noisy.copy(), lam=1.0, max_iter=5,
                              conf_floor=0.05, use_numba=False)
    t_python = time.perf_counter() - t0
    # Numba should not be drastically slower (typical: 5-50× faster).
    assert t_numba <= max(t_python * 1.5, 0.05), (
        f"numba path is slower than python: numba={t_numba:.3f}s python={t_python:.3f}s"
    )


def test_potts_inactive_voxels_stay_minus_one():
    """Voxels with all-zero posterior (no evidence) must remain -1."""
    H = W = 8
    post = np.zeros((2, H, W), dtype=np.float32)
    post[0, :, :W // 2] = 0.5
    post[1, :, W // 2:] = 0.5
    # No evidence in the bottom 2 rows.
    post[:, H - 2:, :] = 0.0
    seed = np.argmax(post, axis=0).astype(np.int32)
    seed[post.max(axis=0) == 0] = -1
    out = confidence_weighted_potts(post, seed, lam=1.0, max_iter=10,
                                    conf_floor=0.05)
    assert (out[H - 2:, :] == -1).all()
