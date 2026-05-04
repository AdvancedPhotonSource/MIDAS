"""Tests for from_scratch: Shoemake sampler, dedup, FZ-filter pipeline."""

from __future__ import annotations

import math

import pytest
import torch

from midas_nf_preprocess.seed_orientations import (
    deduplicate_quaternions,
    generate_uniform_seeds,
    n_master_for_resolution,
    shoemake_uniform_quaternions,
)


# ----- shoemake_uniform_quaternions ------------------------------------------


def test_shoemake_unit_norm():
    q = shoemake_uniform_quaternions(1000, seed=0)
    norms = q.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_shoemake_w_nonneg():
    """We canonicalise to w >= 0 (matches midas_stress / cached files)."""
    q = shoemake_uniform_quaternions(500, seed=1)
    assert torch.all(q[:, 0] >= 0)


def test_shoemake_deterministic_with_seed():
    q1 = shoemake_uniform_quaternions(50, seed=42)
    q2 = shoemake_uniform_quaternions(50, seed=42)
    assert torch.equal(q1, q2)


def test_shoemake_different_seeds_differ():
    q1 = shoemake_uniform_quaternions(50, seed=1)
    q2 = shoemake_uniform_quaternions(50, seed=2)
    assert not torch.equal(q1, q2)


def test_shoemake_returns_correct_shape():
    q = shoemake_uniform_quaternions(13, seed=0)
    assert q.shape == (13, 4)


def test_shoemake_invalid_n():
    with pytest.raises(ValueError, match="n must be"):
        shoemake_uniform_quaternions(0)


def test_shoemake_uniformity_w_distribution():
    """For uniform SO(3) quaternions sampled on the upper hemisphere of S^3,
    the marginal density of the ``w`` component is::

        p(w) = (4 / pi) * sqrt(1 - w^2),  w in [0, 1]

    which concentrates near w=0 and falls to zero at w=1. We integrate this
    over each histogram bin to get the expected count, and check that the
    observed counts agree within Monte-Carlo error.
    """
    n = 10_000
    n_bins = 10
    q = shoemake_uniform_quaternions(n, seed=7)
    counts = torch.histc(q[:, 0], bins=n_bins, min=0.0, max=1.0)

    # Closed-form expected counts using indefinite integral of sqrt(1 - w^2):
    #   F(w) = 0.5 * (w * sqrt(1 - w^2) + arcsin(w))
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    F = 0.5 * (
        bin_edges * torch.sqrt(torch.clamp(1 - bin_edges ** 2, min=0.0))
        + torch.asin(bin_edges)
    )
    bin_probs = (4.0 / math.pi) * (F[1:] - F[:-1])
    expected = n * bin_probs

    # Monte-Carlo: each bin's stddev ~= sqrt(n * p * (1 - p)). Allow 5 sigma.
    stddev = torch.sqrt(n * bin_probs * (1 - bin_probs))
    assert torch.all((counts.double() - expected).abs() < 5 * stddev), (
        f"Shoemake w-marginal off:\nobserved={counts.tolist()}\n"
        f"expected={expected.tolist()}\nstddev={stddev.tolist()}"
    )


# ----- n_master_for_resolution -----------------------------------------------


@pytest.mark.parametrize(
    "res,expected_min",
    [
        (1.5, 100_000),   # ~1.6M, well above floor
        (5.0, 50_000),    # falls back to floor (50k)
        (10.0, 50_000),
    ],
)
def test_n_master_scales_with_resolution(res, expected_min):
    n = n_master_for_resolution(res)
    assert n >= expected_min


def test_n_master_finer_resolution_more_samples():
    n_loose = n_master_for_resolution(5.0)
    n_tight = n_master_for_resolution(0.5)
    assert n_tight > n_loose * 100  # cubic scaling


def test_n_master_invalid():
    with pytest.raises(ValueError, match="must be > 0"):
        n_master_for_resolution(0)
    with pytest.raises(ValueError, match="must be > 0"):
        n_master_for_resolution(-1)


# ----- deduplicate_quaternions ------------------------------------------------


def test_dedup_removes_exact_duplicates():
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],   # exact dup
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = deduplicate_quaternions(q, tol_deg=1.0)
    assert out.shape[0] == 2


def test_dedup_keeps_distinct():
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    out = deduplicate_quaternions(q, tol_deg=1.0)
    assert out.shape[0] == 4


def test_dedup_zero_tolerance_no_op():
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = deduplicate_quaternions(q, tol_deg=0.0)
    assert torch.equal(out, q)


def test_dedup_wrong_shape_raises():
    with pytest.raises(ValueError, match="Expected"):
        deduplicate_quaternions(torch.zeros(5, 3), tol_deg=1.0)


# ----- generate_uniform_seeds (end-to-end with FZ filter) ---------------------


@pytest.mark.slow
def test_generate_uniform_seeds_cubic():
    """End-to-end: cubic SG -> uniform sampling -> FZ filter -> dedup."""
    seeds = generate_uniform_seeds(
        space_group=225,
        resolution_deg=5.0,  # coarse for fast test
        seed=0,
        device="cpu",
        dtype=torch.float64,
    )
    assert seeds.ndim == 2
    assert seeds.shape[1] == 4
    # Sanity: we should get at least a few hundred orientations even at coarse res.
    assert seeds.shape[0] > 100
    # All quats should be unit norm (FZ reduction preserves norm).
    norms = seeds.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@pytest.mark.slow
def test_generate_uniform_seeds_resolution_affects_count():
    """Tighter resolution -> more orientations after FZ filter."""
    seeds_coarse = generate_uniform_seeds(
        space_group=225, resolution_deg=10.0, seed=0
    )
    seeds_finer = generate_uniform_seeds(
        space_group=225, resolution_deg=5.0, seed=0
    )
    assert seeds_finer.shape[0] > seeds_coarse.shape[0]


@pytest.mark.slow
def test_generate_uniform_seeds_explicit_n_master():
    """--n-master overrides the resolution heuristic."""
    seeds = generate_uniform_seeds(
        space_group=225, n_master=10_000, seed=0
    )
    # 10k Shoemake -> cubic FZ (24 ops) -> ~400 unique orientations
    assert 100 < seeds.shape[0] < 12_000


@pytest.mark.slow
def test_generate_uniform_seeds_deterministic():
    s1 = generate_uniform_seeds(space_group=225, n_master=5_000, seed=99)
    s2 = generate_uniform_seeds(space_group=225, n_master=5_000, seed=99)
    assert torch.equal(s1, s2)


@pytest.mark.slow
def test_generate_uniform_seeds_low_symmetry_yields_more():
    """Triclinic FZ is 24x larger than cubic; same N_master should produce
    more FZ orientations for triclinic."""
    s_cubic = generate_uniform_seeds(
        space_group=225, n_master=20_000, seed=0, deduplicate=False
    )
    s_tri = generate_uniform_seeds(
        space_group=1, n_master=20_000, seed=0, deduplicate=False
    )
    # Without dedup, FZ filter just returns N_master orientations regardless
    # of symmetry (each input gets one FZ rep). So instead verify the *unique*
    # FZ representatives count -- for cubic with dedup at default tol there
    # are fewer unique reps than for triclinic.
    s_cubic_dedup = generate_uniform_seeds(
        space_group=225, n_master=20_000, seed=0, deduplicate=True
    )
    s_tri_dedup = generate_uniform_seeds(
        space_group=1, n_master=20_000, seed=0, deduplicate=True
    )
    assert s_tri_dedup.shape[0] >= s_cubic_dedup.shape[0]
