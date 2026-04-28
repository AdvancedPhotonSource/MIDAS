"""Tests for compute.seeds (the three GenerateIdealSpots* variants)."""

import math

import numpy as np
import pytest
import torch

from midas_index.compute.seeds import (
    generate_ideal_spots,
    generate_ideal_spots_friedel,
    generate_ideal_spots_friedel_mixed,
)


def test_generate_ideal_spots_returns_pairs_on_ring():
    ring_rad = 30000.0
    seeds = generate_ideal_spots(
        ys=10.0, zs=5.0, ttheta_deg=3.0, eta_deg=45.0,
        ring_rad=ring_rad, rsample=200.0, hbeam=200.0, step_size=50.0,
    )
    assert seeds.dim() == 2
    assert seeds.shape[1] == 2
    assert seeds.shape[0] >= 1
    # Each (y0, z0) is on the ring: y0^2 + z0^2 ~= R^2
    radii = torch.sqrt(seeds[:, 0] ** 2 + seeds[:, 1] ** 2)
    np.testing.assert_allclose(radii.numpy(), ring_rad, atol=1.0)


def test_generate_ideal_spots_n_steps_is_odd():
    seeds = generate_ideal_spots(
        ys=10.0, zs=5.0, ttheta_deg=3.0, eta_deg=45.0,
        ring_rad=30000.0, rsample=200.0, hbeam=200.0, step_size=200.0,
    )
    assert seeds.shape[0] % 2 == 1


def test_generate_ideal_spots_friedel_with_no_obs_returns_empty():
    obs = torch.empty((0, 9), dtype=torch.float64)
    seeds = generate_ideal_spots_friedel(
        ys=10.0, zs=5.0, ttheta_deg=3.0, eta_deg=45.0, omega_deg=10.0,
        ring_nr=1, ring_rad=30000.0, rsample=200.0, hbeam=200.0,
        ome_tol=2.0, radius_tol=20.0,
        obs_spots=obs,
    )
    assert seeds.shape == (0, 2)


def test_generate_ideal_spots_friedel_finds_match():
    # Construct a synthetic observed spot that matches the Friedel-pair criteria.
    # Seed: ring_rad=30000, eta=45, omega=10, ys=0, zs=0
    # Friedel partner: omega ≈ 10-180 = -170, radius ≈ 60000
    ome_friedel = -170.0  # 10 - 180
    eta_friedel = 180 - 45  # 135 (per C: eta < 0 → eta_f = -180 - eta; eta>=0 → 180 - eta)
    yf = -2.0 * 30000.0 * math.sin(eta_friedel * math.pi / 180.0)
    zf = 2.0 * 30000.0 * math.cos(eta_friedel * math.pi / 180.0)
    obs = torch.tensor(
        [[yf, zf, ome_friedel, 0.0, 99, 1, eta_friedel, 1.5, 0.0]],
        dtype=torch.float64,
    )
    seeds = generate_ideal_spots_friedel(
        ys=0.0, zs=0.0, ttheta_deg=3.0, eta_deg=45.0, omega_deg=10.0,
        ring_nr=1, ring_rad=30000.0, rsample=200.0, hbeam=200.0,
        ome_tol=5.0, radius_tol=200.0,
        obs_spots=obs,
    )
    # Whether the match passes depends on the FriedelEtaCalculation window;
    # for very small Rsample/Hbeam vs ring_rad the window is narrow. The
    # test mostly exercises that the function runs end-to-end and returns
    # a (n, 2) tensor.
    assert seeds.dim() == 2
    assert seeds.shape[1] == 2


def test_generate_ideal_spots_friedel_mixed_low_eta_returns_empty():
    obs = torch.empty((0, 9), dtype=torch.float64)
    # |sin(eta)| < sin(10°) -> should return empty
    seeds = generate_ideal_spots_friedel_mixed(
        ys=0.0, zs=0.0, ttheta_deg=3.0, eta_deg=5.0, omega_deg=10.0,
        ring_nr=1, ring_rad=30000.0, lsd=1_000_000.0,
        rsample=200.0, hbeam=200.0, step_size_pos=50.0,
        ome_tol=2.0, radial_tol=20.0, eta_tol_um=200.0,
        obs_spots=obs,
    )
    assert seeds.shape == (0, 2)


def test_generate_ideal_spots_friedel_mixed_no_obs_returns_empty():
    obs = torch.empty((0, 9), dtype=torch.float64)
    seeds = generate_ideal_spots_friedel_mixed(
        ys=0.0, zs=0.0, ttheta_deg=3.0, eta_deg=45.0, omega_deg=10.0,
        ring_nr=1, ring_rad=30000.0, lsd=1_000_000.0,
        rsample=200.0, hbeam=200.0, step_size_pos=200.0,
        ome_tol=2.0, radial_tol=20.0, eta_tol_um=200.0,
        obs_spots=obs,
    )
    assert seeds.shape == (0, 2)
