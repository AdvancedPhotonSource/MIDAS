"""Tests for compute.position_grid."""

import math

import numpy as np
import torch

from midas_index.compute.position_grid import (
    build_position_grid,
    calc_n_range,
    spot_to_unrotated_batch,
)


def test_calc_n_range_symmetric():
    xi = torch.tensor(1_000_000.0, dtype=torch.float64)
    yi = torch.tensor(0.0, dtype=torch.float64)
    ys = torch.tensor(100.0, dtype=torch.float64)
    y0 = torch.tensor(50.0, dtype=torch.float64)
    n_min, n_max = calc_n_range(xi, yi, ys, y0, r_sample=200.0, step_size=5.0)
    # n_min must equal -n_max
    assert int(n_min.item()) == -int(n_max.item())
    assert int(n_max.item()) > 0


def test_spot_to_unrotated_zero_n_gives_origin_offset():
    xi = torch.tensor(1.0, dtype=torch.float64)
    yi = torch.tensor(0.0, dtype=torch.float64)
    zi = torch.tensor(0.0, dtype=torch.float64)
    ys = torch.tensor(10.0, dtype=torch.float64)
    zs = torch.tensor(20.0, dtype=torch.float64)
    y0 = torch.tensor(3.0, dtype=torch.float64)
    z0 = torch.tensor(5.0, dtype=torch.float64)
    n = torch.tensor(0, dtype=torch.int64)
    omega = torch.tensor(0.0, dtype=torch.float64)
    pos = spot_to_unrotated_batch(
        xi=xi, yi=yi, zi=zi, ys=ys, zs=zs, y0=y0, z0=z0,
        step_size_in_x=1.0, n=n, omega_deg=omega,
    )
    # at n=0, x1=0, y1=ys-y0=7, z1=zs-z0=15. Omega=0 -> ga=0, gb=7, gc=15.
    np.testing.assert_allclose(pos.numpy(), [0.0, 7.0, 15.0], atol=1e-12)


def test_spot_to_unrotated_omega_90_rotates_into_a():
    xi = torch.tensor(1.0, dtype=torch.float64)
    yi = torch.tensor(0.0, dtype=torch.float64)
    zi = torch.tensor(0.0, dtype=torch.float64)
    ys = torch.tensor(10.0, dtype=torch.float64)
    zs = torch.tensor(0.0, dtype=torch.float64)
    y0 = torch.tensor(3.0, dtype=torch.float64)
    z0 = torch.tensor(0.0, dtype=torch.float64)
    n = torch.tensor(0, dtype=torch.int64)
    omega = torch.tensor(90.0, dtype=torch.float64)
    pos = spot_to_unrotated_batch(
        xi=xi, yi=yi, zi=zi, ys=ys, zs=zs, y0=y0, z0=z0,
        step_size_in_x=1.0, n=n, omega_deg=omega,
    )
    # x1=0, y1=7, z1=0. Omega=90 -> ga=y1*sin(90)=7, gb=y1*cos(90)=0, gc=0
    np.testing.assert_allclose(pos.numpy(), [7.0, 0.0, 0.0], atol=1e-12)


def test_build_position_grid_shapes_match_n_range():
    seed_y0 = torch.tensor([50.0, 100.0], dtype=torch.float64)
    seed_z0 = torch.tensor([30.0, 60.0], dtype=torch.float64)
    pos, seed_idx = build_position_grid(
        seed_y0=seed_y0, seed_z0=seed_z0,
        ys=80.0, zs=20.0, omega_deg=10.0,
        distance=1_000_000.0, r_sample=200.0, step_size=5.0,
    )
    assert pos.dim() == 2
    assert pos.shape[1] == 3
    assert pos.shape[0] == seed_idx.shape[0]
    assert int(seed_idx.unique().numel()) == 2  # both seeds represented
