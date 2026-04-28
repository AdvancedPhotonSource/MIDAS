"""Sample-position grid (ga, gb, gc) builder.

Mirrors `spot_to_unrotated_coordinates` and `calc_n_max_min` from
`FF_HEDM/src/IndexerOMP.c:1001-1028`.

For each seed (y0, z0), the indexer sweeps an integer offset n along the beam
direction; for each n it computes the sample-frame position (ga, gb, gc) that
the predicted spot must originate from. This is fed into the forward model
as the `positions` argument.

  spot_to_unrotated_coordinates:
      lambda = step_size * (n / xi)
      x1 = lambda * xi
      y1 = ys - y0 + lambda * yi
      z1 = zs - z0 + lambda * zi
      ga = x1 * cos(omega) + y1 * sin(omega)
      gb = y1 * cos(omega) - x1 * sin(omega)
      gc = z1

  calc_n_max_min: solves the quadratic
      a * lambda² + b * lambda + c = 0
      a = xi² + yi²
      b = 2 * yi * (ys - y0)
      c = (ys - y0)² - R_sample²
      lambda_max = (-b + sqrt(D)) / (2a) + 20
      n_max = floor(lambda_max * xi / step_size)
      n_min = -n_max
"""

from __future__ import annotations

import math

import torch


def calc_n_range(
    xi: torch.Tensor,
    yi: torch.Tensor,
    ys: torch.Tensor,
    y0: torch.Tensor,
    r_sample: float,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (n_min, n_max) integer tensors per seed.

    Mirrors `calc_n_max_min` from IndexerOMP.c:1001. All inputs broadcast.
    """
    dy = ys - y0
    a = xi * xi + yi * yi
    b = 2.0 * yi * dy
    c = dy * dy - r_sample * r_sample
    D = b * b - 4.0 * a * c
    P = torch.sqrt(D.clamp_min(0.0))
    lambda_max = (-b + P) / (2.0 * a) + 20.0
    n_max = torch.floor(lambda_max * xi / step_size).to(torch.int64)
    n_min = -n_max
    return n_min, n_max


def spot_to_unrotated_batch(
    xi: torch.Tensor,
    yi: torch.Tensor,
    zi: torch.Tensor,
    ys: torch.Tensor,
    zs: torch.Tensor,
    y0: torch.Tensor,
    z0: torch.Tensor,
    step_size_in_x: float,
    n: torch.Tensor,
    omega_deg: torch.Tensor,
) -> torch.Tensor:
    """Vectorized port of `spot_to_unrotated_coordinates`.

    All scalar arrays must broadcast to a common shape; output is `(..., 3)`
    holding (ga, gb, gc) per element.
    """
    lam = step_size_in_x * (n.to(xi.dtype) / xi)
    x1 = lam * xi
    y1 = (ys - y0) + lam * yi
    z1 = (zs - z0) + lam * zi
    omega_rad = omega_deg * (math.pi / 180.0)
    cos_o = torch.cos(omega_rad)
    sin_o = torch.sin(omega_rad)
    ga = x1 * cos_o + y1 * sin_o
    gb = y1 * cos_o - x1 * sin_o
    gc = z1
    return torch.stack([ga, gb, gc], dim=-1)


def build_position_grid(
    seed_y0: torch.Tensor,           # (n_seeds,)
    seed_z0: torch.Tensor,           # (n_seeds,)
    ys: float,
    zs: float,
    omega_deg: float,
    distance: float,                  # detector distance (xi)
    r_sample: float,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the full per-seed (n, position) grid.

    For each (y0, z0) seed candidate, expand the n-range and produce one
    (ga, gb, gc) per (seed, n) pair.

    Parameters
    ----------
    seed_y0, seed_z0 : (n_seeds,) tensors of seed positions on the detector.
    ys, zs : seed spot's measured detector position (scalar).
    omega_deg : seed spot's omega (degrees, scalar).
    distance : Lsd (sample-detector distance, scalar). Used as `xi`.
    r_sample : sample radius (Rsample param). Determines the n-range.
    step_size : `IndexerParams.StepsizePos`, in same units as Lsd.

    Returns
    -------
    pos : torch.Tensor (n_total, 3)
        (ga, gb, gc) for each (seed, n) pair, flattened.
    seed_idx : torch.Tensor (n_total,) int64
        Which seed each row came from (so callers can scatter results back).
    """
    device = seed_y0.device
    dtype = seed_y0.dtype

    # Direction vector (sample -> seed point on detector), unit-length —
    # mirrors `MakeUnitLength(Lsd, y0, z0, &xi, &yi, &zi)` in C
    # (IndexerOMP.c:1141 etc.). Without this normalization, `calc_n_range`
    # produces n_max ~ Lsd/step_size (millions of position offsets).
    L = torch.sqrt(
        torch.tensor(distance, device=device, dtype=dtype) ** 2
        + seed_y0 ** 2 + seed_z0 ** 2
    )
    xi_t = torch.tensor(distance, device=device, dtype=dtype) / L
    yi_t = seed_y0 / L
    zi_t = seed_z0 / L
    ys_t = torch.tensor(ys, device=device, dtype=dtype)
    zs_t = torch.tensor(zs, device=device, dtype=dtype)

    n_min, n_max = calc_n_range(
        xi_t, yi_t, ys_t, seed_y0, r_sample, step_size,
    )
    # Broadcast each seed's n-range into an explicit list of (seed_idx, n)
    n_counts = (n_max - n_min + 1).clamp_min(0)
    if n_counts.sum().item() == 0:
        return (
            torch.empty((0, 3), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.int64),
        )
    seed_idx = torch.repeat_interleave(
        torch.arange(seed_y0.numel(), device=device, dtype=torch.int64),
        n_counts,
    )
    # n values: for seed i, n in [n_min[i], n_max[i]]
    cum = torch.cumsum(n_counts, dim=0) - n_counts
    local_n = (
        torch.arange(int(n_counts.sum().item()), device=device, dtype=torch.int64)
        - cum[seed_idx]
    )
    n_vals = n_min[seed_idx] + local_n

    pos = spot_to_unrotated_batch(
        xi=xi_t[seed_idx],
        yi=yi_t[seed_idx],
        zi=zi_t[seed_idx],
        ys=ys_t,
        zs=zs_t,
        y0=seed_y0[seed_idx],
        z0=seed_z0[seed_idx],
        step_size_in_x=step_size,
        n=n_vals,
        omega_deg=torch.tensor(omega_deg, device=device, dtype=dtype),
    )
    return pos, seed_idx
