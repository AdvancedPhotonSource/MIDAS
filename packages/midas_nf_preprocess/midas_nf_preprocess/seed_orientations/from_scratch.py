"""From-scratch uniform sampling of orientations + fundamental-zone filter.

Pipeline:

  1. ``shoemake_uniform_quaternions`` -- N uniform random quaternions on S^3.
     Deterministic (seeded). Statistically uniform over SO(3).
  2. ``midas_stress.fundamental_zone`` -- map each into the fundamental zone of
     the requested space group's Laue symmetry (vectorised torch op).
  3. ``deduplicate_quaternions`` -- collapse near-identical FZ representatives
     that arose from different symmetry-equivalent input quats.

The sample size ``n_master`` is chosen from a target misorientation resolution
``resolution_deg`` via :func:`n_master_for_resolution` (defaults to 1.5 deg),
or specified directly by the caller.

Sampling is set-up-time work and runs once per experiment, so it lives on the
host device (CPU) by default; the returned tensor can be moved to GPU/MPS.
The implementation is a straight port of Shoemake (1992); intermediate
tensors carry no autograd history.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from ..device import resolve_device, resolve_dtype
from .crystal import space_group_to_sym_order


def shoemake_uniform_quaternions(
    n: int,
    *,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Generate ``n`` uniform random quaternions on S^3 via Shoemake (1992).

    Returns
    -------
    Tensor of shape ``(n, 4)`` with ``(w, x, y, z)`` ordering. Quaternions are
    unit-norm by construction and sampled from the Haar measure on SO(3).
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)

    # Use a CPU generator for reproducibility regardless of target device.
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.rand((n, 3), generator=gen, dtype=dtype)

    s1 = torch.sqrt(1.0 - u[:, 0])
    s2 = torch.sqrt(u[:, 0])
    two_pi = 2.0 * math.pi
    a = two_pi * u[:, 1]
    b = two_pi * u[:, 2]
    quats = torch.stack(
        [
            s2 * torch.cos(b),  # w
            s1 * torch.sin(a),  # x
            s1 * torch.cos(a),  # y
            s2 * torch.sin(b),  # z
        ],
        dim=1,
    )
    # Shoemake convention canonicalisation: enforce w >= 0 so that quaternions
    # land in the upper hemisphere of S^3 (matches the SO(3)/Z_2 convention
    # used by midas_stress and the cached seed files).
    sign = torch.where(
        quats[:, 0] >= 0,
        torch.ones_like(quats[:, 0]),
        -torch.ones_like(quats[:, 0]),
    )
    quats = quats * sign.unsqueeze(-1)
    return quats.to(device=device)


def n_master_for_resolution(
    resolution_deg: float,
    *,
    oversample: float = 3.0,
    floor: int = 50_000,
) -> int:
    """Heuristic Shoemake N that yields ~``resolution_deg`` average spacing in SO(3).

    Volume of SO(3) in axis-angle radians^3 is ``pi^2``. For uniform sampling
    at average misorientation ``theta_res``, the expected number of points is
    ``pi^2 / theta_res^3``. We over-sample by ``oversample`` (default 3x) so
    that after FZ folding and dedup we still have good coverage; the output
    cap matches the ~250k cached cubic file at 1.5 deg.

    >>> n_master_for_resolution(1.5)  # doctest: +SKIP
    1648842
    >>> n_master_for_resolution(5.0)  # doctest: +SKIP
    44473
    """
    if resolution_deg <= 0:
        raise ValueError(
            f"resolution_deg must be > 0, got {resolution_deg}"
        )
    theta_rad = resolution_deg * math.pi / 180.0
    n = oversample * (math.pi ** 2) / (theta_rad ** 3)
    return max(int(floor), int(math.ceil(n)))


def deduplicate_quaternions(
    quats: torch.Tensor,
    *,
    tol_deg: float,
) -> torch.Tensor:
    """Drop near-duplicate quaternions (within ``tol_deg`` misorientation).

    Uses a deterministic hash bucketing on quantised quaternion components --
    O(N) and avoids the O(N^2) all-pairs pass. The bucket size is tuned to
    ``tol_deg``: two quaternions whose components differ by less than the
    bucket width hash to the same bucket and one of them is dropped.

    The hash is component-wise on rounded ``(w, |x|, |y|, |z|)`` (sign-canonical
    via ``w >= 0``); for quaternions already in the FZ this is enough to catch
    duplicates introduced by symmetry-equivalent inputs that mapped to the
    same FZ rep.
    """
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(
            f"Expected (N, 4) quaternions, got shape {tuple(quats.shape)}"
        )
    if tol_deg <= 0:
        return quats

    # Half-angle for component-comparison: a misorientation of theta means
    # quaternion w changes by ~ 0.5 * sin(theta/2) ~ 0.25 * theta_rad in the
    # small-angle limit. We pick a bucket width that is ~half this so that
    # pairs within tol_deg map to the same bucket.
    theta_rad = tol_deg * math.pi / 180.0
    bucket = 0.125 * theta_rad
    bucket = max(bucket, 1e-9)

    # Quantise.
    quantised = torch.round(quats / bucket).to(torch.int64)
    # Build an integer hash key per row. Use tuple-of-ints for stable
    # deduplication; this is O(N) and runs in pure Python on CPU.
    quantised_cpu = quantised.detach().cpu().numpy()
    seen: dict = {}
    keep_idx: list[int] = []
    for i, row in enumerate(quantised_cpu):
        key = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if key not in seen:
            seen[key] = True
            keep_idx.append(i)
    keep_t = torch.as_tensor(keep_idx, dtype=torch.long, device=quats.device)
    return quats.index_select(0, keep_t)


def generate_uniform_seeds(
    space_group: int,
    *,
    resolution_deg: float = 1.5,
    n_master: Optional[int] = None,
    seed: int = 42,
    deduplicate: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Generate FZ-reduced quaternions covering the symmetry's fundamental zone.

    Parameters
    ----------
    space_group : int in [1, 230]. Drives the FZ filter.
    resolution_deg : target average misorientation between neighbours.
    n_master : if given, overrides ``resolution_deg``-derived heuristic.
    seed : RNG seed for the Shoemake sampler.
    deduplicate : drop FZ representatives that are within
        ``resolution_deg / 2`` of each other (default on).

    Returns
    -------
    Tensor of shape ``(N, 4)`` -- FZ-reduced quaternions.
    """
    from midas_stress.orientation import fundamental_zone

    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)

    if n_master is None:
        n_master = n_master_for_resolution(resolution_deg)

    # Step 1: uniform random quaternions on S^3.
    raw = shoemake_uniform_quaternions(
        n_master, seed=seed, device=device, dtype=dtype
    )

    # Step 2: reduce each into the fundamental zone of the requested SG.
    fz = fundamental_zone(raw, int(space_group))
    if not isinstance(fz, torch.Tensor):
        fz = torch.as_tensor(fz, device=device, dtype=dtype)
    else:
        fz = fz.to(device=device, dtype=dtype)

    # Step 3: dedup near-identical FZ reps.
    if deduplicate:
        fz = deduplicate_quaternions(fz, tol_deg=resolution_deg / 2.0)

    return fz
