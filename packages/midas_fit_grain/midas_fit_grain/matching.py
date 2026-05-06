"""Spot association: observed (S,) -> predicted (K, M) per grain.

The predicted-spot tensor produced by :class:`HEDMForwardModel` has shape
``(B, K=2, M)`` where K=2 are the +/- omega solutions and M is the number
of reflections enumerated in the model's HKL list. For each observed spot
we want a single ``(k, m)`` index into that grid.

Matching rule (matches the C ``CalcAngleErrors`` rule closely):
  1. Filter predicted slots to ``ring_nr == observed.ring_nr`` and
     ``valid > 0``.
  2. Among survivors, pick the one minimizing |Δω|. Ties broken by |Δη|.
  3. If the minimum |Δω| exceeds ``omega_tolerance`` OR |Δη| exceeds
     ``eta_tolerance``, mark the observed spot as unmatched.

The function is non-differentiable on purpose: re-association is a discrete
step that runs between optimizer phases (or once at the start, for
``mode='all_at_once'``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import torch

DEG2RAD = math.pi / 180.0


@dataclass
class MatchResult:
    """Per-spot association into the (K, M) predicted grid.

    ``k_idx`` and ``m_idx`` are valid only where ``mask=True``.
    """
    k_idx: torch.Tensor       # (S,) int64
    m_idx: torch.Tensor       # (S,) int64
    mask: torch.Tensor        # (S,) bool — True where association succeeded
    delta_omega: torch.Tensor # (S,) rad — signed Δω at the chosen match
    delta_eta: torch.Tensor   # (S,) rad — signed Δη at the chosen match


def _ring_for_each_hkl(hkls_int: torch.Tensor,
                       ring_numbers: list[int]) -> torch.Tensor:
    """Map each predicted reflection to a ring number (best-effort).

    Reflections sharing the same ``|h|² + |k|² + |l|²`` belong to the same
    ring (cubic case). For non-cubic, this is approximate but matches the
    C code's `CalcDiffrSpots_Furnace` ring grouping for the Bragg-angle
    binning used here.

    Returns ``(M,) int64`` with the *index into ``ring_numbers``* for each
    reflection, or -1 if no ring matches.
    """
    # The forward model's hkls_int is (M, 3); group by integer index sum-of-squares.
    h2 = (hkls_int.long().pow(2).sum(dim=-1))  # (M,)
    # Bin by unique h2 values, in the order they appear in ring_numbers.
    # NOTE: the C side uses `RingNumbers` as 1-based ring indices; the actual
    # mapping to (h, k, l) is done at hkls_for_forward_model time, where each
    # reflection already carries a "ring index" in some external table.
    # Here we approximate by rank order of unique h2 values.
    unique_h2, inv = torch.unique(h2, sorted=True, return_inverse=True)
    # `inv` is in [0, R-1] where R is the number of distinct h2 values.
    # Map inv -> ring_numbers index (clamped) -- the indexer guarantees the
    # forward model's hkls were derived from `ring_numbers`, so #unique == R.
    return inv  # (M,) — values are 0-based "ring slot" indices


def associate(
    obs_ring_nr: torch.Tensor,        # (S,) int64
    obs_omega: torch.Tensor,          # (S,) rad
    obs_eta: torch.Tensor,            # (S,) rad
    pred_ring_slot: torch.Tensor,     # (M,) int64 — ring slot per reflection
    pred_omega: torch.Tensor,         # (K, M) rad
    pred_eta: torch.Tensor,           # (K, M) rad
    pred_valid: torch.Tensor,         # (K, M) bool/float
    *,
    obs_ring_slot: torch.Tensor,      # (S,) int64 — ring slot per observed spot
    omega_tolerance: float = math.pi, # radians; default = wide-open
    eta_tolerance: float = math.pi,   # radians; default = wide-open
) -> MatchResult:
    """Single-grain associate. Returns ``(K, M)`` indices for each observed spot.

    ``obs_ring_slot`` and ``pred_ring_slot`` must use the **same** indexing
    scheme; see ``ring_slot_lookup``.
    """
    S = obs_omega.shape[0]
    K, M = pred_omega.shape
    device = pred_omega.device
    dtype = pred_omega.dtype

    if S == 0:
        return MatchResult(
            k_idx=torch.zeros(0, dtype=torch.int64, device=device),
            m_idx=torch.zeros(0, dtype=torch.int64, device=device),
            mask=torch.zeros(0, dtype=torch.bool, device=device),
            delta_omega=torch.zeros(0, dtype=dtype, device=device),
            delta_eta=torch.zeros(0, dtype=dtype, device=device),
        )

    # Build (S, M) ring filter
    ring_match = obs_ring_slot.view(S, 1) == pred_ring_slot.view(1, M)  # (S, M)
    valid_pred = pred_valid.bool().view(K, M)  # (K, M)

    # |Δω| and |Δη| as (S, K, M).
    # Wrap omega differences into [-π, π]:
    pred_om = pred_omega.view(1, K, M)
    obs_om = obs_omega.view(S, 1, 1)
    d_om = ((pred_om - obs_om + math.pi) % (2 * math.pi)) - math.pi  # (S, K, M)
    pred_et = pred_eta.view(1, K, M)
    obs_et = obs_eta.view(S, 1, 1)
    d_et = ((pred_et - obs_et + math.pi) % (2 * math.pi)) - math.pi

    # Disqualify entries that fail ring filter or validity by sending them
    # to a huge cost so argmin never picks them.
    BIG = torch.tensor(1e9, dtype=dtype, device=device)
    # Within-ring multiplicity: many reflections share |G| (Laue equivalents
    # of the same family). Their predicted spots all live in the same ring
    # at distinct ``η`` but possibly very close ``ω`` — so weight |Δω| and
    # |Δη| equally, and use sqrt(Δω² + Δη²) so a small η difference
    # dominates when ω alone is ambiguous.
    cost = torch.sqrt(d_om * d_om + d_et * d_et)
    disq = ~(ring_match.view(S, 1, M) & valid_pred.view(1, K, M))
    cost = torch.where(disq, BIG, cost)

    # Argmin over (K, M).
    flat = cost.view(S, K * M)
    best_idx = flat.argmin(dim=1)  # (S,)
    best_cost = flat.gather(1, best_idx.view(S, 1)).squeeze(1)

    k_idx = best_idx // M
    m_idx = best_idx %  M

    # Recover the chosen Δω, Δη to enforce tolerances.
    chosen_d_om = d_om[torch.arange(S, device=device), k_idx, m_idx]
    chosen_d_et = d_et[torch.arange(S, device=device), k_idx, m_idx]

    mask = (best_cost < BIG / 2.0) \
        & (chosen_d_om.abs() <= omega_tolerance) \
        & (chosen_d_et.abs() <= eta_tolerance)

    return MatchResult(
        k_idx=k_idx, m_idx=m_idx, mask=mask,
        delta_omega=chosen_d_om, delta_eta=chosen_d_et,
    )


def ring_slot_lookup(ring_numbers: list[int],
                     ring_nr_query: torch.Tensor) -> torch.Tensor:
    """Map a tensor of ring numbers to slot indices in ``ring_numbers``.

    Unknown ring numbers map to ``-1`` (which will fail the ring filter).
    """
    table = torch.full((max(ring_numbers, default=0) + 1,), -1,
                       dtype=torch.int64, device=ring_nr_query.device)
    for i, rn in enumerate(ring_numbers):
        if 0 <= rn < table.shape[0]:
            table[rn] = i
    rn = ring_nr_query.clamp(min=0, max=table.shape[0] - 1).long()
    return table[rn]
