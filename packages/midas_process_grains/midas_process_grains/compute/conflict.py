"""Phase 3 — per-hkl SpotID conflict resolution within a Phase-2 grain.

For each grain candidate from :mod:`refine_cluster`, walk every member and
collect their *aligned* claims at every theoretical hkl row. Resolve
conflicts to produce a single (hkl → SpotID) table with at most one observed
SpotID per hkl — the physical invariant a single grain must satisfy.

Resolution policies
-------------------

``"vote_then_residual"`` (default):
    1. If only one distinct SpotID is claimed at this row, take it.
    2. Else: count claims per distinct SpotID; if there's a strict majority,
       take it.
    3. Else (tie, or no claim): break by smallest |Δω| residual among the
       claims of each candidate SpotID; pick the candidate with the smallest
       median |Δω|.

``"forward_sim"`` (optional, gated by a CLI flag):
    Falls back to a forward-simulation arbitration when the vote ties. Asks
    ``midas_diffract.HEDMForwardModel`` for the predicted (y, z, ω) at the
    rep's orientation/position/lattice for this hkl, then picks the SpotID
    whose observed centroid is closest. This is expensive, so it's reserved
    for genuinely ambiguous cases; emit-rate is bounded.

Output is the per-grain SpotMatrix-ready row set: ``(GrainID, SpotID, …)``
with one row per resolved hkl. Down-stream the pipeline pulls the per-spot
fields from ``FitBest.bin``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "ResolvedClaim",
    "PerHklClaim",
    "resolve_conflicts",
]


@dataclass(frozen=True)
class PerHklClaim:
    """A single (member, SpotID, residual) triple for one theoretical hkl."""

    member_idx: int
    spot_id: int
    delta_omega: float


@dataclass(frozen=True)
class ResolvedClaim:
    """The output of conflict resolution for one (grain, hkl) pair.

    Attributes
    ----------
    hkl_row : int
        Row index in the canonical hkl table.
    spot_id : int
        Winning SpotID.
    n_supporters : int
        Number of cluster members whose aligned claim at this row was
        ``spot_id``.
    n_total_claims : int
        Total members that claimed something non-zero at this row.
    delta_omega : float
        Median |Δω| residual of the winning SpotID's supporting claims.
    policy_used : str
        Which branch of the policy decided this row (``"unique"``,
        ``"majority"``, ``"residual_tie"``, ``"forward_sim"``).
    """

    hkl_row: int
    spot_id: int
    n_supporters: int
    n_total_claims: int
    delta_omega: float
    policy_used: str


# ---------------------------------------------------------------------------
# Vote-then-residual policy
# ---------------------------------------------------------------------------


def _aggregate_per_hkl(
    aligned_col0: np.ndarray,
    aligned_col1: np.ndarray,
) -> Dict[int, List[PerHklClaim]]:
    """Build a map ``hkl_row → list[PerHklClaim]`` from the aligned tables.

    Parameters
    ----------
    aligned_col0 : np.ndarray
        ``(n_members, n_hkls)`` int64. Aligned SpotID column.
    aligned_col1 : np.ndarray
        ``(n_members, n_hkls)`` float64. Aligned Δω column. Same shape; rows
        permuted in lockstep with ``aligned_col0``.
    """
    n_members, n_hkls = aligned_col0.shape
    out: Dict[int, List[PerHklClaim]] = {}
    for k in range(n_hkls):
        nz_mem = np.flatnonzero(aligned_col0[:, k])
        if nz_mem.size == 0:
            continue
        out[k] = [
            PerHklClaim(
                member_idx=int(m),
                spot_id=int(aligned_col0[m, k]),
                delta_omega=float(aligned_col1[m, k]),
            )
            for m in nz_mem
        ]
    return out


def _resolve_one_hkl_vote_then_residual(
    hkl_row: int,
    claims: Sequence[PerHklClaim],
) -> ResolvedClaim:
    """Apply the vote-then-residual policy to one hkl's claims."""
    distinct = {c.spot_id for c in claims}
    n_total = len(claims)

    if len(distinct) == 1:
        sid = next(iter(distinct))
        sup = [c for c in claims if c.spot_id == sid]
        return ResolvedClaim(
            hkl_row=hkl_row,
            spot_id=sid,
            n_supporters=len(sup),
            n_total_claims=n_total,
            delta_omega=float(np.median([abs(c.delta_omega) for c in sup])),
            policy_used="unique",
        )

    counter = Counter(c.spot_id for c in claims)
    most_common = counter.most_common()
    top_count = most_common[0][1]
    top_ids = [sid for sid, cnt in most_common if cnt == top_count]
    if len(top_ids) == 1:
        sid = top_ids[0]
        sup = [c for c in claims if c.spot_id == sid]
        return ResolvedClaim(
            hkl_row=hkl_row,
            spot_id=sid,
            n_supporters=len(sup),
            n_total_claims=n_total,
            delta_omega=float(np.median([abs(c.delta_omega) for c in sup])),
            policy_used="majority",
        )

    # Tie among top_ids — break on smallest median |Δω|.
    best_sid = -1
    best_med = float("inf")
    best_sup: List[PerHklClaim] = []
    for sid in top_ids:
        sup = [c for c in claims if c.spot_id == sid]
        med = float(np.median([abs(c.delta_omega) for c in sup]))
        if med < best_med:
            best_med = med
            best_sid = sid
            best_sup = sup
    return ResolvedClaim(
        hkl_row=hkl_row,
        spot_id=int(best_sid),
        n_supporters=len(best_sup),
        n_total_claims=n_total,
        delta_omega=best_med,
        policy_used="residual_tie",
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def resolve_conflicts(
    aligned_col0: np.ndarray,
    aligned_col1: np.ndarray,
    *,
    policy: str = "vote_then_residual",
    forward_sim_fn: Optional[Callable[[int, Sequence[PerHklClaim]], int]] = None,
) -> List[ResolvedClaim]:
    """Resolve a grain's per-hkl SpotID claims.

    Parameters
    ----------
    aligned_col0 : np.ndarray
        ``(n_members, n_hkls)`` int64; aligned SpotID column.
    aligned_col1 : np.ndarray
        ``(n_members, n_hkls)`` float64; aligned Δω column.
    policy : {"vote_then_residual", "forward_sim"}
        Conflict-resolution policy.
    forward_sim_fn : callable, optional
        Required when ``policy="forward_sim"`` and a tie is unbreakable by
        residual. Signature: ``(hkl_row, claims) -> winning_spot_id``.

    Returns
    -------
    list[ResolvedClaim]
        One per resolved hkl row, in ascending row-index order. Rows where
        no member made a claim are omitted.
    """
    if policy not in {"vote_then_residual", "forward_sim"}:
        raise ValueError(
            f"policy must be vote_then_residual or forward_sim; got {policy!r}"
        )
    if aligned_col0.shape != aligned_col1.shape:
        raise ValueError(
            f"col0 and col1 must have the same shape; "
            f"got {aligned_col0.shape} and {aligned_col1.shape}"
        )

    by_hkl = _aggregate_per_hkl(aligned_col0, aligned_col1)
    out: List[ResolvedClaim] = []
    for k in sorted(by_hkl.keys()):
        claims = by_hkl[k]
        resolved = _resolve_one_hkl_vote_then_residual(k, claims)
        if (
            policy == "forward_sim"
            and resolved.policy_used == "residual_tie"
            and forward_sim_fn is not None
        ):
            forced = forward_sim_fn(k, claims)
            sup = [c for c in claims if c.spot_id == forced]
            if sup:
                resolved = ResolvedClaim(
                    hkl_row=k,
                    spot_id=int(forced),
                    n_supporters=len(sup),
                    n_total_claims=len(claims),
                    delta_omega=float(
                        np.median([abs(c.delta_omega) for c in sup])
                    ),
                    policy_used="forward_sim",
                )
        out.append(resolved)
    return out
