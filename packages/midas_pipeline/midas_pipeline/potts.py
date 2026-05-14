"""Confidence-weighted Potts (ICM) smoothing of a grain-ID map.

Port of ``FF_HEDM/workflows/pf_MIDAS.py:262 confidence_weighted_potts``,
with the inner double-``for`` sweep replaced by a numba-JIT'd kernel.
The pure-Python implementation is retained as a fallback for environments
without numba.

Per-voxel margin = ``(top1 - top2) / max(posterior)``. For voxel V with
margin ``m_V``, the data term is scaled by ``(m_V + conf_floor)``:

    ``cost(g | V) = (m_V + conf_floor) * unary(g, V) + lam * pair_cost(g | V)``

where ``unary(g, V) = -log(posterior(g, V) + eps)`` and ``pair_cost``
counts 4-connected neighbors with grain != g. High-margin voxels (clear
winner) → unary dominates → don't flip. Low-margin voxels (ambiguous) →
pair cost dominates → pulled toward neighborhood majority.

Voxels with ``max_id == -1`` (no evidence) are excluded from the
optimization and remain ``-1`` in the output.
"""

from __future__ import annotations

import logging

import numpy as np

LOG = logging.getLogger("midas_pipeline.potts")

try:
    from numba import jit

    _NUMBA_AVAILABLE = True

    @jit(nopython=True, cache=True)
    def _icm_sweep_numba(L, active, unary, weights, lam, n_grs, H, W):
        n_chg = 0
        L_new = L.copy()
        for r in range(H):
            for c in range(W):
                if not active[r, c]:
                    continue
                # collect up-to-4 neighbors
                nbr0 = -1
                nbr1 = -1
                nbr2 = -1
                nbr3 = -1
                nn = 0
                if r > 0 and active[r - 1, c]:
                    nbr0 = L[r - 1, c]; nn += 1
                if r < H - 1 and active[r + 1, c]:
                    if nn == 0: nbr0 = L[r + 1, c]
                    elif nn == 1: nbr1 = L[r + 1, c]
                    elif nn == 2: nbr2 = L[r + 1, c]
                    else: nbr3 = L[r + 1, c]
                    nn += 1
                if c > 0 and active[r, c - 1]:
                    if nn == 0: nbr0 = L[r, c - 1]
                    elif nn == 1: nbr1 = L[r, c - 1]
                    elif nn == 2: nbr2 = L[r, c - 1]
                    else: nbr3 = L[r, c - 1]
                    nn += 1
                if c < W - 1 and active[r, c + 1]:
                    if nn == 0: nbr0 = L[r, c + 1]
                    elif nn == 1: nbr1 = L[r, c + 1]
                    elif nn == 2: nbr2 = L[r, c + 1]
                    else: nbr3 = L[r, c + 1]
                    nn += 1
                if nn == 0:
                    continue
                # cost over all grains
                best_g = L[r, c]
                best_cost = 1e30
                w = weights[r, c]
                for g in range(n_grs):
                    pair_cost = 0
                    if nn > 0 and nbr0 != g: pair_cost += 1
                    if nn > 1 and nbr1 != g: pair_cost += 1
                    if nn > 2 and nbr2 != g: pair_cost += 1
                    if nn > 3 and nbr3 != g: pair_cost += 1
                    cost = w * unary[g, r, c] + lam * pair_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_g = g
                if best_g != L[r, c]:
                    L_new[r, c] = best_g
                    n_chg += 1
        return L_new, n_chg

except ImportError:  # pragma: no cover - numba is in pyproject deps
    _NUMBA_AVAILABLE = False
    _icm_sweep_numba = None


def _icm_sweep_python(L, active, unary, weights, lam, n_grs, H, W):
    """Pure-Python ICM sweep (fallback when numba isn't available)."""
    L_new = L.copy()
    n_chg = 0
    for r in range(H):
        for c in range(W):
            if not active[r, c]:
                continue
            nbr = []
            if r > 0 and active[r - 1, c]:
                nbr.append(L[r - 1, c])
            if r < H - 1 and active[r + 1, c]:
                nbr.append(L[r + 1, c])
            if c > 0 and active[r, c - 1]:
                nbr.append(L[r, c - 1])
            if c < W - 1 and active[r, c + 1]:
                nbr.append(L[r, c + 1])
            if not nbr:
                continue
            pair_cost = np.array(
                [sum(1 for n in nbr if n != g) for g in range(n_grs)]
            )
            cost = weights[r, c] * unary[:, r, c] + lam * pair_cost
            best = int(np.argmin(cost))
            if best != L[r, c]:
                L_new[r, c] = best
                n_chg += 1
    return L_new, n_chg


def confidence_weighted_potts(
    posterior: np.ndarray,
    max_id: np.ndarray,
    lam: float,
    *,
    max_iter: int = 30,
    conf_floor: float = 0.05,
    use_numba: bool = True,
) -> np.ndarray:
    """Confidence-weighted Potts (ICM) smoothing.

    Parameters
    ----------
    posterior : ndarray, shape (n_grains, H, W), float
        Bayesian posterior (shape × orient_score), pre-argmax.
    max_id : ndarray, shape (H, W), int32
        Initial grain assignment (e.g. argmax of posterior). -1 means
        no evidence.
    lam : float
        Pairwise penalty per disagreeing neighbor.
    max_iter : int, default 30
    conf_floor : float, default 0.05
        Minimum data weight (prevents fully-overriding any voxel
        by neighbors).
    use_numba : bool, default True
        If True and numba is importable, use the JIT'd inner sweep.
        Setting this to False forces the pure-Python path (useful for
        benchmarking and tests).

    Returns
    -------
    ndarray, shape (H, W), int32
        Smoothed grain-ID map. Inactive voxels remain ``-1``.
    """
    n_grs, H, W = posterior.shape
    eps = 1e-6
    full_max = posterior.max(axis=0)
    no_evidence = (full_max <= 0)
    active = ~no_evidence

    # Per-voxel margin → data weight
    sorted_p = np.sort(posterior, axis=0)[::-1]
    margin = sorted_p[0] - (sorted_p[1] if n_grs > 1 else 0.0)
    if posterior.max() > 0:
        margin = margin / posterior.max()
    weights = (margin + conf_floor).astype(np.float64)

    unary = (-np.log(posterior + eps)).astype(np.float64)
    if no_evidence.any():
        unary[:, no_evidence] = 0.0

    L = max_id.copy().astype(np.int32)
    L[L < 0] = 0

    sweep = (
        _icm_sweep_numba
        if (use_numba and _NUMBA_AVAILABLE and _icm_sweep_numba is not None)
        else _icm_sweep_python
    )

    n_changed_total = 0
    it = 0
    for it in range(max_iter):
        L_new, n_chg = sweep(L, active, unary, weights, lam, n_grs, H, W)
        L = L_new
        n_changed_total += n_chg
        if n_chg == 0:
            break
    L[no_evidence] = -1
    LOG.info(
        "  CW-Potts (λ=%g, conf_floor=%g): %d voxel flips across %d ICM passes",
        lam, conf_floor, n_changed_total, it + 1,
    )
    return L
