"""Stage A of merged-FF seeding: per-scan rotation-axis alignment.

Per the merged-FF plan §4 (``implementation_plan_pf_merged_ff_seeding.md``):
mechanical / thermal drift over a long pf scan shifts the rotation-
axis projection on the detector by sub-pixel to a few pixels between
scans. Without correction, the same physical reflection from the same
grain seen across N scans appears at slightly different
(Y, Z, η, 2θ) on each scan; the merge step's tolerance-based dedup
then inflates the merged spot list with near-duplicates.

This module fits one or more Debye-Scherrer ring centers per scan and
records ``(ΔBC_y, ΔBC_z)`` corrections. The actual ring-center fitter
lives in ``midas-calibrate-v2`` (pyFAI-based) — this module just orchestrates
the per-scan fit + diagnostics emit.

**Status**: scaffold. Real implementation lands when
midas-calibrate-v2 integration is wired up. Currently raises
``NotImplementedError`` from the orchestrator entry point; callers
should pre-correct spot positions if drift compensation is needed.

The module is here so the merged-FF pipeline has a clear seam for
the alignment step — when align is unimplemented, the merged-FF
``run_merged_ff_seeding`` flow either skips alignment (assumes data
is already aligned) or raises with a helpful message.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AlignmentDiagnostics:
    """Per-scan alignment record."""

    scan_idx: int
    delta_bc_y: float      # detector Y shift (µm)
    delta_bc_z: float      # detector Z shift (µm)
    residual_px: float     # post-correction residual
    method: str            # "ring-center" | "cross-correlation" | "none"


def align_per_scan(
    *,
    layer_dir: str | Path,
    n_scans: int,
    method: str = "ring-center",
    reference_scan: int = -1,        # -1 → n_scans // 2
) -> List[AlignmentDiagnostics]:
    """Compute per-scan ``(ΔBC_y, ΔBC_z)`` corrections.

    Parameters
    ----------
    layer_dir : path
        Layer working directory containing per-scan zarrs / spot files.
    n_scans : int
        Total scan count.
    method : "ring-center" | "cross-correlation" | "none"
        Alignment algorithm. ``none`` skips alignment (returns zero
        corrections — useful for synthetic data where drift = 0).
    reference_scan : int
        Reference scan index; corrections are computed relative to
        this. ``-1`` resolves to ``n_scans // 2`` (the canonical
        choice per the user's locked decision).

    Returns
    -------
    list of :class:`AlignmentDiagnostics`, one per scan.

    Notes
    -----
    The ``ring-center`` and ``cross-correlation`` methods are not yet
    implemented in this package; they need midas-calibrate-v2 + a
    detector-image source. Use ``method="none"`` for runs where the
    forward simulator already enforces zero drift (which is the case
    for the test_pf_hedm.py synthetic).
    """
    if reference_scan < 0:
        reference_scan = n_scans // 2

    if method == "none":
        return [
            AlignmentDiagnostics(
                scan_idx=i, delta_bc_y=0.0, delta_bc_z=0.0,
                residual_px=0.0, method="none",
            )
            for i in range(n_scans)
        ]

    raise NotImplementedError(
        f"Alignment method {method!r} not yet wired up. Use "
        "method='none' for synthetic-drift-free runs, or implement "
        "midas-calibrate-v2 integration here. See plan §4 "
        "(implementation_plan_pf_merged_ff_seeding.md) for the "
        "ring-center contract."
    )
