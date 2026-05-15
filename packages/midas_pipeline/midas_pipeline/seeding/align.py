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


def _read_per_scan_spots(layer_dir: Path, scan_idx: int) -> "np.ndarray | None":
    """Load ``InputAllExtraInfoFittingAll{scan_idx}.csv`` if present.

    Returns the (n_spots, 18) array or ``None`` when the file is absent
    or empty.
    """
    import numpy as np
    path = Path(layer_dir) / f"InputAllExtraInfoFittingAll{scan_idx}.csv"
    if not path.exists() or path.stat().st_size < 16:
        return None
    arr = np.loadtxt(path, skiprows=1, ndmin=2)
    return arr if arr.size else None


def _median_spot_centroid(spots: "np.ndarray") -> tuple[float, float]:
    """Median (y_det, z_det) of a per-scan spot table.

    Spot table columns follow the InputAllExtraInfoFittingAll layout
    from ``transforms`` stage:
        col 0: ID, col 1: omega(deg), col 2: y_det, col 3: z_det,
        col 4: ring, col 5: eta, ...
    Returns the median y_det and z_det across all spots in the scan.
    """
    import numpy as np
    y = np.median(spots[:, 2])
    z = np.median(spots[:, 3])
    return float(y), float(z)


def align_per_scan(
    *,
    layer_dir: str | Path,
    n_scans: int,
    method: str = "ring-center",
    reference_scan: int = -1,        # -1 → n_scans // 2
) -> List[AlignmentDiagnostics]:
    """Compute per-scan ``(ΔBC_y, ΔBC_z)`` corrections.

    Methods
    -------
    ``none``
        Returns zero corrections for every scan. Right for synthetic
        drift-free data (test_pf_hedm.py) or when alignment was done
        upstream (e.g. via midas-calibrate-v2 on the raw scan zarrs
        before this pipeline ran).

    ``centroid``
        Uses the per-scan ``InputAllExtraInfoFittingAll{i}.csv``
        spot lists (already produced upstream of seeding). For each
        scan, takes the median ``(y_det, z_det)`` over all spots and
        records the offset relative to the reference scan's median.
        This is a coarse, signal-driven estimate — useful when the
        drift is dominated by a translation of the entire diffraction
        pattern (e.g. thermal drift of the rotation-axis projection on
        the detector). It does NOT fit ring centers; for that you want
        ``ring-center`` once midas-calibrate-v2 is wired through.

    ``ring-center`` / ``cross-correlation``
        Not yet wired — both need a per-scan detector-image source
        (zarrs from the per-scan ``MIDAS.zip`` files) which this stage
        does not currently load. Use ``centroid`` as the production
        fallback when spot-list drift compensation is enough, or wait
        on the midas-calibrate-v2 integration for ring-center.

    Parameters
    ----------
    layer_dir : path
        Layer working directory containing per-scan spot files.
    n_scans : int
        Total scan count.
    method : "ring-center" | "cross-correlation" | "centroid" | "none"
    reference_scan : int
        Reference scan index; corrections are relative to this.
        ``-1`` resolves to ``n_scans // 2``.

    Returns
    -------
    list of :class:`AlignmentDiagnostics`, one per scan.
    """
    layer_dir = Path(layer_dir)
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

    if method == "centroid":
        import numpy as np
        ref_spots = _read_per_scan_spots(layer_dir, reference_scan)
        if ref_spots is None:
            raise FileNotFoundError(
                f"centroid alignment needs InputAllExtraInfoFittingAll"
                f"{reference_scan}.csv at {layer_dir} (the reference scan)."
            )
        ref_y, ref_z = _median_spot_centroid(ref_spots)
        out: List[AlignmentDiagnostics] = []
        for i in range(n_scans):
            if i == reference_scan:
                out.append(AlignmentDiagnostics(
                    scan_idx=i, delta_bc_y=0.0, delta_bc_z=0.0,
                    residual_px=0.0, method="centroid",
                ))
                continue
            spots = _read_per_scan_spots(layer_dir, i)
            if spots is None:
                # Missing scan → record zero correction + a clear
                # residual marker (NaN); merge_all decides how to handle.
                out.append(AlignmentDiagnostics(
                    scan_idx=i, delta_bc_y=0.0, delta_bc_z=0.0,
                    residual_px=float("nan"), method="centroid",
                ))
                continue
            y, z = _median_spot_centroid(spots)
            out.append(AlignmentDiagnostics(
                scan_idx=i,
                delta_bc_y=y - ref_y,
                delta_bc_z=z - ref_z,
                residual_px=float(np.hypot(y - ref_y, z - ref_z)),
                method="centroid",
            ))
        return out

    raise NotImplementedError(
        f"Alignment method {method!r} not yet wired up. Available: "
        "'none' (drift-free / pre-aligned data), 'centroid' (median "
        "spot-position drift from InputAllExtraInfoFittingAll*.csv). "
        "ring-center / cross-correlation need a per-scan detector-image "
        "source (zarrs) which this stage does not currently load — "
        "tracked as future midas-calibrate-v2 integration."
    )
