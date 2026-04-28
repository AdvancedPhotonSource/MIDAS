"""Joint differentiable engine — fully end-to-end LM over geometry + per-region peak-shape parameters.

**Status (v0.1.0): scaffolded but not yet wired.**

The infrastructure is in place:

- ``midas_peakfit.lm_solve_arrowhead`` provides a Schur-complement-reduced LM
  solver for the joint  J = [J_dense | block_diag(J_block_k)]  problem (see
  the §13 plan).
- ``midas_calibrate.geometry_torch.pixel_to_REta_torch`` is a fully torch /
  autograd compatible forward model.
- ``midas_calibrate.refine`` already exercises lm_solve_generic with the
  geometry parameters as a 23-dim dense vector — the joint formulation just
  enlarges this to add per-(ring, η-bin) peak-shape blocks.

What's left (deferred to v0.2):

1. ``forward_cake(θ_geom, θ_shape)`` — predict the (R, η) cake intensity by
   summing pseudo-Voigts whose centers are determined by θ_geom (NOT by θ_shape).
2. Block-arrow Jacobian assembly: per-region 5×M peak-shape Jacobian +
   coupled geometry columns, packaged for ``lm_solve_arrowhead``.
3. Warm-start: 1-2 iterations of ``orchestrator.autocalibrate`` (alternating)
   followed by per-region peak-shape seeding.

For now ``autocalibrate_joint`` is a thin wrapper that delegates to the
alternating engine; users get the same end result through a more conventional
implementation.  Swap-in is local to this module — no caller changes needed
once the joint forward model lands.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .orchestrator import CalibrationResult, autocalibrate
from .params import CalibrationParams


def autocalibrate_joint(
    params: CalibrationParams,
    image: np.ndarray,
    *, dark: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> CalibrationResult:
    """Full differentiable end-to-end calibration.

    v0.1: delegates to the alternating engine.  See module docstring for the
    deferred-but-scoped roadmap to the true joint formulation.
    """
    if verbose:
        print("[joint] v0.1 stub — delegating to alternating engine")
    return autocalibrate(params, image, dark=dark, verbose=verbose)
