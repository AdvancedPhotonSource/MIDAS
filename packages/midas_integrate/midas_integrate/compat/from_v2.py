"""Adapt a ``midas_calibrate_v2`` calibration result for ``midas_integrate``.

Two concrete entry points:

- :func:`params_from_v2_unpacked` — take an unpacked v2 parameter dict
  (the ``unpacked`` field of ``PVCalibrationResult``) plus a v1 template
  carrying the non-refined fields (NrPixels*, RhoD, etc.) and produce
  an :class:`midas_integrate.params.IntegrationParams` ready for
  :func:`midas_integrate.detector_mapper.build_map`.

- :func:`params_from_calibration_spec` — same but takes a v2
  :class:`midas_calibrate_v2.parameters.spec.CalibrationSpec` directly.
  Useful at the auto-seed entry point where there is no v1 template.

Both routes apply the v2 → v1 distortion-name remap (``iso_R2`` → ``p2``
etc.) so the existing forward model in :mod:`midas_integrate.geometry`
keeps working unchanged.

What does NOT round-trip through this adapter:

- **Per-ring ``δr_k`` (F2 fix)** — ``integrate`` v1's radial map has no
  per-ring concept; the offset is dropped here.  Use
  :func:`midas_calibrate_v2.compat.to_integrate.write_per_ring_offsets_json`
  to emit a sidecar for downstream peak-fit / Rietveld tools.

- **Stage 4 thin-plate spline** — ``integrate`` v1 expects a binary
  per-pixel ΔR grid (:mod:`midas_integrate.residual_corr`).  Use
  :func:`midas_calibrate_v2.compat.to_integrate.write_residual_correction_from_spline`
  to evaluate the spline on the detector grid and write the binary
  format.  Then point ``IntegrationParams.ResidualCorrectionMap`` at
  the resulting file.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from midas_integrate.params import IntegrationParams


# v2 distortion canonical name → v1 p-index slot.
# Keep in sync with midas_calibrate_v2.compat.to_v1._V2_TO_V1_DISTORTION
# (single source of truth would import from v2 but we keep a copy here
# so this module is import-safe even if v2 is not installed at runtime).
_V2_TO_V1_DISTORTION: Dict[str, str] = {
    "iso_R2": "p2", "iso_R4": "p5", "iso_R6": "p4",
    "a1": "p7",  "phi1": "p8",
    "a2": "p0",  "phi2": "p6",
    "a3": "p9",  "phi3": "p10",
    "a4": "p1",  "phi4": "p3",
    "a5": "p11", "phi5": "p12",
    "a6": "p13", "phi6": "p14",
}


def _scalar(val: Any) -> float:
    """Extract a Python float from a torch tensor / numpy scalar / float."""
    if hasattr(val, "detach"):
        v = val.detach().cpu()
        if v.ndim == 0:
            return float(v.item())
        return float(v.reshape(-1)[0].item())
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def params_from_v2_unpacked(
    unpacked: Dict[str, Any],
    *,
    template: IntegrationParams,
    warn_on_dropped: bool = True,
) -> IntegrationParams:
    """Build an :class:`IntegrationParams` from a v2 unpacked dict.

    Parameters
    ----------
    unpacked :
        A v2 parameter dict — typically ``res.unpacked`` from
        ``autocalibrate_pv``, ``autocalibrate_four_stage`` etc.
        Tensor / numpy / scalar values all accepted.
    template :
        An :class:`IntegrationParams` whose non-refined fields
        (NrPixelsY/Z, RhoD, RBinSize, RMin/RMax, EtaBinSize, TransOpt,
        binning, residual-correction file path, etc.) are already set
        correctly.  Easiest to construct via
        :func:`midas_integrate.params.parse_params` from the same
        seed paramstest the v2 calibration started from.
    warn_on_dropped :
        Emit a UserWarning when ``unpacked`` carries a v2-only
        parameter (``delta_r_k``, panel blocks) that this adapter
        cannot put into an ``IntegrationParams``.

    Returns
    -------
    A new :class:`IntegrationParams` carrying the v2-converged geometry,
    distortion (remapped to v1 slots), Parallax, Wavelength, and pxY/pxZ.
    """
    from copy import deepcopy
    out = deepcopy(template)

    dropped = []
    for name, val in unpacked.items():
        # Skip per-panel blocks; they go to a separate panel-shifts file.
        if name in ("panel_delta_yz", "panel_delta_theta",
                    "panel_delta_lsd", "panel_delta_p2"):
            dropped.append(name)
            continue
        # Skip per-ring offsets — no v1 slot.
        if name == "delta_r_k":
            dropped.append(name)
            continue

        # Map v2 distortion names back to v1 p-indices.
        target = _V2_TO_V1_DISTORTION.get(name, name)
        if hasattr(out, target):
            try:
                setattr(out, target, _scalar(val))
            except Exception:
                pass    # silently keep template's value if cast fails
        # else: this v2 parameter has no v1 IntegrationParams slot.
        #       Wavelength / pxY / pxZ / Lsd / BC_y / BC_z / tx / ty / tz
        #       all have direct slots and pass through.

    if dropped and warn_on_dropped:
        warnings.warn(
            "v2 parameters not representable in IntegrationParams: "
            f"{dropped}. Per-panel shifts → use "
            "midas_calibrate_v2.compat.to_v1.write_panel_shifts_file. "
            "delta_r_k → use "
            "midas_calibrate_v2.compat.to_integrate.write_per_ring_offsets_json. "
            "The integration map itself is unaffected.",
            UserWarning, stacklevel=2,
        )
    return out


def params_from_calibration_spec(
    spec: Any,
    *,
    template: IntegrationParams,
    warn_on_dropped: bool = True,
) -> IntegrationParams:
    """Build an :class:`IntegrationParams` from a v2 :class:`CalibrationSpec`.

    Convenience wrapper for cases where you only have the spec (e.g.
    after :func:`first_time_calibrate`) and want the converged geometry.
    Reads each ``Parameter.init`` (which the v2 pipeline updates in
    place to the converged value at the end of every iteration) and
    forwards through :func:`params_from_v2_unpacked`.
    """
    unpacked = {}
    for name, p in spec.parameters.items():
        if not getattr(p, "refined", False):
            # Non-refined params already match the seed paramstest;
            # the template carries them.
            continue
        unpacked[name] = p.init
    return params_from_v2_unpacked(
        unpacked, template=template, warn_on_dropped=warn_on_dropped,
    )


__all__ = [
    "_V2_TO_V1_DISTORTION",
    "params_from_v2_unpacked",
    "params_from_calibration_spec",
]
