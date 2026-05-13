"""Spec-aware Levenberg-Marquardt wrapper.

Operates on any :class:`midas_peakfit.spec.ParameterSpec` (or subclass —
e.g. ``CalibrationSpec``) plus a residual closure that takes the unpacked
parameter dict.  Internally extracts the refined subset and delegates to
:func:`midas_peakfit.lm_solve_generic` with an autograd Jacobian.

Promoted from ``midas_calibrate_v2.inference.lm``: the wrapper has no
calibration-specific behaviour and is reused by HEDM grain refinement and
joint pipelines.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

from midas_peakfit.lm_generic import GenericLMConfig, lm_solve_generic
from midas_peakfit.reparam import u_to_x

from midas_peakfit.pack import (
    refined_indices, refined_bounds,
    write_refined_back, unpack_spec, pack_spec,
)
from midas_peakfit.spec import ParameterSpec


def lm_minimise(
    spec: ParameterSpec,
    residual_dict_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    *,
    config: GenericLMConfig = GenericLMConfig(),
    fallback_span: float = 1.0,
    use_jacfwd: bool = True,    # forward-mode Jacobian: N forward passes
                                # vs M backward passes for jacrev.  At our
                                # typical (M=4500, N=20) shapes, this is
                                # ~10-50× faster.  The earlier ``vmap+jacfwd``
                                # path had a sigmoid-reparam shape bug;
                                # we now use a direct (non-vmap) jacfwd
                                # for single-problem (B=1) LM.
    dtype=torch.float64, device="cpu",
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    """Run LM with a residual-vector closure operating on a parameter dict.

    Parameters
    ----------
    residual_dict_fn : ``unpacked -> [M] residuals``
        v2's residual closure — operates on the unpacked parameter dict, so
        adding a new refined parameter is automatic (no closure rewrites).

    Returns
    -------
    (final_unpacked, cost, rc)
    """
    x_full, info = pack_spec(spec, dtype=dtype, device=device)
    lo, hi = refined_bounds(spec, info, fallback_span=fallback_span,
                             dtype=dtype, device=device)
    refined_idx = refined_indices(info).to(device)
    x_ref0 = x_full.index_select(0, refined_idx)

    def res_closure(u, lo_, hi_):
        # u, lo_, hi_ all [B=1, N_ref]
        x_ref_now = u_to_x(u, lo_, hi_).squeeze(0)
        x_full_now = write_refined_back(x_full, x_ref_now, info)
        unpacked = unpack_spec(x_full_now, info, spec)
        r = residual_dict_fn(unpacked)
        return r.unsqueeze(0)

    jacobian_fn = None
    if use_jacfwd:
        # Forward-mode AD: N forward passes through the residual.
        # For our shapes (M ~ 1.5k–4.5k, N ~ 5–25) this is ~10-50× faster
        # than jacrev, which would do M backward passes.  No vmap needed
        # for single-problem (B=1) LM; that avoids the prior shape bug
        # with sigmoid reparam.
        #
        # We use ``has_aux=True`` to return both the function value and
        # the Jacobian from a single jacfwd call — saves one forward
        # pass per LM step compared to computing them separately.
        try:
            from torch.func import jacfwd
            def _jacfwd_jac(u, lo_, hi_):
                # u, lo_, hi_ are [1, N] (single-problem batch).
                lo_b = lo_.squeeze(0); hi_b = hi_.squeeze(0)
                def _f_with_aux(u_b):
                    r_b = res_closure(u_b.unsqueeze(0),
                                       lo_b.unsqueeze(0),
                                       hi_b.unsqueeze(0)).squeeze(0)
                    return r_b, r_b                       # primal, aux=primal
                J, r = jacfwd(_f_with_aux, has_aux=True)(u.squeeze(0))
                return r.unsqueeze(0), J.unsqueeze(0)      # [1, M], [1, M, N]
            jacobian_fn = _jacfwd_jac
        except Exception:
            jacobian_fn = None
    if jacobian_fn is None:
        # Fallback: jacrev (M backward passes).  Slower at our shapes but
        # always works.
        try:
            from torch.func import jacrev
            def _jacrev_jac(u, lo_, hi_):
                lo_b = lo_.squeeze(0); hi_b = hi_.squeeze(0)
                def _f(u_b):
                    return res_closure(u_b.unsqueeze(0),
                                         lo_b.unsqueeze(0),
                                         hi_b.unsqueeze(0)).squeeze(0)
                J = jacrev(_f)(u.squeeze(0))             # [M, N]
                r = res_closure(u, lo_, hi_)              # [1, M]
                return r, J.unsqueeze(0)                   # [1, M, N]
            jacobian_fn = _jacrev_jac
        except Exception:
            jacobian_fn = None
    x_final, cost, rc = lm_solve_generic(
        x_ref0.unsqueeze(0), lo.unsqueeze(0), hi.unsqueeze(0),
        residual_fn=res_closure, jacobian_fn=jacobian_fn, config=config,
    )
    x_ref_final = x_final.squeeze(0)
    x_full_final = write_refined_back(x_full, x_ref_final, info)
    unpacked_final = unpack_spec(x_full_final, info, spec)
    return unpacked_final, float(cost.item()), int(rc.item())


__all__ = ["lm_minimise"]
