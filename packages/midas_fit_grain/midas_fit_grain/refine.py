"""Single-grain refinement entry point.

Public API: :func:`refine_grain`.

The grain state is parameterised as 12 scalars: ``(position[3],
euler[3], lattice[6])``. ``mode`` selects between

* ``"all_at_once"`` — one solver call over all 12 parameters; the
  observed↔predicted association is computed once at entry and held
  fixed (matches the user's spec — "if we fit everything together, we
  don't update spots").
* ``"iterative"`` — four sequential solver calls: position only,
  orientation only, strain only, joint polish. After each call, the
  spot association is recomputed against the updated state. This
  mirrors the C ``FitPosOrStrainsOMP`` default.

All three loss kinds (``pixel``, ``angular``, ``internal_angle``) are
supported via the same residual layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch

from midas_diffract import HEDMForwardModel  # type: ignore

from .config import FitConfig, LossKind
from .matching import MatchResult, associate, ring_slot_lookup
from .observations import ObservedSpots
from .residuals import grain_residuals
from .solvers import closure_kind, get_solver

DEG2RAD = math.pi / 180.0


@dataclass
class GrainFitResult:
    """Output of :func:`refine_grain`."""
    position: torch.Tensor          # (3,) um
    euler: torch.Tensor             # (3,) rad
    lattice: torch.Tensor           # (6,)  a,b,c,alpha,beta,gamma
    final_loss: float
    n_matched: int
    history: list[float]
    converged: bool
    match: MatchResult              # final association
    per_spot_residuals: torch.Tensor  # (S_matched, K_residual)


def _match_with_state(
    model: HEDMForwardModel,
    *,
    pos: torch.Tensor,
    euler: torch.Tensor,
    lattice: torch.Tensor,
    obs: ObservedSpots,
    obs_ring_slot: torch.Tensor,
    pred_ring_slot: torch.Tensor,
    omega_tolerance: float,
    eta_tolerance: float,
) -> MatchResult:
    """Recompute observed↔predicted association at the current state."""
    with torch.no_grad():
        spots = model(euler.view(1, 1, 3), pos.view(1, 1, 3),
                      lattice_params=lattice.view(1, 6))
        # Squeeze (B=1, N=1) leading dims
        def _sq(t):
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
                if t.dim() == 0:
                    break
            return t
        return associate(
            obs_ring_nr=obs.ring_nr,
            obs_omega=obs.omega,
            obs_eta=obs.eta,
            pred_ring_slot=pred_ring_slot,
            pred_omega=_sq(spots.omega),
            pred_eta=_sq(spots.eta),
            pred_valid=_sq(spots.valid),
            obs_ring_slot=obs_ring_slot,
            omega_tolerance=omega_tolerance,
            eta_tolerance=eta_tolerance,
        )


def _make_closures(
    *,
    model: HEDMForwardModel,
    obs: ObservedSpots,
    match: MatchResult,
    pos_scaled: torch.Tensor, pos_scale: float,
    euler: torch.Tensor, lattice: torch.Tensor,
    px: float, y_BC: float, z_BC: float,
    loss_kind: LossKind,
    active_params: list[torch.Tensor],
):
    """Build a dict of closure flavors keyed by solver-protocol name.

    Returns ``{kind: callable}`` for ``"scalar_with_backward"``,
    ``"scalar_no_backward"``, and ``"residual_no_backward"``. Each variant
    differs only in whether ``backward()`` is called and whether the loss
    or the un-summed residual is returned.
    """

    def _residual() -> torch.Tensor:
        pos = pos_scaled * pos_scale
        res = grain_residuals(
            model,
            grain_euler=euler,
            grain_position=pos,
            grain_lattice=lattice,
            obs=obs,
            match=match,
            kind=loss_kind,
            px=px, y_BC=y_BC, z_BC=z_BC,
        )
        return res

    def _scalar_loss(res: torch.Tensor) -> torch.Tensor:
        if res.numel() == 0:
            loss = torch.tensor(1e10, dtype=pos_scaled.dtype, device=pos_scaled.device)
        else:
            loss = (res * res).sum()
        # Make sure every active param touches the autograd graph, even
        # when the loss is independent of it (e.g. position fit under
        # angular/internal_angle losses — a no-op by design).
        nop = torch.zeros((), dtype=loss.dtype, device=loss.device)
        for p in active_params:
            nop = nop + 0.0 * p.sum()
        return loss + nop

    def closure_with_backward() -> torch.Tensor:
        for p in active_params:
            if p.grad is not None:
                p.grad.zero_()
        loss = _scalar_loss(_residual())
        loss.backward()
        return loss

    def closure_no_backward() -> torch.Tensor:
        with torch.no_grad():
            return _scalar_loss(_residual())

    def residual_no_backward() -> torch.Tensor:
        with torch.no_grad():
            return _residual().reshape(-1)

    return {
        "scalar_with_backward": closure_with_backward,
        "scalar_no_backward":   closure_no_backward,
        "residual_no_backward": residual_no_backward,
    }


def refine_grain(
    cfg: FitConfig,
    *,
    model: HEDMForwardModel,
    obs: ObservedSpots,
    init_position: torch.Tensor,    # (3,) um
    init_euler: torch.Tensor,       # (3,) rad
    init_lattice: torch.Tensor,     # (6,)
    pred_ring_slot: torch.Tensor,   # (M,) — ring-slot per reflection in model
    pos_scale: float = 100.0,       # internal rescale: pos_um = pos_scale * pos_param
    precomputed_match: MatchResult | None = None,
) -> GrainFitResult:
    """Refine one grain.

    The caller provides ``model`` and ``pred_ring_slot`` (built once for the
    full block) so per-grain work is minimal.

    Initial guess in radians for euler, micrometers for position, refined
    lattice constants for lattice.
    """
    device = init_position.device
    dtype = init_position.dtype

    # The optimizer parameters (`pos_scaled`, euler, lattice) are crafted so
    # all three have comparable gradient magnitudes. ``pos_scaled`` is in
    # units of ``pos_scale`` micrometers per unit; the closure converts back
    # via ``pos = pos_scaled * pos_scale``.
    pos_scaled = (init_position.clone().to(device=device, dtype=dtype) / pos_scale)
    euler = init_euler.clone().to(device=device, dtype=dtype)
    lattice = init_lattice.clone().to(device=device, dtype=dtype)
    pos_scaled.requires_grad_(False)
    euler.requires_grad_(False)
    lattice.requires_grad_(False)

    # Pre-compute the ring slot per observed spot (does not depend on state).
    obs_ring_slot = ring_slot_lookup(cfg.RingNumbers, obs.ring_nr)

    # Tolerances for re-association (radians).
    omega_tol = max(cfg.MarginOme, 2.0) * DEG2RAD
    eta_tol = max(cfg.MarginEta, 5.0) * DEG2RAD

    # Initial association.
    if precomputed_match is not None:
        match = precomputed_match
    else:
        match = _match_with_state(
            model, pos=pos_scaled * pos_scale, euler=euler, lattice=lattice,
            obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
            omega_tolerance=omega_tol, eta_tolerance=eta_tol,
        )

    solver_fn = get_solver(cfg.solver)

    # Helper: run one solver phase with a given active parameter set.
    histories: list[float] = []
    converged_phases: list[bool] = []

    kind = closure_kind(cfg.solver)

    def _run_phase(active: list[torch.Tensor], **solver_opts):
        for p in active:
            p.requires_grad_(True)
        closures = _make_closures(
            model=model, obs=obs, match=match,
            pos_scaled=pos_scaled, pos_scale=pos_scale,
            euler=euler, lattice=lattice,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
            loss_kind=cfg.loss,
            active_params=active,
        )
        opts = {"max_iter": cfg.max_iter, "ftol": cfg.ftol, "xtol": cfg.xtol}
        opts.update(solver_opts)        # caller wins
        result = solver_fn(closures[kind], active, **opts)
        for p in active:
            p.requires_grad_(False)
        histories.extend(result["history"])
        converged_phases.append(result["converged"])
        return result

    def _rematch():
        nonlocal match
        match = _match_with_state(
            model, pos=pos_scaled * pos_scale, euler=euler, lattice=lattice,
            obs=obs, obs_ring_slot=obs_ring_slot, pred_ring_slot=pred_ring_slot,
            omega_tolerance=omega_tol, eta_tolerance=eta_tol,
        )

    if cfg.mode == "all_at_once":
        _run_phase([pos_scaled, euler, lattice])
    elif cfg.mode == "iterative":
        ph_pos, ph_or, ph_lat, ph_joint = cfg.phase_steps
        _run_phase([pos_scaled], max_iter=ph_pos * 5 + 5)
        _rematch()
        _run_phase([euler], max_iter=ph_or * 5 + 5)
        _rematch()
        _run_phase([lattice], max_iter=ph_lat * 5 + 5)
        _rematch()
        # Final joint polish — no further re-match, per spec.
        _run_phase([pos_scaled, euler, lattice], max_iter=ph_joint * 5 + 5)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    pos_final = (pos_scaled * pos_scale).detach()

    # Final residuals at converged state.
    with torch.no_grad():
        res = grain_residuals(
            model,
            grain_euler=euler, grain_position=pos_final, grain_lattice=lattice,
            obs=obs, match=match, kind=cfg.loss,
            px=cfg.px, y_BC=model.y_BC, z_BC=model.z_BC,
        )
        loss_final = float((res * res).sum().item()) if res.numel() else float("inf")

    return GrainFitResult(
        position=pos_final,
        euler=euler.detach(),
        lattice=lattice.detach(),
        final_loss=loss_final,
        n_matched=int(match.mask.sum().item()),
        history=histories,
        converged=any(converged_phases),
        match=match,
        per_spot_residuals=res.detach(),
    )
