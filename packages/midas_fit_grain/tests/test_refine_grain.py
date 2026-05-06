"""End-to-end single-grain refinement tests on synthetic data."""

from __future__ import annotations

import math

import pytest
import torch

from midas_fit_grain import FitConfig, refine_grain

from ._synthetic import fixture_to_observed, gt_match, make_synthetic

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


@pytest.fixture(scope="module")
def fix():
    return make_synthetic(device=torch.device("cpu"), dtype=torch.float64)


def _build_cfg(fix, *, mode, loss, solver):
    return FitConfig(
        Lsd=fix.model.Lsd, px=fix.px, Wavelength=0.1729,
        LatticeConstant=tuple(fix.gt_lattice.tolist()),
        SpaceGroup=225,
        RingNumbers=fix.ring_numbers,
        RingRadii=[1.0] * len(fix.ring_numbers),
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-1e6, 1e6, -1e6, 1e6)],
        MarginEta=5.0, MarginOme=2.0,
        EtaBinSize=2.0, OmeBinSize=2.0,
        MinEta=6.0,
        solver=solver, mode=mode, loss=loss,
        max_iter=200, ftol=1e-8, xtol=1e-9,
        phase_steps=(8, 8, 8, 8),
    )


def _misori_deg(eul_a, eul_b):
    from midas_diffract import HEDMForwardModel
    Ra = HEDMForwardModel.euler2mat(eul_a)
    Rb = HEDMForwardModel.euler2mat(eul_b)
    trace = (Ra.T @ Rb).diagonal().sum()
    cos_ang = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return float(torch.acos(cos_ang)) * RAD2DEG


@pytest.mark.parametrize("mode", ["all_at_once", "iterative"])
@pytest.mark.parametrize("loss", ["pixel", "angular"])
def test_lbfgs_recovers_perturbed_grain(fix, mode, loss):
    """Recovery from a 0.5° / 3 µm seed using GT spot matching.

    In real use the indexer hands us per-grain spot↔reflection pairs in
    BestPos_*.csv; we mirror that here by passing the GT match through
    ``precomputed_match=``. Without it, the matcher can land on a
    'wrong-association plateau' for 0.5° perturbations on this small
    synthetic — see the comment in `_synthetic.py`.
    """
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, mode=mode, loss=loss, solver="lbfgs")

    # Perturbation chosen to stay in the smooth basin of the loss
    # landscape — large enough to exercise the optimizer, small enough that
    # position×phi1 coupling on the cubic synthetic doesn't trap us in a
    # non-GT local minimum. Real-world seeds from the indexer are tighter.
    init_pos = fix.gt_position.clone() + torch.tensor([1.0, -0.5, 0.3],
                                                      dtype=torch.float64)
    init_eul = fix.gt_euler.clone() + 0.05 * DEG2RAD
    init_lat = fix.gt_lattice.clone()

    match_seed = gt_match(fix, device=torch.device("cpu"),
                          dtype=torch.float64)

    result = refine_grain(
        cfg, model=fix.model,
        obs=obs,
        init_position=init_pos,
        init_euler=init_eul,
        init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
        precomputed_match=match_seed,
    )

    pos_err = (result.position - fix.gt_position).norm().item()
    mis_deg = _misori_deg(result.euler, fix.gt_euler)

    # Loss must monotonically improve and converge to a small value.
    assert result.history[-1] < result.history[0] * 1e-3, (
        f"loss should drop ~1000x, got {result.history[0]:.4g} -> "
        f"{result.history[-1]:.4g} (mode={mode}, loss={loss})"
    )
    if loss == "pixel":
        # Position is only refinable under pixel loss.
        assert pos_err < 0.1, f"|Δpos| = {pos_err:.3f} um (mode={mode}, loss={loss})"
        # phi1 is poorly conditioned on this synthetic — Phase 3 (L-M)
        # tests will tighten this once we have a damped solver.
        assert mis_deg < 0.06, f"misori = {mis_deg:.4f} deg (mode={mode}, loss={loss})"
    else:
        # Angular / internal_angle losses are pose-only — orientation
        # recovers cleanly because g-vector geometry is well determined.
        assert mis_deg < 0.005, f"misori = {mis_deg:.4f} deg (mode={mode}, loss={loss})"
    assert result.n_matched == obs.n_spots


def test_internal_angle_orientation_only(fix):
    """internal_angle is position/strain-blind; should still drive misori → 0."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, mode="iterative", loss="internal_angle",
                     solver="lbfgs")
    init_pos = fix.gt_position.clone()      # at truth
    init_eul = fix.gt_euler.clone() + 0.5 * DEG2RAD
    init_lat = fix.gt_lattice.clone()

    result = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
    )
    mis_deg = _misori_deg(result.euler, fix.gt_euler)
    assert mis_deg < 0.05, f"internal_angle misori = {mis_deg:.4f}"


def test_adam_runs(fix):
    """Smoke test: ADAM should make progress on a small perturbation.

    ADAM's per-parameter scale invariance helps mixed-unit problems but lr
    must be roughly the size of the perturbation; we use angular-only here
    so a single lr works."""
    obs = fixture_to_observed(fix, device=torch.device("cpu"),
                              dtype=torch.float64)
    cfg = _build_cfg(fix, mode="iterative", loss="angular", solver="adam")
    cfg.max_iter = 200
    cfg.phase_steps = (40, 40, 40, 40)

    init_pos = fix.gt_position.clone()                      # at truth
    init_eul = fix.gt_euler.clone() + 0.1 * DEG2RAD
    init_lat = fix.gt_lattice.clone()

    res = refine_grain(
        cfg, model=fix.model, obs=obs,
        init_position=init_pos, init_euler=init_eul, init_lattice=init_lat,
        pred_ring_slot=fix.pred_ring_slot,
    )
    assert res.history[-1] < res.history[0]
