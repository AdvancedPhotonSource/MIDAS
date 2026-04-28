"""Powder-intensity helpers + an end-to-end B-factor refinement demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


DATA = Path(__file__).parent / "data"


@pytest.fixture
def torch_only():
    return pytest.importorskip("torch")


def test_lorentz_polarization_unpolarized(torch_only):
    import torch
    from midas_hkls import lorentz_polarization

    tt = torch.tensor([0.5, 1.0, 2.0])  # radians
    Lp = lorentz_polarization(tt, polarization=0.5)
    assert (Lp > 0).all()
    # finite
    assert torch.isfinite(Lp).all()


def test_powder_intensity_basic(torch_only):
    import torch
    from midas_hkls import (
        Atom, Crystal, Lattice, SpaceGroup, generate_hkls,
        powder_intensity, structure_factors,
    )

    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=5.4112)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[
        Atom("Ce", (0.0, 0.0, 0.0), B_iso=0.4),
        Atom("O", (0.25, 0.25, 0.25), B_iso=0.8),
    ])
    xt_t = xt.to_torch()
    refs = generate_hkls(sg, lat, wavelength_A=1.5418, two_theta_max_deg=80.0)

    F = structure_factors(xt_t, [(r.h, r.k, r.l) for r in refs])
    tt = torch.tensor([np.deg2rad(r.two_theta_deg) for r in refs], dtype=torch.float64)
    m = torch.tensor([r.multiplicity for r in refs], dtype=torch.float64)
    I = powder_intensity(F, m, tt)
    # All intensities non-negative (sanity)
    assert (I >= 0).all()


def test_intensity_from_crystal_one_shot(torch_only):
    import torch
    from midas_hkls import read_cif, generate_hkls, intensity_from_crystal

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch()
    refs = generate_hkls(xt.space_group, xt.lattice,
                         wavelength_A=1.5418, two_theta_max_deg=80.0)
    F, I = intensity_from_crystal(xt_t, refs, wavelength_A=1.5418)
    assert F.shape == I.shape == (len(refs),)
    assert torch.isfinite(I).all()
    assert (I >= 0).all()


def test_attach_intensities(torch_only):
    import torch
    from midas_hkls import read_cif, generate_hkls, intensity_from_crystal, attach_intensities

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch()
    refs = generate_hkls(xt.space_group, xt.lattice,
                         wavelength_A=1.5418, two_theta_max_deg=80.0)
    F, I = intensity_from_crystal(xt_t, refs, wavelength_A=1.5418)
    enriched = attach_intensities(refs, F, I)
    assert all(r.F_real is not None for r in enriched)
    assert all(r.intensity is not None and r.intensity >= 0 for r in enriched)


def test_b_factor_refinement_synthetic(torch_only):
    """End-to-end: synthesize CeO2 powder I(2θ) at known (B_Ce, B_O), then
    refit B by gradient descent on |F|² residual.  Sanity check for the
    differentiable pipeline."""
    import torch
    from midas_hkls import (
        Atom, Crystal, Lattice, SpaceGroup, generate_hkls,
        intensity_from_crystal,
    )

    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=5.4112)
    true_B = (0.40, 0.80)

    def make(B_ce, B_o):
        return Crystal(lattice=lat, space_group=sg, atoms=[
            Atom("Ce", (0.0, 0.0, 0.0), B_iso=B_ce),
            Atom("O", (0.25, 0.25, 0.25), B_iso=B_o),
        ])

    refs = generate_hkls(sg, lat, wavelength_A=1.5418, two_theta_max_deg=80.0)
    target_xt = make(*true_B).to_torch()
    _, I_target = intensity_from_crystal(target_xt, refs, wavelength_A=1.5418)
    I_target = I_target.detach()

    # Start from a wrong guess and refine.
    xt = make(0.05, 0.05).to_torch(requires_grad={"B_iso": True})
    opt = torch.optim.Adam([xt.B_iso_asu], lr=0.05)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        _, I = intensity_from_crystal(xt, refs, wavelength_A=1.5418)
        # Fit on log intensity to be scale-stable across orders of magnitude
        loss = ((torch.log(I + 1e-3) - torch.log(I_target + 1e-3)) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Loss should drop by orders of magnitude
    assert losses[-1] < losses[0] * 1e-2, f"loss didn't drop: {losses[0]} → {losses[-1]}"
    # Recovered B factors should be within 0.1 Å² of truth (Adam is approximate)
    fitted = xt.B_iso_asu.detach().tolist()
    assert abs(fitted[0] - true_B[0]) < 0.1, f"Ce: {fitted[0]} vs {true_B[0]}"
    assert abs(fitted[1] - true_B[1]) < 0.1, f"O: {fitted[1]} vs {true_B[1]}"


def test_lattice_refinement_synthetic(torch_only):
    """Refine lattice constant ``a`` from synthetic CeO2 intensities."""
    import torch
    from midas_hkls import (
        Atom, Crystal, Lattice, SpaceGroup, generate_hkls,
        intensity_from_crystal,
    )

    sg = SpaceGroup.from_number(225)
    a_true = 5.4112
    lat_true = Lattice.for_system("cubic", a=a_true)
    atoms = [
        Atom("Ce", (0.0, 0.0, 0.0), B_iso=0.4),
        Atom("O", (0.25, 0.25, 0.25), B_iso=0.8),
    ]
    refs = generate_hkls(sg, lat_true, wavelength_A=1.5418, two_theta_max_deg=80.0)
    target_xt = Crystal(lattice=lat_true, space_group=sg, atoms=atoms).to_torch()
    _, I_target = intensity_from_crystal(target_xt, refs, wavelength_A=1.5418)
    I_target = I_target.detach()

    # Start with wrong a, refine.
    a_wrong = 5.30
    lat_wrong = Lattice.for_system("cubic", a=a_wrong)
    xt = Crystal(lattice=lat_wrong, space_group=sg, atoms=atoms).to_torch(requires_grad={"lattice": True})
    opt = torch.optim.Adam([xt.lattice_params], lr=0.005)

    for _ in range(400):
        opt.zero_grad()
        _, I = intensity_from_crystal(xt, refs, wavelength_A=1.5418)
        loss = ((torch.log(I + 1e-3) - torch.log(I_target + 1e-3)) ** 2).mean()
        loss.backward()
        opt.step()

    a_fitted = xt.lattice_params[0].item()
    # Within 0.005 Å
    assert abs(a_fitted - a_true) < 0.005, f"a: {a_fitted} vs {a_true}"
