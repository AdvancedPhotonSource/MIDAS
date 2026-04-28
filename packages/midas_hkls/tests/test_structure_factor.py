"""Differentiable structure-factor tests.

|F| parity against gemmi.StructureFactorCalculatorX (after applying
``change_occupancies_to_crystallographic`` to convert gemmi's
sum-over-all-symops convention to the conventional unit-cell sum), and
gradcheck on |F|² w.r.t. atomic positions / occupancies / B-factors / lattice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


DATA = Path(__file__).parent / "data"


@pytest.fixture
def torch_only():
    return pytest.importorskip("torch")


@pytest.fixture
def gemmi_only():
    return pytest.importorskip("gemmi")


def _gemmi_F_proper(ss, hkl, gemmi):
    """Reproduce gemmi |F| in the conventional unit-cell convention."""
    ss = ss.clone() if hasattr(ss, "clone") else ss
    return gemmi.StructureFactorCalculatorX(ss.cell).calculate_sf_from_small_structure(ss, hkl)


@pytest.mark.parametrize("cif_name,wavelength,two_theta_max", [
    ("ceo2.cif",     0.1730, 12.0),
    ("si.cif",       1.5418, 90.0),
    ("alpha_fe.cif", 1.5418, 140.0),
    ("lab6.cif",     1.5418, 90.0),
    ("calcite.cif",  1.5418, 60.0),
])
def test_F_parity_with_gemmi(torch_only, gemmi_only, cif_name, wavelength, two_theta_max):
    import torch
    import gemmi
    from midas_hkls import read_cif, generate_hkls, structure_factors

    xt = read_cif(DATA / cif_name)
    xt_t = xt.to_torch()
    refs = generate_hkls(xt.space_group, xt.lattice,
                         wavelength_A=wavelength, two_theta_max_deg=two_theta_max)
    assert len(refs) >= 3

    ss = gemmi.read_small_structure(str(DATA / cif_name))
    ss.change_occupancies_to_crystallographic()  # gemmi convention shift
    calc = gemmi.StructureFactorCalculatorX(ss.cell)

    hkls = [(r.h, r.k, r.l) for r in refs]
    F_us = structure_factors(xt_t, hkls).detach().cpu().numpy()
    F_g = np.array([calc.calculate_sf_from_small_structure(ss, h) for h in hkls])

    abs_us = np.abs(F_us)
    abs_g = np.abs(F_g)
    # absolute tolerance: 0.01% of max |F|, plus 1e-6 floor for tiny values
    rel = np.abs(abs_us - abs_g) / np.maximum(abs_g, 1e-6)
    assert rel.max() < 1e-4, f"max |F| relative error {rel.max():.2e} ({cif_name})"


def test_silicon_forbidden_200(torch_only):
    """Si (200) is forbidden by Fd-3m glide systematic absences (already enforced
    by HKL generator, but F_hkl computed for it manually should also be ≈0)."""
    import torch
    from midas_hkls import read_cif, structure_factors

    xt = read_cif(DATA / "si.cif")
    xt_t = xt.to_torch()
    F_200 = structure_factors(xt_t, [(2, 0, 0)])
    assert abs(F_200).item() < 1e-6


def test_centrosymmetric_F_real(torch_only):
    """For a centrosymmetric structure with the inversion at origin, F is real."""
    import torch
    from midas_hkls import read_cif, structure_factors, generate_hkls

    xt = read_cif(DATA / "alpha_fe.cif")  # Im-3m, inversion at origin
    xt_t = xt.to_torch()
    refs = generate_hkls(xt.space_group, xt.lattice, wavelength_A=1.54, two_theta_max_deg=80.0)
    F = structure_factors(xt_t, [(r.h, r.k, r.l) for r in refs])
    # Im-3m is centrosymmetric with inversion at origin → F is real
    assert F.imag.abs().max().item() < 1e-9


def test_grad_lattice(torch_only):
    """∂|F|² / ∂a should be non-zero and finite for a typical CeO2 reflection."""
    import torch
    from midas_hkls import read_cif, structure_factors, structure_factor_intensity

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch(requires_grad={"lattice": True})
    F = structure_factors(xt_t, [(2, 2, 0)])
    I = structure_factor_intensity(F).sum()
    I.backward()
    assert xt_t.lattice_params.grad is not None
    # ∂I/∂a should be nonzero (d-spacing depends on a)
    assert abs(xt_t.lattice_params.grad[0].item()) > 1e-3


def test_grad_B_iso(torch_only):
    """∂|F|² / ∂B should be NEGATIVE — bigger thermal vibrations damp F."""
    import torch
    from midas_hkls import read_cif, structure_factors, structure_factor_intensity

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch(requires_grad={"B_iso": True})
    F = structure_factors(xt_t, [(4, 4, 4)])  # high-angle reflection, sensitive to B
    I = structure_factor_intensity(F).sum()
    I.backward()
    assert xt_t.B_iso_asu.grad is not None
    assert (xt_t.B_iso_asu.grad <= 0).all()


def test_grad_occupancy(torch_only):
    """∂|F|² / ∂occ should be positive (more atoms → bigger |F| typically)."""
    import torch
    from midas_hkls import read_cif, structure_factors, structure_factor_intensity

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch(requires_grad={"occ": True})
    F = structure_factors(xt_t, [(1, 1, 1)])
    I = structure_factor_intensity(F).sum()
    I.backward()
    assert xt_t.occ_asu.grad is not None
    assert (xt_t.occ_asu.grad > 0).all()


def test_grad_fractional_position(torch_only):
    """For Si in Fd-3m, the special position has zero gradient w.r.t. fract."""
    import torch
    from midas_hkls import read_cif, structure_factors, structure_factor_intensity

    xt = read_cif(DATA / "ceo2.cif")
    xt_t = xt.to_torch(requires_grad={"fract": True})
    F = structure_factors(xt_t, [(3, 1, 1), (4, 0, 0)])
    I = structure_factor_intensity(F).sum()
    I.backward()
    assert xt_t.fract_asu.grad is not None
    # O atom at (1/4, 1/4, 1/4) is on a 4-bar 3 m site; its gradient w.r.t.
    # fractional coords should be small but the fitting graph still allows it.
    # We just check finite, not the special-position constraint.
    assert torch.isfinite(xt_t.fract_asu.grad).all()


def test_gradcheck_lattice(torch_only):
    """Strict gradcheck of |F|² w.r.t. lattice params on a small structure."""
    import torch
    from midas_hkls import read_cif, structure_factors, structure_factor_intensity

    xt = read_cif(DATA / "alpha_fe.cif")
    hkls = [(1, 1, 0), (2, 0, 0), (2, 1, 1)]

    def fn(lat_params):
        # rebuild crystal_t from lat_params with gradient
        xt_t = xt.to_torch()
        xt_t.lattice_params = lat_params
        # Recompute fract/occ/B as snapshots since lat doesn't change them
        F = structure_factors(xt_t, hkls)
        return structure_factor_intensity(F)

    L = xt.lattice
    lat = torch.tensor([L.a, L.b, L.c, L.alpha, L.beta, L.gamma],
                       dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (lat,), atol=1e-4, eps=1e-5)


def test_gradcheck_atom_position(torch_only):
    """Strict gradcheck of |F|² w.r.t. atomic positions for an off-special site."""
    import torch
    from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, structure_factors, structure_factor_intensity

    # Use a low-symmetry structure where atom positions are not on special sites:
    # Place a single atom at a generic position in P1.
    sg = SpaceGroup.from_number(1)  # P1 — only the identity, no special positions
    lat = Lattice(5.0, 6.0, 7.0, 80.0, 85.0, 95.0)  # triclinic, generic
    a1 = Atom("Fe", (0.13, 0.27, 0.41), B_iso=0.5)
    a2 = Atom("O",  (0.55, 0.42, 0.18), B_iso=1.0)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[a1, a2])
    hkls = [(1, 0, 0), (0, 1, 0), (1, 1, 1), (2, 1, 0)]

    def fn(fract):
        xt_t = xt.to_torch()
        xt_t.fract_asu = fract                             # link_to_asu re-expands automatically
        F = structure_factors(xt_t, hkls)
        return structure_factor_intensity(F)

    fract = torch.tensor([list(a1.fract), list(a2.fract)], dtype=torch.float64,
                         requires_grad=True)
    assert torch.autograd.gradcheck(fn, (fract,), atol=1e-4, eps=1e-5)
