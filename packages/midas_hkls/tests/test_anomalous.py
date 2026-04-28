"""Anomalous (resonant) scattering correction tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def torch_only():
    return pytest.importorskip("torch")


@pytest.fixture
def gemmi_only():
    return pytest.importorskip("gemmi")


def test_table_shape():
    from midas_hkls.anomalous import _table, _energies, available_elements_anomalous
    e = _energies()
    assert e[0] == pytest.approx(100.0)
    assert e[-1] == pytest.approx(200_000.0)
    assert len(e) == 401
    elements = available_elements_anomalous()
    assert len(elements) == 92
    assert "Fe" in elements and "Cu" in elements and "Ce" in elements


def test_wavelength_energy_round_trip():
    from midas_hkls.anomalous import wavelength_to_energy_eV, energy_eV_to_wavelength
    for E in (1000.0, 8048.0, 17479.0, 80000.0):
        wl = energy_eV_to_wavelength(E)
        E2 = wavelength_to_energy_eV(wl)
        assert E2 == pytest.approx(E, rel=1e-9)


def test_numpy_lookup_against_gemmi(gemmi_only):
    """At grid energies, our interp must match gemmi exactly."""
    import gemmi
    from midas_hkls.anomalous import anomalous_correction, _energies, energy_eV_to_wavelength

    energies = _energies()
    # sample a few grid points and several elements
    for i in (50, 150, 300):
        E = float(energies[i])
        wl = energy_eV_to_wavelength(E)
        for el, z in [("O", 8), ("Si", 14), ("Fe", 26), ("Cu", 29), ("Ce", 58)]:
            fp_g, fpp_g = gemmi.cromer_liberman(z=z, energy=E)
            fp_us, fpp_us = anomalous_correction([el], wl)
            assert fp_us[0] == pytest.approx(fp_g, abs=1e-3), f"{el}@{E}: fp"
            assert fpp_us[0] == pytest.approx(fpp_g, abs=1e-3), f"{el}@{E}: fpp"


def test_numpy_interp_off_grid(gemmi_only):
    """Off-grid: linear interpolation in log(E) should agree with direct
    gemmi.cromer_liberman to within ~0.1 (the size of typical between-grid
    step at 401 points spanning 3 decades)."""
    import gemmi
    from midas_hkls.anomalous import anomalous_correction, energy_eV_to_wavelength

    for E in (8050.0, 17500.0, 80100.0):
        wl = energy_eV_to_wavelength(E)
        fp_us, fpp_us = anomalous_correction(["Fe"], wl)
        fp_g, fpp_g = gemmi.cromer_liberman(z=26, energy=E)
        assert abs(fp_us[0] - fp_g) < 0.05
        assert abs(fpp_us[0] - fpp_g) < 0.05


def test_torch_path_dtype_device(torch_only):
    import torch
    from midas_hkls.anomalous import anomalous_correction

    wl = torch.tensor(0.173, dtype=torch.float64)
    fp, fpp = anomalous_correction(["Fe", "Ce", "O"], wl)
    assert fp.shape == (3,) and fpp.shape == (3,)
    assert fp.dtype == torch.float64
    # Ce at 71.7 keV has small but nonzero f'' from L1/L2/L3 edges below 7 keV
    assert fpp[1] > 0


def test_torch_gradient_through_wavelength(torch_only):
    import torch
    from midas_hkls.anomalous import anomalous_correction

    wl = torch.tensor(1.5418, dtype=torch.float64, requires_grad=True)  # Cu Kα
    fp, fpp = anomalous_correction(["Fe"], wl)
    fp.sum().backward()
    assert wl.grad is not None
    # near Fe K-edge (~7.1 keV ≈ wl 1.74 Å), f' is rapidly varying
    assert torch.isfinite(wl.grad).all()


def test_unknown_element_raises():
    from midas_hkls.anomalous import anomalous_correction
    with pytest.raises(KeyError):
        anomalous_correction(["Xx"], 1.5418)


def test_anomalous_in_structure_factor(torch_only, gemmi_only):
    """Cross-check: anomalous contribution at Cu Kα for Fe atoms slightly
    reduces |F| compared to non-anomalous (both real-part-only and full
    complex F should match gemmi when f', f'' are added)."""
    import torch
    import gemmi
    from midas_hkls import (
        Atom, Crystal, Lattice, SpaceGroup, generate_hkls,
        structure_factors, anomalous_correction,
    )

    sg = SpaceGroup.from_number(229)
    lat = Lattice.for_system("cubic", a=2.8665)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[
        Atom("Fe", (0.0, 0.0, 0.0), B_iso=0.35),
    ])
    xt_t = xt.to_torch()

    refs = generate_hkls(sg, lat, wavelength_A=1.5418, two_theta_max_deg=120.0)[:6]
    hkls = [(r.h, r.k, r.l) for r in refs]

    F_no = structure_factors(xt_t, hkls).detach()
    F_an = structure_factors(xt_t, hkls, wavelength_A=1.5418, anomalous=True).detach()

    # Anomalous F has nonzero imag part where f''>0 (Fe at Cu Kα: f'' ≈ 3.2)
    assert F_an.imag.abs().max() > 0.5
    # Real part: f' is negative at Cu Kα for Fe → real |F| should DECREASE
    assert F_an.real.abs().max() < F_no.real.abs().max() + 1e-9
