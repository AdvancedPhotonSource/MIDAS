"""Tests for the midas-hkls -> forward-model adapter.

These tests skip cleanly when the optional ``midas-hkls`` package is not
installed.

Run with:
    cd packages/midas_diffract
    python -m pytest tests/test_hkls.py -v
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

midas_hkls = pytest.importorskip("midas_hkls")
from midas_hkls import Lattice, SpaceGroup

from midas_diffract import (
    HEDMForwardModel,
    HEDMGeometry,
    hkls_for_forward_model,
)
from midas_diffract.hkls import _cartesian_B_matrix

DEG2RAD = math.pi / 180.0
MIDAS_HOME = Path(os.environ.get("MIDAS_HOME", "/Users/hsharma/opt/MIDAS"))
GETHKLLIST = MIDAS_HOME / "build" / "bin" / "GetHKLList"


@pytest.fixture
def fcc_au():
    return SpaceGroup.from_number(225), Lattice.for_system("cubic", a=4.08)


def test_returns_three_torch_tensors(fcc_au):
    sg, lat = fcc_au
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=8.0,
    )
    assert isinstance(hkls_cart, torch.Tensor)
    assert isinstance(thetas, torch.Tensor)
    assert isinstance(hkls_int, torch.Tensor)
    M = hkls_cart.shape[0]
    assert M > 0
    assert hkls_cart.shape == (M, 3)
    assert thetas.shape == (M,)
    assert hkls_int.shape == (M, 3)


def test_default_dtype_is_float64(fcc_au):
    sg, lat = fcc_au
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=8.0,
    )
    assert hkls_cart.dtype == torch.float64
    assert thetas.dtype == torch.float64
    assert hkls_int.dtype == torch.float64


def test_hkls_int_are_integer_valued(fcc_au):
    sg, lat = fcc_au
    _, _, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
    )
    assert torch.allclose(hkls_int, hkls_int.round(), atol=0.0)


def test_expansion_count_matches_total_multiplicity(fcc_au):
    sg, lat = fcc_au
    refs = midas_hkls.generate_hkls(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
    )
    expected = sum(len(sg.equivalent_hkls(r.h, r.k, r.l)) for r in refs)
    _, _, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
        expand_equivalents=True,
    )
    assert hkls_int.shape[0] == expected


def test_expand_equivalents_false_returns_asu_only(fcc_au):
    sg, lat = fcc_au
    refs = midas_hkls.generate_hkls(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
    )
    _, _, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
        expand_equivalents=False,
    )
    assert hkls_int.shape[0] == len(refs)


def test_thetas_consistent_with_bragg_law(fcc_au):
    """|G| * lambda / 2 = sin(theta) by Bragg's law."""
    sg, lat = fcc_au
    wl = 0.172979
    hkls_cart, thetas, _ = hkls_for_forward_model(
        sg, lat, wavelength_A=wl, two_theta_max_deg=15.0,
    )
    g_mag = torch.linalg.norm(hkls_cart, dim=-1)
    lhs = g_mag * wl / 2.0
    rhs = torch.sin(thetas)
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_b_matrix_matches_model_cubic(fcc_au):
    """For a cubic lattice, B = (1/a) * I."""
    _, lat = fcc_au
    B = _cartesian_B_matrix((lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma))
    expected = (1.0 / lat.a) * np.eye(3)
    np.testing.assert_allclose(B, expected, atol=1e-10)


def test_b_matrix_consistent_with_correct_hkls_latc(fcc_au):
    """Helper-built G-vectors must match the model's strain-recompute path.

    The forward model rebuilds B from lattice parameters in
    ``correct_hkls_latc``. If we pass the helper output to the constructor
    and then call ``correct_hkls_latc(reference_lattice)``, both paths must
    agree -- otherwise the strain code would silently disagree with the
    reference state.
    """
    sg, lat = fcc_au
    wl = 0.172979
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=wl, two_theta_max_deg=10.0,
    )
    geom = HEDMGeometry(
        Lsd=1e6, y_BC=1024, z_BC=1024, px=200,
        omega_start=0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=wl,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )
    latc = torch.tensor([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma],
                         dtype=torch.float64)
    G_model, theta_model = model.correct_hkls_latc(latc)
    # correct_hkls_latc carries the input dtype (float64) through; helper
    # also returns float64 by default. Tolerance ~ 1e-9 (mostly geometry
    # epsilon clamps in the B-matrix builder, fwd vs analytic form here).
    assert torch.allclose(hkls_cart, G_model.double(), atol=1e-8)
    assert torch.allclose(thetas, theta_model.double(), atol=1e-8)


def test_forward_pass_runs_with_helper_output(fcc_au):
    sg, lat = fcc_au
    wl = 0.172979
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=wl, two_theta_max_deg=8.0,
    )
    geom = HEDMGeometry(
        Lsd=1e6, y_BC=1024, z_BC=1024, px=200,
        omega_start=0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0, wavelength=wl,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )
    euler = torch.tensor([[0.5, 0.7, 0.3]], dtype=torch.float64)
    pos = torch.zeros(1, 3, dtype=torch.float64)
    spots = model(euler, pos)
    n_valid = int((spots.valid > 0.5).sum().item())
    assert n_valid > 0
    assert n_valid <= hkls_cart.shape[0] * 2  # at most 2 omega solutions per hkl


def test_no_reflections_raises(fcc_au):
    """Asking for a cutoff below the lowest reflection raises informatively."""
    sg, lat = fcc_au
    with pytest.raises(ValueError, match="no reflections"):
        # 2theta_max=0.1 deg is below the (111) reflection
        hkls_for_forward_model(
            sg, lat, wavelength_A=0.172979, two_theta_max_deg=0.1,
        )


def test_missing_cutoff_raises(fcc_au):
    sg, lat = fcc_au
    with pytest.raises(ValueError, match="d_min OR"):
        hkls_for_forward_model(sg, lat, wavelength_A=0.172979)


@pytest.mark.skipif(
    not GETHKLLIST.exists(),
    reason="MIDAS GetHKLList not built; cross-check skipped",
)
def test_matches_gethkllist_cubic_fcc(tmp_path, fcc_au):
    """Cross-validate row count, 2theta values, and (h,k,l) set against the C tool.

    GetHKLList enumerates all symmetry-equivalent reflections within a
    radial cutoff (``MaxRingRad`` microns at ``Lsd`` microns). We pick a
    cutoff that yields a small but non-trivial set and compare line-for-line.
    """
    import subprocess

    sg, lat = fcc_au
    wl = 0.172979
    Lsd = 1_000_000.0
    max_ring_rad = 140_000.0
    two_theta_max_deg = math.degrees(math.atan(max_ring_rad / Lsd)) + 0.5

    param_file = tmp_path / "params.txt"
    param_file.write_text(
        "LatticeParameter 4.08 4.08 4.08 90.0 90.0 90.0\n"
        f"Wavelength {wl}\n"
        "SpaceGroup 225\n"
        f"Lsd {Lsd}\n"
        f"MaxRingRad {max_ring_rad}\n"
    )
    result = subprocess.run(
        [str(GETHKLLIST), str(param_file)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"GetHKLList failed: {result.stderr}"

    c_data = np.loadtxt(tmp_path / "hkls.csv", skiprows=1)
    c_hkls = set(map(tuple, c_data[:, 0:3].astype(int).tolist()))
    c_two_thetas = sorted(set(np.round(c_data[:, 9], 6).tolist()))

    _, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=wl, two_theta_max_deg=two_theta_max_deg,
    )
    py_hkls = set(map(tuple, hkls_int.numpy().astype(int).tolist()))
    py_two_thetas = sorted(set(
        np.round((2 * thetas.numpy() * 180.0 / math.pi), 6).tolist()
    ))

    # The Python helper may admit a sliver of extra reflections in the
    # 2theta_max + 0.5deg buffer. Trim to the C cutoff for fair comparison.
    c_max = c_data[:, 9].max()
    py_keep = (2 * thetas.numpy() * 180.0 / math.pi) <= c_max + 1e-6
    py_hkls_trimmed = set(
        map(tuple,
            hkls_int.numpy()[py_keep].astype(int).tolist())
    )

    assert py_hkls_trimmed == c_hkls, (
        f"hkls set mismatch: missing in Py {c_hkls - py_hkls_trimmed}, "
        f"extra in Py {py_hkls_trimmed - c_hkls}"
    )
    # Compare the unique 2theta rings (ignoring the buffer)
    py_2t_trimmed = sorted(set(
        np.round((2 * thetas.numpy()[py_keep] * 180.0 / math.pi), 6).tolist()
    ))
    np.testing.assert_allclose(py_2t_trimmed, c_two_thetas, atol=1e-5)
