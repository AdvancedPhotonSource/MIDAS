"""Cromer-Mann form-factor numerics — checks against IT92 reference values."""
from __future__ import annotations

import numpy as np
import pytest

from midas_hkls import form_factor, form_factor_batch, available_elements


def test_table_size():
    elements = available_elements()
    assert len(elements) == 98  # Z = 1..98 from gemmi IT92 export
    for sym in ("H", "C", "N", "O", "Si", "Fe", "Cu", "Ce"):
        assert sym in elements


def test_f_at_zero_equals_Z():
    """f(s=0) = Σ aᵢ + c, which IT92 fits to within ~1% of the atomic number."""
    cases = [("H", 1), ("C", 6), ("O", 8), ("Si", 14), ("Fe", 26), ("Cu", 29), ("Ce", 58)]
    for sym, Z in cases:
        f0 = form_factor(0.0, sym)
        assert abs(f0 - Z) < 0.05 * Z, f"{sym}: f(0)={f0} vs Z={Z}"


def test_monotonic_decay():
    s2 = np.linspace(0.0, 4.0, 50)
    for sym in ("Fe", "O", "Si"):
        f = form_factor(s2, sym)
        diffs = np.diff(f)
        # IT92 fits are monotonically decreasing for s² in [0, large].
        assert (diffs <= 1e-10).all(), f"{sym}: non-monotonic decay"


def test_known_values_iron():
    """Spot values for Fe from gemmi IT92 evaluator (used as ground truth)."""
    fe_at = lambda s2: float(form_factor(s2, "Fe"))
    assert fe_at(0.0) == pytest.approx(25.9904, abs=1e-3)
    assert fe_at(0.5) == pytest.approx(8.43694, abs=1e-3)


def test_charge_normalization():
    """Fe2+ falls back to neutral Fe (gemmi IT92 only ships neutral atoms)."""
    f_neutral = form_factor(0.5, "Fe")
    f_ion = form_factor(0.5, "Fe2+")
    assert f_ion == pytest.approx(f_neutral)


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        form_factor(0.0, "Xx")


def test_batch_shape():
    s2 = np.array([0.0, 0.1, 0.5, 1.0])
    elements = ["Ce", "O", "O"]
    out = form_factor_batch(s2, elements)
    assert out.shape == (4, 3)
    # f(0) Ce ~ 58, O ~ 8
    assert out[0, 0] == pytest.approx(58, abs=1.0)
    assert out[0, 1] == pytest.approx(8, abs=0.5)
    assert out[0, 2] == pytest.approx(8, abs=0.5)


def test_torch_gradients():
    torch = pytest.importorskip("torch")
    s2 = torch.tensor([0.0, 0.1, 0.5], dtype=torch.float64, requires_grad=True)
    f = form_factor(s2, "Fe")
    f.sum().backward()
    assert s2.grad is not None
    # df/ds² = -Σ aᵢbᵢ exp(-bᵢ s²); should be negative (form factor decays in s²)
    assert (s2.grad <= 0).all()


def test_torch_batch_shape():
    torch = pytest.importorskip("torch")
    s2 = torch.tensor([[0.0, 0.5], [0.1, 0.2]], dtype=torch.float64)
    out = form_factor_batch(s2, ["Fe", "O", "Si"])
    assert tuple(out.shape) == (2, 2, 3)
