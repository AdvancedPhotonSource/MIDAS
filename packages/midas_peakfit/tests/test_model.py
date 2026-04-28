"""Test the differentiable Pseudo-Voigt model."""
import numpy as np
import torch

from midas_peakfit.model import forward_pseudo_voigt, residuals, integrated_intensity


def test_forward_at_peak_center():
    """At the peak center: G=L=1, so model = bg + Imax."""
    # 1 region, 1 peak, 1 pixel at peak center
    x = torch.tensor([[5.0, 100.0, 50.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64)
    Rs = torch.tensor([[50.0]], dtype=torch.float64)  # at center
    Etas = torch.tensor([[0.0]], dtype=torch.float64)
    out = forward_pseudo_voigt(x, Rs, Etas, n_peaks=1)
    expected = 5.0 + 100.0  # bg + Imax × (μ × 1 + (1-μ) × 1)
    assert abs(float(out[0, 0]) - expected) < 1e-9


def test_residual_zero_at_perfect_fit():
    """Residual on a noiseless target evaluated at truth = 0."""
    x = torch.tensor([[5.0, 100.0, 50.0, 0.0, 0.5, 1.0, 0.8, 1.0, 0.8]], dtype=torch.float64)
    Rs = torch.linspace(48, 52, 50).unsqueeze(0).double()
    Etas = torch.linspace(-2, 2, 50).unsqueeze(0).double()
    pmask = torch.ones_like(Rs)
    z = forward_pseudo_voigt(x, Rs, Etas, n_peaks=1)
    r = residuals(x, z, Rs, Etas, pmask, n_peaks=1)
    np.testing.assert_allclose(r.numpy(), 0.0, atol=1e-10)


def test_pixel_mask_zeroes_padding():
    """Padded pixels (mask=0) contribute zero to residual."""
    x = torch.tensor([[5.0, 100.0, 50.0, 0.0, 0.5, 1.0, 0.8, 1.0, 0.8]], dtype=torch.float64)
    Rs = torch.tensor([[50.0, 99.0, 99.0]], dtype=torch.float64)
    Etas = torch.zeros((1, 3), dtype=torch.float64)
    z = torch.tensor([[100.0, 99.0, 99.0]], dtype=torch.float64)
    pmask = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    r = residuals(x, z, Rs, Etas, pmask, n_peaks=1)
    # First pixel: model = 105, target = 100, residual = 5 (×1 mask)
    # Other two: padded → 0
    assert abs(float(r[0, 0]) - 5.0) < 1e-9
    assert float(r[0, 1]) == 0.0
    assert float(r[0, 2]) == 0.0


def test_integrated_intensity_smoke():
    """Integrated intensity returns positive values at peak center."""
    x = torch.tensor([[5.0, 100.0, 50.0, 0.0, 0.5, 1.0, 0.8, 1.0, 0.8]], dtype=torch.float64)
    Rs = torch.linspace(48, 52, 50).unsqueeze(0).double()
    Etas = torch.linspace(-2, 2, 50).unsqueeze(0).double()
    pmask = torch.ones_like(Rs)
    ii, np_ = integrated_intensity(x, Rs, Etas, pmask, n_peaks=1)
    assert ii.shape == (1, 1)
    assert float(ii[0, 0]) > 0.0
    assert int(np_[0, 0]) > 0
