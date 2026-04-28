"""Test the batched Levenberg-Marquardt solver."""
import numpy as np
import torch

from midas_peakfit.lm import LMConfig, lm_solve
from midas_peakfit.model import forward_pseudo_voigt


def _build_synthetic_batch(B=4, M=200, n_peaks=1, seed=0):
    torch.manual_seed(seed)
    Rs = torch.linspace(98.0, 102.0, M).unsqueeze(0).repeat(B, 1).double()
    Etas = torch.linspace(-2.0, 2.0, M).unsqueeze(0).repeat(B, 1).double()
    pmask = torch.ones_like(Rs)
    x_true = torch.tensor(
        [[5.0, 200.0, 100.0, 0.0, 0.3, 1.0, 0.8, 0.5, 0.4]] * B, dtype=torch.float64
    )
    z = forward_pseudo_voigt(x_true, Rs, Etas, n_peaks)
    return x_true, z, Rs, Etas, pmask


def test_lm_recovers_position():
    """LM should recover R, Eta nearly exactly on noiseless data."""
    x_true, z, Rs, Etas, pmask = _build_synthetic_batch()
    B, _ = x_true.shape

    x_init = torch.tensor(
        [[4.0, 180.0, 99.5, 0.5, 0.5, 1.5, 1.5, 0.7, 0.7]] * B, dtype=torch.float64
    )
    lo = torch.tensor(
        [[0.0, 50.0, 99.0, -1.0, 0.0, 0.01, 0.01, 0.005, 0.005]] * B,
        dtype=torch.float64,
    )
    hi = torch.tensor(
        [[50.0, 1000.0, 101.0, 1.0, 1.0, 10.0, 10.0, 5.0, 5.0]] * B,
        dtype=torch.float64,
    )

    x_fit, c_fit, rc, _ = lm_solve(
        x_init, lo, hi, z, Rs, Etas, pmask, n_peaks=1,
        config=LMConfig(max_iter=80),
    )
    # All regions converged
    assert (rc == 0).all().item()
    # Position parameters match within sub-pixel
    R_fit = x_fit[:, 2]
    Eta_fit = x_fit[:, 3]
    assert ((R_fit - 100.0).abs() < 0.05).all().item()
    assert ((Eta_fit - 0.0).abs() < 0.05).all().item()


def test_lm_returns_correct_shapes():
    x_true, z, Rs, Etas, pmask = _build_synthetic_batch(B=3)
    B = x_true.shape[0]
    x_init = x_true.clone()
    lo = x_true - 1.0
    hi = x_true + 1.0
    x_fit, c_fit, rc, _ = lm_solve(
        x_init, lo, hi, z, Rs, Etas, pmask, n_peaks=1,
        config=LMConfig(max_iter=10),
    )
    assert x_fit.shape == x_true.shape
    assert c_fit.shape == (B,)
    assert rc.shape == (B,)
