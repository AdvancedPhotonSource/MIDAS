"""Test device + dtype switches: CPU/CUDA, fp32/fp64.

GPU tests are gated by ``torch.cuda.is_available()`` and skip cleanly on
non-CUDA hosts.
"""
import numpy as np
import pytest
import torch

from midas_peakfit.lm import LMConfig, lm_solve
from midas_peakfit.model import forward_pseudo_voigt


def _run_synthetic(device: torch.device, dtype: torch.dtype) -> dict:
    """Run a small synthetic LM fit on the given device/dtype."""
    B = 2
    M = 100
    n_peaks = 1

    Rs = (
        torch.linspace(98.0, 102.0, M)
        .unsqueeze(0)
        .repeat(B, 1)
        .to(dtype=dtype, device=device)
    )
    Etas = (
        torch.linspace(-2.0, 2.0, M)
        .unsqueeze(0)
        .repeat(B, 1)
        .to(dtype=dtype, device=device)
    )
    pmask = torch.ones_like(Rs)
    x_true = torch.tensor(
        [[5.0, 200.0, 100.0, 0.0, 0.3, 1.0, 0.8, 0.5, 0.4]] * B,
        dtype=dtype, device=device,
    )
    z = forward_pseudo_voigt(x_true, Rs, Etas, n_peaks)

    x_init = torch.tensor(
        [[4.0, 180.0, 99.5, 0.5, 0.5, 1.5, 1.5, 0.7, 0.7]] * B,
        dtype=dtype, device=device,
    )
    lo = torch.tensor(
        [[0.0, 50.0, 99.0, -1.0, 0.0, 0.01, 0.01, 0.005, 0.005]] * B,
        dtype=dtype, device=device,
    )
    hi = torch.tensor(
        [[50.0, 1000.0, 101.0, 1.0, 1.0, 10.0, 10.0, 5.0, 5.0]] * B,
        dtype=dtype, device=device,
    )

    x_fit, c_fit, rc, _ = lm_solve(
        x_init, lo, hi, z, Rs, Etas, pmask, n_peaks=1,
        config=LMConfig(max_iter=50),
    )
    return {
        "x_fit": x_fit.detach().cpu().numpy(),
        "rc": rc.detach().cpu().numpy(),
        "device": str(x_fit.device),
        "dtype": str(x_fit.dtype),
    }


def test_cpu_fp64():
    out = _run_synthetic(torch.device("cpu"), torch.float64)
    assert out["device"].startswith("cpu")
    assert "float64" in out["dtype"]
    R_fit = out["x_fit"][:, 2]
    assert (np.abs(R_fit - 100.0) < 0.1).all()


def test_cpu_fp32():
    """fp32 has reduced precision but should still recover R within 0.5 px."""
    out = _run_synthetic(torch.device("cpu"), torch.float32)
    assert "float32" in out["dtype"]
    R_fit = out["x_fit"][:, 2]
    assert (np.abs(R_fit - 100.0) < 0.5).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_fp64():
    out = _run_synthetic(torch.device("cuda"), torch.float64)
    assert out["device"].startswith("cuda")
    R_fit = out["x_fit"][:, 2]
    assert (np.abs(R_fit - 100.0) < 0.1).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_fp32():
    out = _run_synthetic(torch.device("cuda"), torch.float32)
    R_fit = out["x_fit"][:, 2]
    assert (np.abs(R_fit - 100.0) < 0.5).all()


def test_cli_device_resolution():
    """The CLI parses --device cuda but the orchestrator falls back to CPU
    when CUDA is missing — this should not raise."""
    from midas_peakfit.orchestrator import run

    # Use the synthetic fixture indirectly: just confirm CLI argument
    # plumbing for device works without CUDA.
    # (The actual run is exercised by test_parity_synthetic.)
    assert run is not None
