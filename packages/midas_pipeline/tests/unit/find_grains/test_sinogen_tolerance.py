"""Tolerance-mode sinogen tests."""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    SinogenOutputs,
    SpotData,
    SpotList,
    apply_variant_torch,
    generate_sinograms_tolerance,
)


def make_spotlist_one_grain(*, ring=1, omega_vals=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)):
    spots = [
        SpotData(
            omega=om, eta=10.0 + idx * 0.05, ring_nr=ring,
            merged_id=idx + 1, scan_nr=0, grain_nr=0, spot_nr=idx,
        )
        for idx, om in enumerate(omega_vals)
    ]
    return SpotList(spots=spots, max_n_hkls=len(spots))


def synth_all_spots(spotlist, n_scans, intensity=100.0):
    """Make an ``all_spots`` (n_spots, 10) array. Each scan gets one row per
    spot, with same omega/eta/ring values, scanNr varying 0..n_scans-1."""
    rows = []
    for sc in range(n_scans):
        for sd in spotlist.spots:
            rows.append([
                0.0,                # y
                0.0,                # z
                sd.omega,           # omega
                intensity + sc,     # intensity per scan (varies for sinogen variance)
                sd.merged_id + sc * 100,  # spotID
                sd.ring_nr,         # ring
                sd.eta,             # eta
                5.0,                # theta
                1.0,                # dspacing
                sc,                 # scanNr
            ])
    return np.asarray(rows, dtype=np.float64)


def test_sinogen_tolerance_1grain_6hkl_5scan_fills_correctly(tmp_path):
    sl = make_spotlist_one_grain()
    n_scans = 5
    all_spots = synth_all_spots(sl, n_scans, intensity=100.0)
    out = generate_sinograms_tolerance(
        spot_list=sl,
        n_unique=1,
        all_spots=all_spots,
        n_scans=n_scans,
        tol_ome=0.5,
        tol_eta=0.1,
        output_dir=tmp_path,
    )
    assert out.n_grains == 1
    assert out.max_n_hkls == 6
    assert out.n_scans == n_scans

    # Read the raw sino back and check intensities match.
    raw = np.frombuffer(
        (tmp_path / out.sino_paths["raw"].split("/")[-1]).read_bytes(),
        dtype=np.float64,
    ).reshape(1, 6, n_scans)
    # Each cell should be intensity = 100 + scanNr.
    expected = np.broadcast_to(
        100.0 + np.arange(n_scans, dtype=np.float64)[None, None, :], (1, 6, n_scans),
    )
    np.testing.assert_allclose(raw, expected, atol=1e-12)


def test_sinogen_tolerance_bit_exact_float64(tmp_path):
    """Two runs of the same inputs produce byte-identical raw output."""
    sl = make_spotlist_one_grain()
    n_scans = 4
    all_spots = synth_all_spots(sl, n_scans, intensity=50.0)
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    out_a = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=a,
    )
    out_b = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=b,
    )
    assert (a / out_a.sino_paths["raw"].split("/")[-1]).read_bytes() == \
           (b / out_b.sino_paths["raw"].split("/")[-1]).read_bytes()


def test_apply_variant_torch_matches_numpy(tmp_path):
    """Torch normalize/abs variants match numpy element-wise to 1e-12."""
    torch = pytest.importorskip("torch")
    sl = make_spotlist_one_grain()
    n_scans = 4
    all_spots = synth_all_spots(sl, n_scans, intensity=10.0)
    out = generate_sinograms_tolerance(
        spot_list=sl, n_unique=1, all_spots=all_spots, n_scans=n_scans,
        tol_ome=0.5, tol_eta=0.1, output_dir=tmp_path,
    )
    raw = np.frombuffer(
        (tmp_path / out.sino_paths["raw"].split("/")[-1]).read_bytes(),
        dtype=np.float64,
    ).reshape(1, 6, n_scans)
    norm = np.frombuffer(
        (tmp_path / out.sino_paths["norm"].split("/")[-1]).read_bytes(),
        dtype=np.float64,
    ).reshape(1, 6, n_scans)
    # The norm variant divides each cell by per-spot max intensity.
    max_int = raw.max(axis=-1, keepdims=True)
    expected = np.where(raw > 0, raw / np.where(max_int > 0, max_int, 1.0), raw)
    np.testing.assert_allclose(norm, expected, atol=1e-12)

    # Torch path on the raw tensor.
    raw_t = torch.tensor(raw, dtype=torch.float64)
    mi_t = torch.tensor(max_int.squeeze(-1), dtype=torch.float64)
    norm_t = apply_variant_torch(raw_t, mi_t, normalize=True, abs_transform=False)
    np.testing.assert_allclose(norm_t.cpu().numpy(), expected, atol=1e-12)


def test_apply_variant_torch_is_differentiable():
    torch = pytest.importorskip("torch")
    raw_t = torch.tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
                         dtype=torch.float64, requires_grad=True)
    mi_t = torch.tensor(np.array([[3.0, 6.0]]), dtype=torch.float64)
    out = apply_variant_torch(raw_t, mi_t, normalize=True, abs_transform=False)
    loss = out.sum()
    loss.backward()
    assert raw_t.grad is not None
    # Gradient should be 1/max_int at each cell where sino>0.
    # (Differentiability through the torch.where is verified — non-zero grad.)
    assert (raw_t.grad.abs().sum() > 0)


def test_apply_variant_torch_runs_on_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    raw_t = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64, device="cuda")
    mi_t = torch.tensor([[2.0]], dtype=torch.float64, device="cuda")
    out = apply_variant_torch(raw_t, mi_t, normalize=True, abs_transform=False)
    assert out.device.type == "cuda"


def test_apply_variant_torch_runs_on_mps():
    torch = pytest.importorskip("torch")
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    # MPS only supports fp32.
    raw_t = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32, device="mps")
    mi_t = torch.tensor([[2.0]], dtype=torch.float32, device="mps")
    out = apply_variant_torch(raw_t, mi_t, normalize=True, abs_transform=False)
    assert out.device.type == "mps"
