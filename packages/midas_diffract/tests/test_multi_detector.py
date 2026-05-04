"""Tests for multi-detector ("panel" mode) FF-HEDM forward simulation.

The single-detector FF case has been validated pixel-exact in
``tests/test_c_comparison.py``; these tests cover the new
``multi_mode="panel"`` and per-detector tilts/Lsd/BC handling that landed
to support 4-panel FF-HEDM forward simulation and joint refinement.

Run with:
    cd packages/midas_diffract
    python -m pytest tests/test_multi_detector.py -v
"""
import math

import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry


DEG2RAD = math.pi / 180.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _au_hkls():
    """Cubic Au HKLs through ring 10, in Cartesian B*hkl form (1/A)."""
    a = 4.08
    wl = 0.22291
    int_hkls = []
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-4, 5):
                if h == 0 and k == 0 and l == 0:
                    continue
                # FCC selection rule: hkl all even or all odd
                parities = (h % 2, k % 2, l % 2)
                if not (parities == (0, 0, 0) or parities == (1, 1, 1)):
                    continue
                int_hkls.append((h, k, l))
    int_hkls = torch.tensor(int_hkls, dtype=torch.float64)
    cart = int_hkls / a  # B = (1/a) * I for cubic
    g_norm = torch.linalg.norm(cart, dim=1)
    sin_th = wl * g_norm / 2.0
    keep = sin_th < 0.95
    cart = cart[keep]
    sin_th = sin_th[keep]
    thetas = torch.asin(sin_th)
    return cart.float(), thetas.float(), int_hkls[keep].float()


def _ring_geometry(n_det: int, *, tx_list=None, apply_tilts=False, mode="panel"):
    """Build a 4-panel-style ring geometry (or N-panel) with BC at the
    inner-corner of each panel (off-detector at ~y=z=2100 px).
    """
    if tx_list is None:
        # 90° spaced + 15° rotation about beam axis
        tx_list = [15.0 + 90.0 * i for i in range(n_det)]
        tx_list = [t if t <= 180 else t - 360 for t in tx_list]
    return HEDMGeometry(
        Lsd=[1.0e6] * n_det,
        y_BC=[2100.0] * n_det,
        z_BC=[2100.0] * n_det,
        tx=tx_list,
        ty=[0.0] * n_det,
        tz=[0.0] * n_det,
        px=200.0,
        omega_start=180.0,
        omega_step=-0.25,
        n_frames=1440,
        n_pixels_y=2048,
        n_pixels_z=2048,
        min_eta=6.0,
        wavelength=0.22291,
        flip_y=True,            # FF mode
        apply_tilts=apply_tilts,
        multi_mode=mode,
    )


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_n1_parity_scalar_vs_list():
    """Geometry with scalar Lsd/BC/tilt vs list-of-1 should give identical
    spot predictions."""
    hkls, thetas, _ = _au_hkls()

    g_scalar = HEDMGeometry(
        Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=180.0, omega_step=-0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291,
    )
    g_list = HEDMGeometry(
        Lsd=[1.0e6], y_BC=[1024.0], z_BC=[1024.0], px=200.0,
        omega_start=180.0, omega_step=-0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291,
    )

    m_scalar = HEDMForwardModel(hkls, thetas, g_scalar)
    m_list = HEDMForwardModel(hkls, thetas, g_list)

    euler = torch.tensor([[0.3, 0.5, 0.7]])
    pos = torch.zeros(1, 3)

    s_scalar = m_scalar(euler, pos)
    s_list = m_list(euler, pos)

    torch.testing.assert_close(s_scalar.y_pixel, s_list.y_pixel)
    torch.testing.assert_close(s_scalar.z_pixel, s_list.z_pixel)
    torch.testing.assert_close(s_scalar.valid, s_list.valid)
    assert s_scalar.det_id is None and s_list.det_id is None


def test_panel_mode_no_tilt_n1_matches_layered():
    """panel mode with N=1 panel and no tilts == layered mode result."""
    hkls, thetas, _ = _au_hkls()

    common = dict(
        Lsd=[1.0e6], y_BC=[1024.0], z_BC=[1024.0], px=200.0,
        omega_start=180.0, omega_step=-0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291, flip_y=True,
    )
    g_layered = HEDMGeometry(**common, multi_mode="layered")
    g_panel = HEDMGeometry(**common, multi_mode="panel")

    m_lay = HEDMForwardModel(hkls, thetas, g_layered)
    m_pan = HEDMForwardModel(hkls, thetas, g_panel)

    euler = torch.tensor([[0.3, 0.5, 0.7]])
    pos = torch.zeros(1, 3)

    s_lay = m_lay(euler, pos)
    s_pan = m_pan(euler, pos)

    torch.testing.assert_close(s_lay.y_pixel, s_pan.y_pixel)
    torch.testing.assert_close(s_lay.z_pixel, s_pan.z_pixel)
    torch.testing.assert_close(s_lay.valid, s_pan.valid)
    # In panel mode with N=1, det_id is all 0 where valid, undefined elsewhere
    assert s_pan.det_id is not None
    assert torch.all(s_pan.det_id == 0)


def test_panel_mode_4det_each_spot_lands_on_at_most_one_panel():
    """With BC at each panel's inner corner and the 90+15° pinwheel layout,
    no two panels overlap, so every valid spot should hit exactly one panel.
    """
    hkls, thetas, _ = _au_hkls()
    g = _ring_geometry(4, apply_tilts=True, mode="panel")
    model = HEDMForwardModel(hkls, thetas, g)

    # 5-grain population, broadcast to (G, 1, 1, 1)
    torch.manual_seed(0)
    euler = torch.rand(5, 3) * (2 * math.pi)
    pos = (torch.rand(5, 3) - 0.5) * 100.0  # within +-50 um cube

    spots = model(euler, pos)

    # spots.layer_valid is (D, G, K, M) -- count panels per spot
    panel_count = (spots.layer_valid > 0.5).sum(dim=0)  # (G, K, M)
    # Where spot is valid, exactly one panel hit; where invalid, zero
    valid_mask = spots.valid > 0.5
    assert torch.all(panel_count[valid_mask] == 1), (
        f"some valid spots hit multiple panels: max count "
        f"{panel_count[valid_mask].max().item()}"
    )
    assert torch.all(panel_count[~valid_mask] == 0)
    # det_id matches the panel where layer_valid is True for valid spots
    for d in range(4):
        on_d = (spots.layer_valid[d] > 0.5)
        assert torch.all(spots.det_id[on_d] == d)


def test_panel_mode_4det_coverage():
    """All four panels see a meaningful share of the valid spots."""
    hkls, thetas, _ = _au_hkls()
    g = _ring_geometry(4, apply_tilts=True, mode="panel")
    model = HEDMForwardModel(hkls, thetas, g)

    torch.manual_seed(1)
    euler = torch.rand(20, 3) * (2 * math.pi)
    pos = torch.zeros(20, 3)

    spots = model(euler, pos)
    valid_mask = spots.valid > 0.5
    det_ids_seen = spots.det_id[valid_mask]
    counts = torch.bincount(det_ids_seen, minlength=4)
    total = counts.sum().item()
    assert total > 0, "no valid spots produced"
    # No panel should be starved of spots given the symmetric layout
    fractions = counts.float() / total
    assert torch.all(fractions > 0.10), (
        f"panel coverage uneven: fractions {fractions.tolist()}"
    )


def test_per_detector_tilt_actually_applied():
    """Different per-detector tilts must produce different y/z pixels for
    spots that land on different panels.
    """
    hkls, thetas, _ = _au_hkls()
    # Two panels with very different txs
    g_a = _ring_geometry(2, tx_list=[0.0, 0.0], apply_tilts=True, mode="panel")
    g_b = _ring_geometry(2, tx_list=[0.0, 90.0], apply_tilts=True, mode="panel")
    m_a = HEDMForwardModel(hkls, thetas, g_a)
    m_b = HEDMForwardModel(hkls, thetas, g_b)

    torch.manual_seed(7)
    euler = torch.rand(3, 3) * (2 * math.pi)
    pos = torch.zeros(3, 3)

    s_a = m_a(euler, pos)
    s_b = m_b(euler, pos)

    # Detector 0 (tx=0 for both): same predictions
    on_d0_a = (s_a.det_id == 0) & (s_a.valid > 0.5)
    on_d0_b = (s_b.det_id == 0) & (s_b.valid > 0.5)
    assert torch.equal(on_d0_a, on_d0_b)
    torch.testing.assert_close(s_a.y_pixel[on_d0_a], s_b.y_pixel[on_d0_b])
    torch.testing.assert_close(s_a.z_pixel[on_d0_a], s_b.z_pixel[on_d0_b])

    # Detector 1 (tx differs): predictions on det 1 must differ where both
    # configurations placed a spot.
    on_d1_a = (s_a.det_id == 1) & (s_a.valid > 0.5)
    on_d1_b = (s_b.det_id == 1) & (s_b.valid > 0.5)
    common = on_d1_a & on_d1_b
    if common.any():
        diff = (s_a.y_pixel - s_b.y_pixel).abs() + (s_a.z_pixel - s_b.z_pixel).abs()
        assert diff[common].max().item() > 1e-3


def test_apply_tilts_flag_false_skips_in_ff_mode():
    """When apply_tilts=False (the default), FF mode must NOT apply tilts
    even when self.tilts has nonzero rows."""
    hkls, thetas, _ = _au_hkls()
    # Same tilts on every panel; should be a pure shift in lab frame, but
    # apply_tilts=False keeps it as-if tilts are zero.
    g_off = _ring_geometry(2, tx_list=[7.5, 7.5], apply_tilts=False, mode="panel")
    g_zero = _ring_geometry(2, tx_list=[0.0, 0.0], apply_tilts=False, mode="panel")

    m_off = HEDMForwardModel(hkls, thetas, g_off)
    m_zero = HEDMForwardModel(hkls, thetas, g_zero)

    torch.manual_seed(2)
    euler = torch.rand(4, 3) * (2 * math.pi)
    pos = torch.zeros(4, 3)
    s_off = m_off(euler, pos)
    s_zero = m_zero(euler, pos)
    torch.testing.assert_close(s_off.y_pixel, s_zero.y_pixel)
    torch.testing.assert_close(s_off.z_pixel, s_zero.z_pixel)
    torch.testing.assert_close(s_off.valid, s_zero.valid)


def test_per_detector_tilt_gradient_finite_difference():
    """Autograd gradient w.r.t. one panel's tx must match a central
    finite-difference estimate.

    The loss must not gate by ``valid``, because the validity mask is
    non-smooth at panel boundaries -- a small FD step can flip a spot in
    or out of bounds, producing a finite-jump term that autograd does
    not see. We instead build a smooth loss over only the spots that are
    valid on detector 1 and stay valid across the FD step.
    """
    hkls, thetas, _ = _au_hkls()
    g = _ring_geometry(4, apply_tilts=True, mode="panel")
    model = HEDMForwardModel(hkls, thetas, g)
    model.tilts.requires_grad_(True)

    torch.manual_seed(3)
    euler = torch.rand(3, 3) * (2 * math.pi)
    pos = torch.zeros(3, 3)

    # Pin the spot subset (mask) at the unperturbed configuration so the FD
    # step doesn't change which spots contribute to the loss.
    with torch.no_grad():
        s0 = model(euler, pos)
        on_d1_mask = ((s0.det_id == 1) & (s0.valid > 0.5))
    assert on_d1_mask.any(), "no spots on detector 1 at this seed"

    def loss_fn():
        s = model(euler, pos)
        # Sum of squared pixel coords for the FIXED set of spots.
        y = s.y_pixel[on_d1_mask]
        z = s.z_pixel[on_d1_mask]
        return (y.pow(2).sum() + z.pow(2).sum())

    loss = loss_fn()
    grad_auto = torch.autograd.grad(loss, model.tilts)[0].detach().clone()

    # Central FD on tilts[1, 0] (panel 1 tx)
    eps = 1e-3
    with torch.no_grad():
        model.tilts[1, 0] += eps
        l_plus = loss_fn().item()
        model.tilts[1, 0] -= 2 * eps
        l_minus = loss_fn().item()
        model.tilts[1, 0] += eps
    grad_fd = (l_plus - l_minus) / (2 * eps)

    assert torch.isfinite(grad_auto).all()
    assert math.isfinite(grad_fd)
    rel = abs(grad_auto[1, 0].item() - grad_fd) / (abs(grad_fd) + 1e-6)
    assert rel < 5e-3, (
        f"autograd vs FD mismatch: auto={grad_auto[1,0].item():.6e}, "
        f"fd={grad_fd:.6e}, rel={rel:.3e}"
    )

    # Other panels' tx should have ~zero gradient (loss only depends on det 1)
    for d in (0, 2, 3):
        assert abs(grad_auto[d, 0].item()) < 1e-3 * abs(grad_auto[1, 0].item()) + 1e-2


def test_invalid_multi_mode_raises():
    g = HEDMGeometry(
        Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291, multi_mode="bogus",
    )
    with pytest.raises(ValueError, match="multi_mode"):
        HEDMForwardModel(torch.zeros(1, 3), torch.zeros(1), g)


def test_tx_list_length_mismatch_raises():
    g = HEDMGeometry(
        Lsd=[1.0e6, 1.0e6], y_BC=[1024.0, 1024.0], z_BC=[1024.0, 1024.0],
        tx=[0.0, 0.0, 0.0],  # length 3 vs n_distances=2
        px=200.0, omega_start=0.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
        wavelength=0.22291,
    )
    with pytest.raises(ValueError, match="n_distances"):
        HEDMForwardModel(torch.zeros(1, 3), torch.zeros(1), g)
