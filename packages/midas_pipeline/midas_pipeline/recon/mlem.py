"""MLEM / OS-EM sinogram reconstruction (port of utils/mlem_recon.py).

Same numerical recipe as the legacy ``utils/mlem_recon.py`` (the
original file stays in place untouched), with one structural upgrade:
the forward/back projector and the iterative loop are dispatched on
input type. Tensor inputs flow through torch with autograd intact and
run on CUDA / MPS / CPU according to the input device; ndarray inputs
use the legacy NumPy path verbatim.

The dispatch follows the canonical pattern in
``midas_stress.orientation`` / ``midas_stress.diffraction``:

    if isinstance(x, torch.Tensor): torch path
    else:                            numpy path

Both paths share the same iteration math; only the array library
differs. The torch path is correct for autograd because every
operation is differentiable (no boolean indexing on the output, no
in-place updates on parameters, no ``.detach()``).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

try:  # torch is an optional runtime dependency of the package
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is in pyproject deps
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ---------------------------------------------------------------------------
# NumPy projectors (lifted verbatim from utils/mlem_recon.py)
# ---------------------------------------------------------------------------


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def _forward_project_np(image: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    N = image.shape[0]
    assert image.shape[1] == N, "Image must be square"
    n_thetas = len(angles_deg)
    sino = np.zeros((n_thetas, N), dtype=np.float64)

    center = (N - 1) / 2.0
    x = np.arange(N) - center

    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        for j, t in enumerate(x):
            s_vals = x
            ray_x = t * cos_a - s_vals * sin_a + center
            ray_y = t * sin_a + s_vals * cos_a + center
            ix = np.floor(ray_x).astype(int)
            iy = np.floor(ray_y).astype(int)
            fx = ray_x - ix
            fy = ray_y - iy
            valid = (ix >= 0) & (ix < N - 1) & (iy >= 0) & (iy < N - 1)
            ix = np.clip(ix, 0, N - 2)
            iy = np.clip(iy, 0, N - 2)
            vals = (
                image[iy, ix] * (1 - fx) * (1 - fy)
                + image[iy, ix + 1] * fx * (1 - fy)
                + image[iy + 1, ix] * (1 - fx) * fy
                + image[iy + 1, ix + 1] * fx * fy
            )
            sino[i, j] = np.sum(vals * valid)
    return sino


def _back_project_np(sinogram: np.ndarray, angles_deg: np.ndarray, N: int) -> np.ndarray:
    M = sinogram.shape[1]
    image = np.zeros((N, N), dtype=np.float64)
    center_img = (N - 1) / 2.0
    center_det = (M - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N]
    xx = xx.astype(np.float64) - center_img
    yy = yy.astype(np.float64) - center_img
    for i, angle in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        t = xx * cos_a + yy * sin_a + center_det
        it = np.floor(t).astype(int)
        ft = t - it
        valid = (it >= 0) & (it < M - 1)
        it_safe = np.clip(it, 0, M - 2)
        vals = sinogram[i, it_safe] * (1 - ft) + sinogram[i, it_safe + 1] * ft
        image += vals * valid
    return image


# ---------------------------------------------------------------------------
# Torch projectors (vectorized, differentiable, device-portable)
# ---------------------------------------------------------------------------


def _forward_project_torch(image: "torch.Tensor", angles_deg: "torch.Tensor") -> "torch.Tensor":
    """Torch radon transform.

    Vectorized over angles & detector pixels. Operates on the image
    parameter's dtype/device. Bilinear interpolation matches the
    numpy path to within float precision; autograd-safe because we
    never index in-place.
    """
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Image must be 2-D and square")
    dtype = image.dtype
    device = image.device
    N = image.shape[0]
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    n_thetas = angles_deg.shape[0]

    center = (N - 1) / 2.0
    coord = torch.arange(N, dtype=dtype, device=device) - center  # (N,)
    cos_a = torch.cos(torch.deg2rad(angles_deg))  # (T,)
    sin_a = torch.sin(torch.deg2rad(angles_deg))

    # Ray coords: (T, N_det, N_int) → world (ray_x, ray_y)
    t = coord.view(1, -1, 1)          # (1, N, 1)
    s = coord.view(1, 1, -1)          # (1, 1, N)
    ca = cos_a.view(-1, 1, 1)         # (T, 1, 1)
    sa = sin_a.view(-1, 1, 1)
    ray_x = t * ca - s * sa + center  # (T, N, N)
    ray_y = t * sa + s * ca + center

    # Bilinear interpolation
    ix0 = torch.floor(ray_x).long()
    iy0 = torch.floor(ray_y).long()
    fx = ray_x - ix0.to(dtype)
    fy = ray_y - iy0.to(dtype)

    in_bounds = (ix0 >= 0) & (ix0 < N - 1) & (iy0 >= 0) & (iy0 < N - 1)
    ix_safe = ix0.clamp(0, N - 2)
    iy_safe = iy0.clamp(0, N - 2)

    # Linear-index into a 1-D view of the image so we never index image with
    # out-of-bounds tensors (would error or produce nondeterministic backward).
    flat = image.reshape(-1)
    i00 = (iy_safe * N + ix_safe).reshape(-1)
    i01 = (iy_safe * N + (ix_safe + 1)).reshape(-1)
    i10 = ((iy_safe + 1) * N + ix_safe).reshape(-1)
    i11 = ((iy_safe + 1) * N + (ix_safe + 1)).reshape(-1)
    v00 = flat[i00].reshape(ix_safe.shape)
    v01 = flat[i01].reshape(ix_safe.shape)
    v10 = flat[i10].reshape(ix_safe.shape)
    v11 = flat[i11].reshape(ix_safe.shape)

    sample = (
        v00 * (1 - fx) * (1 - fy)
        + v01 * fx * (1 - fy)
        + v10 * (1 - fx) * fy
        + v11 * fx * fy
    )
    sample = sample * in_bounds.to(dtype)
    sino = sample.sum(dim=-1)  # (T, N)
    return sino


def _back_project_torch(sinogram: "torch.Tensor", angles_deg: "torch.Tensor", N: int) -> "torch.Tensor":
    dtype = sinogram.dtype
    device = sinogram.device
    M = sinogram.shape[1]
    angles_deg = angles_deg.to(device=device, dtype=dtype)
    center_img = (N - 1) / 2.0
    center_det = (M - 1) / 2.0

    grid = torch.arange(N, dtype=dtype, device=device)
    yy = grid.view(-1, 1).expand(N, N) - center_img
    xx = grid.view(1, -1).expand(N, N) - center_img

    cos_a = torch.cos(torch.deg2rad(angles_deg))  # (T,)
    sin_a = torch.sin(torch.deg2rad(angles_deg))

    image = torch.zeros((N, N), dtype=dtype, device=device)
    for i in range(angles_deg.shape[0]):
        t = xx * cos_a[i] + yy * sin_a[i] + center_det
        it0 = torch.floor(t).long()
        ft = t - it0.to(dtype)
        in_bounds = (it0 >= 0) & (it0 < M - 1)
        it_safe = it0.clamp(0, M - 2)
        row = sinogram[i]
        v_lo = row[it_safe]
        v_hi = row[(it_safe + 1).clamp(0, M - 1)]
        vals = v_lo * (1 - ft) + v_hi * ft
        image = image + vals * in_bounds.to(dtype)
    return image


# ---------------------------------------------------------------------------
# Public API: forward_project / back_project / mlem / osem (dispatch)
# ---------------------------------------------------------------------------


def forward_project(image: ArrayLike, angles_deg: ArrayLike) -> ArrayLike:
    """Radon transform.

    Tensor input → tensor output on input's device/dtype, autograd
    intact. ndarray input → ndarray output via the legacy NumPy path.
    """
    if _TORCH_AVAILABLE and isinstance(image, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=image.dtype, device=image.device)
        )
        return _forward_project_torch(image, angles_t)
    return _forward_project_np(np.asarray(image), np.asarray(angles_deg))


def back_project(sinogram: ArrayLike, angles_deg: ArrayLike, N: int) -> ArrayLike:
    if _TORCH_AVAILABLE and isinstance(sinogram, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=sinogram.dtype, device=sinogram.device)
        )
        return _back_project_torch(sinogram, angles_t, N)
    return _back_project_np(np.asarray(sinogram), np.asarray(angles_deg), N)


# ----- NumPy MLEM/OSEM (verbatim copy of utils/mlem_recon.py logic) ---------


def _mlem_np(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    n_iter: int,
    init: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    n_thetas, M = sinogram.shape
    N = M
    if mask is None:
        row_has_data = np.any(sinogram > 0, axis=1)
    else:
        row_has_data = np.any(mask, axis=1)
    valid_idx = np.where(row_has_data)[0]
    sino_valid = sinogram[valid_idx]
    angles_valid = angles_deg[valid_idx]
    if len(valid_idx) == 0:
        return np.zeros((N, N))

    mask_sino = (sino_valid > 0).astype(np.float64)
    sensitivity = _back_project_np(mask_sino, angles_valid, N)
    image_support = sensitivity > eps
    sensitivity = np.maximum(sensitivity, eps)

    if init is not None:
        estimate = init.astype(np.float64).copy()
    else:
        estimate = np.ones((N, N), dtype=np.float64)
    estimate = np.where(image_support, estimate, 0.0)

    sino_scale = float(sino_valid.max()) if sino_valid.size else 1.0
    proj_floor = max(eps, 1e-6 * sino_scale)
    UPD_LO, UPD_HI = 0.1, 10.0
    for _ in range(n_iter):
        proj = _forward_project_np(estimate, angles_valid)
        proj = np.maximum(proj, proj_floor)
        ratio = np.where(mask_sino > 0, sino_valid / proj, 0.0)
        correction = _back_project_np(ratio, angles_valid, N)
        update = correction / sensitivity
        update = np.clip(update, UPD_LO, UPD_HI)
        estimate = np.where(image_support, estimate * update, 0.0)
    return np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)


def _mlem_torch(
    sinogram: "torch.Tensor",
    angles_deg: "torch.Tensor",
    n_iter: int,
    init: Optional["torch.Tensor"],
    eps: float,
) -> "torch.Tensor":
    """Differentiable, multi-device MLEM.

    The estimate stays a Tensor on the sinogram's device the entire
    way through; every step is differentiable so the chain rule
    follows the iteration count. No ``.cpu().numpy()`` round-trips.
    """
    dtype = sinogram.dtype
    device = sinogram.device
    n_thetas, M = sinogram.shape
    N = M
    angles_deg = angles_deg.to(device=device, dtype=dtype)

    # Row mask: any cell > 0 in that row.
    row_has_data = (sinogram > 0).any(dim=1)
    if not row_has_data.any():
        return torch.zeros((N, N), dtype=dtype, device=device)
    valid_idx = torch.nonzero(row_has_data, as_tuple=False).reshape(-1)
    sino_valid = sinogram.index_select(0, valid_idx)
    angles_valid = angles_deg.index_select(0, valid_idx)

    mask_sino = (sino_valid > 0).to(dtype)
    sensitivity = _back_project_torch(mask_sino, angles_valid, N)
    image_support = sensitivity > eps
    sensitivity = torch.clamp(sensitivity, min=eps)

    if init is None:
        estimate = torch.ones((N, N), dtype=dtype, device=device)
    else:
        estimate = init.to(device=device, dtype=dtype).clone()
    estimate = torch.where(image_support, estimate, torch.zeros_like(estimate))

    sino_scale = float(sino_valid.max().item()) if sino_valid.numel() else 1.0
    proj_floor = max(eps, 1e-6 * sino_scale)
    UPD_LO, UPD_HI = 0.1, 10.0
    for _ in range(n_iter):
        proj = _forward_project_torch(estimate, angles_valid)
        proj = torch.clamp(proj, min=proj_floor)
        ratio = torch.where(mask_sino > 0, sino_valid / proj, torch.zeros_like(sino_valid))
        correction = _back_project_torch(ratio, angles_valid, N)
        update = correction / sensitivity
        update = torch.clamp(update, UPD_LO, UPD_HI)
        estimate = torch.where(image_support, estimate * update, torch.zeros_like(estimate))
    estimate = torch.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
    return estimate


def mlem_recon(
    sinogram: ArrayLike,
    angles_deg: ArrayLike,
    *,
    n_iter: int = 50,
    init: Optional[ArrayLike] = None,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-10,
) -> ArrayLike:
    """Maximum Likelihood Expectation Maximization reconstruction.

    Same recipe as the legacy ``utils/mlem_recon.py``. Tensor input
    flows through the torch path with autograd intact on the input's
    device; ndarray input runs the legacy NumPy path.

    Parameters
    ----------
    sinogram : ndarray (T, M) or torch.Tensor (T, M)
    angles_deg : ndarray (T,) or torch.Tensor (T,)
    n_iter : int
    init : optional initial estimate (M, M); same type as ``sinogram``.
    mask : numpy bool array; numpy path only (ignored on torch path,
        whose mask is derived from ``sinogram > 0``).
    eps : small positive constant.

    Returns
    -------
    Same array type as the input ``sinogram``.
    """
    if _TORCH_AVAILABLE and isinstance(sinogram, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=sinogram.dtype, device=sinogram.device)
        )
        init_t = (
            init
            if init is None or isinstance(init, torch.Tensor)
            else torch.as_tensor(init, dtype=sinogram.dtype, device=sinogram.device)
        )
        return _mlem_torch(sinogram, angles_t, n_iter, init_t, eps)
    return _mlem_np(
        np.asarray(sinogram, dtype=np.float64),
        np.asarray(angles_deg, dtype=np.float64),
        n_iter,
        None if init is None else np.asarray(init, dtype=np.float64),
        mask,
        eps,
    )


def _osem_np(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    n_iter: int,
    n_subsets: int,
    init: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    n_thetas, M = sinogram.shape
    N = M
    row_has_data = np.any(sinogram > 0, axis=1)
    valid_idx = np.where(row_has_data)[0]
    if len(valid_idx) == 0:
        return np.zeros((N, N))
    subsets = [valid_idx[i::n_subsets] for i in range(n_subsets)]
    if init is not None:
        estimate = init.astype(np.float64).copy()
    else:
        estimate = np.ones((N, N), dtype=np.float64)
    full_mask = (sinogram[valid_idx] > 0).astype(np.float64)
    full_sensitivity = _back_project_np(full_mask, angles_deg[valid_idx], N)
    image_support = full_sensitivity > eps
    estimate = np.where(image_support, estimate, 0.0)
    sino_scale = float(sinogram[valid_idx].max()) if len(valid_idx) else 1.0
    proj_floor = max(eps, 1e-6 * sino_scale)
    UPD_LO, UPD_HI = 0.1, 10.0
    for _ in range(n_iter):
        for subset_idx in subsets:
            sino_sub = sinogram[subset_idx]
            angles_sub = angles_deg[subset_idx]
            mask_sub = (sino_sub > 0).astype(np.float64)
            sensitivity = _back_project_np(mask_sub, angles_sub, N)
            sensitivity = np.maximum(sensitivity, eps)
            proj = _forward_project_np(estimate, angles_sub)
            proj = np.maximum(proj, proj_floor)
            ratio = np.where(mask_sub > 0, sino_sub / proj, 0.0)
            correction = _back_project_np(ratio, angles_sub, N)
            update = correction / sensitivity
            update = np.clip(update, UPD_LO, UPD_HI)
            estimate = np.where(image_support, estimate * update, 0.0)
    return np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)


def _osem_torch(
    sinogram: "torch.Tensor",
    angles_deg: "torch.Tensor",
    n_iter: int,
    n_subsets: int,
    init: Optional["torch.Tensor"],
    eps: float,
) -> "torch.Tensor":
    dtype = sinogram.dtype
    device = sinogram.device
    n_thetas, M = sinogram.shape
    N = M
    angles_deg = angles_deg.to(device=device, dtype=dtype)

    row_has_data = (sinogram > 0).any(dim=1)
    if not row_has_data.any():
        return torch.zeros((N, N), dtype=dtype, device=device)
    valid_idx = torch.nonzero(row_has_data, as_tuple=False).reshape(-1)
    full_mask = (sinogram.index_select(0, valid_idx) > 0).to(dtype)
    full_sensitivity = _back_project_torch(
        full_mask, angles_deg.index_select(0, valid_idx), N
    )
    image_support = full_sensitivity > eps

    if init is None:
        estimate = torch.ones((N, N), dtype=dtype, device=device)
    else:
        estimate = init.to(device=device, dtype=dtype).clone()
    estimate = torch.where(image_support, estimate, torch.zeros_like(estimate))
    sino_scale = float(sinogram.index_select(0, valid_idx).max().item()) if valid_idx.numel() else 1.0
    proj_floor = max(eps, 1e-6 * sino_scale)
    UPD_LO, UPD_HI = 0.1, 10.0

    # interleaved subsets
    subsets = [valid_idx[i::n_subsets] for i in range(n_subsets)]
    for _ in range(n_iter):
        for subset_idx in subsets:
            if subset_idx.numel() == 0:
                continue
            sino_sub = sinogram.index_select(0, subset_idx)
            angles_sub = angles_deg.index_select(0, subset_idx)
            mask_sub = (sino_sub > 0).to(dtype)
            sensitivity = _back_project_torch(mask_sub, angles_sub, N)
            sensitivity = torch.clamp(sensitivity, min=eps)
            proj = _forward_project_torch(estimate, angles_sub)
            proj = torch.clamp(proj, min=proj_floor)
            ratio = torch.where(mask_sub > 0, sino_sub / proj, torch.zeros_like(sino_sub))
            correction = _back_project_torch(ratio, angles_sub, N)
            update = correction / sensitivity
            update = torch.clamp(update, UPD_LO, UPD_HI)
            estimate = torch.where(image_support, estimate * update, torch.zeros_like(estimate))
    return torch.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)


def osem_recon(
    sinogram: ArrayLike,
    angles_deg: ArrayLike,
    *,
    n_iter: int = 10,
    n_subsets: int = 4,
    init: Optional[ArrayLike] = None,
    eps: float = 1e-10,
) -> ArrayLike:
    """Ordered Subsets Expectation Maximization (accelerated MLEM).

    Tensor in → Tensor out (autograd intact, multi-device). ndarray
    in → ndarray out via the legacy NumPy path.
    """
    if _TORCH_AVAILABLE and isinstance(sinogram, torch.Tensor):
        angles_t = (
            angles_deg
            if isinstance(angles_deg, torch.Tensor)
            else torch.as_tensor(angles_deg, dtype=sinogram.dtype, device=sinogram.device)
        )
        init_t = (
            init
            if init is None or isinstance(init, torch.Tensor)
            else torch.as_tensor(init, dtype=sinogram.dtype, device=sinogram.device)
        )
        return _osem_torch(sinogram, angles_t, n_iter, n_subsets, init_t, eps)
    return _osem_np(
        np.asarray(sinogram, dtype=np.float64),
        np.asarray(angles_deg, dtype=np.float64),
        n_iter,
        n_subsets,
        None if init is None else np.asarray(init, dtype=np.float64),
        eps,
    )


# Back-compat aliases (mirrors legacy ``mlem_recon.mlem`` / ``.osem``)
mlem = mlem_recon
osem = osem_recon
