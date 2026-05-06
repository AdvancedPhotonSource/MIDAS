"""Lab-frame Bragg geometry for the per-spot strain solver.

For each matched spot we have an observed lab-frame detector position
``(y_obs, z_obs)`` (µm), a sample-rotation angle ``ω_obs`` (deg), the
sample-to-detector distance ``Lsd`` (µm), and the X-ray wavelength
``λ`` (Å). From these we compute:

  * The observed g-vector direction in the lab frame at ``ω_obs``::

        ĝ_lab = (k_out − k_in) / |k_out − k_in|

    where ``k_in = (1, 0, 0)`` (incident) and ``k_out`` is the unit vector
    from the sample to the detected spot.

  * The observed d-spacing via Bragg's law::

        2θ = atan2(sqrt(y² + z²), Lsd)
        d_obs = λ / (2 sin θ)

These two quantities are exactly what ``solve_strain_lstsq`` needs (paper
Eq. 8-11).

Notes
-----
* We deliberately keep the geometry self-contained here rather than calling
  ``midas_diffract.HEDMForwardModel`` because that's the *forward* path
  (orientation + position + lattice → predicted detector positions). For
  per-spot strain we need the *inverse* — observed detector positions →
  observed g and d. The math is a few lines and avoids a heavy dependency
  for this small subset of work.

* Wedge / detector tilts are assumed already corrected upstream (the
  observed (y, z) in ``FitBest.bin`` columns 1, 2 has already been
  flattened to the lab plane by ``FitPosOrStrainsOMP``). For non-zero
  wedge the rotation about the wedge axis adds a small correction that the
  current strain test data doesn't exercise; left as a TODO.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def lab_obs_to_g_and_d(
    y_obs: torch.Tensor,
    z_obs: torch.Tensor,
    lsd: float,
    wavelength_a: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert lab-frame ``(y, z)`` to ``(ĝ_lab, d_obs)``.

    Parameters
    ----------
    y_obs, z_obs : torch.Tensor
        ``(n,)`` observed lab-frame detector positions (µm). Sign convention
        matches ``FitBest.bin`` columns 1 and 2.
    lsd : float
        Sample-to-detector distance (µm).
    wavelength_a : float
        Wavelength (Å).

    Returns
    -------
    g_hat : torch.Tensor
        ``(n, 3)`` unit g-vector in lab frame at the diffraction event.
    d_obs : torch.Tensor
        ``(n,)`` observed d-spacing (Å).
    """
    if y_obs.shape != z_obs.shape:
        raise ValueError(
            f"y and z must have the same shape; got {y_obs.shape} and {z_obs.shape}"
        )
    common_dtype = torch.promote_types(y_obs.dtype, z_obs.dtype)
    y = y_obs.to(common_dtype)
    z = z_obs.to(common_dtype)

    # k_out = (Lsd, y, z) / |(Lsd, y, z)|
    L = torch.full_like(y, fill_value=float(lsd))
    norm = torch.sqrt(L * L + y * y + z * z)
    norm = torch.clamp(norm, min=1e-30)
    k_out = torch.stack([L / norm, y / norm, z / norm], dim=-1)            # (n, 3)
    k_in = torch.zeros_like(k_out)
    k_in[..., 0] = 1.0
    delta = k_out - k_in                                                    # (n, 3)
    g_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    g_norm = torch.clamp(g_norm, min=1e-30)
    g_hat = delta / g_norm

    # 2θ = angle between k_in and k_out, equivalently atan2(sqrt(y²+z²), Lsd)
    rho = torch.sqrt(y * y + z * z)
    two_theta = torch.atan2(rho, L)
    sin_theta = torch.sin(two_theta * 0.5)
    sin_theta = torch.clamp(sin_theta, min=1e-30)
    d_obs = wavelength_a / (2.0 * sin_theta)

    return g_hat, d_obs


def numpy_lab_obs_to_g_and_d(
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    lsd: float,
    wavelength_a: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy convenience wrapper around :func:`lab_obs_to_g_and_d`."""
    g_t, d_t = lab_obs_to_g_and_d(
        torch.from_numpy(np.asarray(y_obs, dtype=np.float64)),
        torch.from_numpy(np.asarray(z_obs, dtype=np.float64)),
        float(lsd),
        float(wavelength_a),
    )
    return g_t.numpy(), d_t.numpy()
