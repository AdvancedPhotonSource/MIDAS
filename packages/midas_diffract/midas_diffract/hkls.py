"""Build forward-model reflection lists from ``midas-hkls`` outputs.

``HEDMForwardModel`` consumes three tensors that are conventionally supplied
by ``GetHKLList`` (the C tool in MIDAS):

  * ``hkls_int``   -- (M, 3) integer Miller indices, **one row per spot**
                      (i.e. all symmetry-equivalent variants of each ASU
                      representative are enumerated)
  * ``hkls_cart``  -- (M, 3) reference Cartesian G-vectors in 1/Angstroms
  * ``thetas``     -- (M,)   reference Bragg angles in radians

This module produces the same triplet from the pure-Python ``midas-hkls``
package, so users do not need the MIDAS C build to drive the forward model.

Example
-------
    from midas_hkls import SpaceGroup, Lattice
    import midas_diffract as md

    sg = SpaceGroup.from_number(225)                # FCC (Cu/Au/Ni)
    lat = sg_lat = md.Lattice.for_system("cubic", a=4.08)  # if you re-export
    hkls_cart, thetas, hkls_int = md.hkls_for_forward_model(
        sg, lat, wavelength_A=0.172979, two_theta_max_deg=15.0,
    )
    model = md.HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
    )
"""
from __future__ import annotations

from math import cos, pi, sin
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:
    from midas_hkls import Lattice, SpaceGroup

DEG2RAD = pi / 180.0


def _cartesian_B_matrix(latc: "tuple[float, float, float, float, float, float]") -> np.ndarray:
    """Reference reciprocal-lattice basis in Cartesian coords (column = a*, b*, c*).

    Mirrors the B-matrix convention in
    :meth:`midas_diffract.forward.HEDMForwardModel.correct_hkls_latc`, which
    in turn is the C convention from ``CorrectHKLsLatC`` in
    ``FF_HEDM/src/FitPosOrStrainsDoubleDataset.c:214-252``. Keeping the
    convention bit-aligned guarantees that ``hkls_cart = B @ hkls_int^T``
    here matches the model's strain path, so passing ``lattice_params=`` at
    forward time recomputes the same numbers up to floating-point error.
    """
    a, b, c, alpha_d, beta_d, gamma_d = latc
    alpha = alpha_d * DEG2RAD
    beta = beta_d * DEG2RAD
    gamma = gamma_d * DEG2RAD
    sin_a, cos_a = sin(alpha), cos(alpha)
    sin_b, cos_b = sin(beta), cos(beta)
    sin_g, cos_g = sin(gamma), cos(gamma)

    eps = 1e-7
    gamma_pr = np.arccos(np.clip(
        (cos_a * cos_b - cos_g) / (sin_a * sin_b + eps), -1 + eps, 1 - eps,
    ))
    beta_pr = np.arccos(np.clip(
        (cos_g * cos_a - cos_b) / (sin_g * sin_a + eps), -1 + eps, 1 - eps,
    ))
    sin_beta_pr = np.sin(beta_pr)

    vol = a * b * c * sin_a * sin_beta_pr * sin_g
    a_pr = b * c * sin_a / (vol + eps)
    b_pr = c * a * sin_b / (vol + eps)
    c_pr = a * b * sin_g / (vol + eps)

    B = np.array([
        [a_pr, b_pr * np.cos(gamma_pr), c_pr * np.cos(beta_pr)],
        [0.0,  b_pr * np.sin(gamma_pr), -c_pr * sin_beta_pr * cos_a],
        [0.0,  0.0,                      c_pr * sin_beta_pr * sin_a],
    ])
    return B


def hkls_for_forward_model(
    space_group: "SpaceGroup",
    lattice: "Lattice",
    *,
    wavelength_A: float,
    two_theta_max_deg: Optional[float] = None,
    d_min: Optional[float] = None,
    expand_equivalents: bool = True,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (``hkls_cart``, ``thetas``, ``hkls_int``) for ``HEDMForwardModel``.

    Wraps :func:`midas_hkls.generate_hkls` -- which returns ASU
    representatives -- and (by default) expands each to all
    Laue-equivalent integer Miller indices, so every detector spot is
    enumerated. Then computes the Cartesian G-vectors using a B-matrix
    convention that is consistent with the forward model's internal
    strain-recompute path.

    Parameters
    ----------
    space_group, lattice
        From the ``midas-hkls`` package.
    wavelength_A : float
        X-ray wavelength in Angstroms.
    two_theta_max_deg, d_min
        Cutoff for reflection enumeration. At least one must be supplied.
        See :func:`midas_hkls.generate_hkls`.
    expand_equivalents : bool, default True
        If True, return one row per Laue-equivalent reflection (matches
        ``GetHKLList`` output and is what the forward model expects). If
        False, return only ASU representatives -- useful for diagnostics.
    dtype : torch.dtype
        Output tensor dtype. Defaults to float64; the model casts to
        float32 internally for the buffers but keeps double precision in
        the input pipeline if requested.

    Returns
    -------
    hkls_cart : Tensor (M, 3)
        Cartesian reciprocal-space G-vectors in 1/Angstroms.
    thetas : Tensor (M,)
        Bragg angles in radians.
    hkls_int : Tensor (M, 3)
        Integer Miller indices (one row per spot), as floats so they can
        be moved through ``torch.matmul`` cleanly inside the model.
    """
    try:
        from midas_hkls import generate_hkls  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "midas_diffract.hkls requires the optional 'midas-hkls' package. "
            "Install with: pip install midas-hkls"
        ) from exc

    refs = generate_hkls(
        space_group,
        lattice,
        wavelength_A=wavelength_A,
        two_theta_max_deg=two_theta_max_deg,
        d_min=d_min,
    )
    if not refs:
        raise ValueError(
            "midas_hkls.generate_hkls returned no reflections; check "
            "wavelength / cutoff arguments."
        )

    rows = []
    for r in refs:
        if expand_equivalents:
            rows.extend(space_group.equivalent_hkls(r.h, r.k, r.l))
        else:
            rows.append((r.h, r.k, r.l))
    hkls_int_np = np.asarray(rows, dtype=np.float64)

    B = _cartesian_B_matrix(
        (lattice.a, lattice.b, lattice.c,
         lattice.alpha, lattice.beta, lattice.gamma)
    )
    G_cart = hkls_int_np @ B.T  # (M, 3) Cartesian G in 1/A

    g_mag = np.linalg.norm(G_cart, axis=-1)
    s = g_mag * wavelength_A / 2.0
    if np.any(s > 1.0):
        bad = int(np.sum(s > 1.0))
        raise ValueError(
            f"{bad} reflections fall outside the Bragg cutoff (|G|*lambda/2 > 1) "
            "for the requested cutoff -- tighten two_theta_max_deg / d_min."
        )
    thetas_np = np.arcsin(s)

    return (
        torch.tensor(G_cart, dtype=dtype),
        torch.tensor(thetas_np, dtype=dtype),
        torch.tensor(hkls_int_np, dtype=dtype),
    )
