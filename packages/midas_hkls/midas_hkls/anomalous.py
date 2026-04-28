"""Anomalous (resonant) scattering corrections f', f''.

Linear interpolation in log(energy) over a precomputed Cromer-Liberman grid
shipped at ``data/anomalous_cl.json``.  Differentiable through ``wavelength_A``
when the wavelength is a torch tensor.

Energy/wavelength conversion: E [eV] = 12398.4 / λ [Å].

Public API::

    from midas_hkls.anomalous import anomalous_correction
    fp, fpp = anomalous_correction(["Fe", "O"], wavelength_A=0.173)

Returns torch tensors of shape (N_atoms,) when torch is available; numpy
arrays otherwise.  ``wavelength_A`` may be a Python float or a torch tensor;
in the latter case gradients flow through the linear interpolation.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "anomalous_correction",
    "wavelength_to_energy_eV",
    "energy_eV_to_wavelength",
    "available_elements_anomalous",
]


_HC_eV_A = 12398.419843320026          # h·c in eV·Å (CODATA-2018)


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(files("midas_hkls").joinpath("data/anomalous_cl.json").read_text())


@lru_cache(maxsize=1)
def _energies() -> np.ndarray:
    return np.asarray(_table()["_energy_eV"], dtype=np.float64)


_CHARGE_RE = re.compile(r"^([A-Z][a-z]?)(?:[0-9]*[+-]?)?$")


def _normalize_symbol(symbol: str) -> str:
    s = symbol.strip()
    if not s:
        raise ValueError("empty element symbol")
    s = re.sub(r"\([IVX]+\)", "", s)
    m = _CHARGE_RE.match(s)
    return m.group(1) if m else s


def available_elements_anomalous() -> list[str]:
    return sorted(k for k in _table().keys() if not k.startswith("_"))


def wavelength_to_energy_eV(wavelength_A):
    """E [eV] = 12398.4 / λ [Å].  Differentiable through tensors."""
    return _HC_eV_A / wavelength_A


def energy_eV_to_wavelength(energy_eV):
    """λ [Å] = 12398.4 / E [eV].  Differentiable through tensors."""
    return _HC_eV_A / energy_eV


# -------------------------------------------------------------- core lookup

def _is_torch(x) -> bool:
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def anomalous_correction(elements, wavelength_A, *, dtype=None, device=None):
    """Return (fp, fpp), each of shape (len(elements),).

    ``wavelength_A`` may be a Python float / numpy scalar / torch scalar tensor
    (a single common wavelength for all elements).  If you need per-element
    wavelengths, pass a tensor of length len(elements).

    The interpolation is linear in log(E) — small kink at each table node, but
    everywhere differentiable in λ for gradient-based fitting.
    """
    elements = list(elements)
    if not elements:
        raise ValueError("elements must be non-empty")

    norm = [_normalize_symbol(e) for e in elements]
    table = _table()
    for sym, raw in zip(norm, elements):
        if sym not in table:
            raise KeyError(f"no anomalous data for element {raw!r} (normalized {sym!r})")

    fp_grid = np.stack([np.asarray(table[s]["fp"], dtype=np.float64) for s in norm])
    fpp_grid = np.stack([np.asarray(table[s]["fpp"], dtype=np.float64) for s in norm])
    log_E_grid = np.log(_energies())                           # (E_n,)

    use_torch = _is_torch(wavelength_A) or dtype is not None or device is not None
    if use_torch:
        import torch
        if not _is_torch(wavelength_A):
            wavelength_A = torch.as_tensor(float(wavelength_A), dtype=dtype, device=device)
        if dtype is None:
            dtype = wavelength_A.dtype
        if device is None:
            device = wavelength_A.device

        E = _HC_eV_A / wavelength_A                            # (broadcast OK)
        log_E = torch.log(E)

        log_E_grid_t = torch.as_tensor(log_E_grid, dtype=dtype, device=device)
        fp_grid_t = torch.as_tensor(fp_grid, dtype=dtype, device=device)         # (N, E_n)
        fpp_grid_t = torch.as_tensor(fpp_grid, dtype=dtype, device=device)

        fp = _torch_lerp_along_grid(log_E, log_E_grid_t, fp_grid_t)              # (N,)
        fpp = _torch_lerp_along_grid(log_E, log_E_grid_t, fpp_grid_t)
        return fp, fpp

    # numpy path
    E = float(_HC_eV_A / float(wavelength_A))
    log_E = float(np.log(E))
    fp = np.array([_np_lerp(log_E, log_E_grid, fp_grid[i]) for i in range(len(elements))],
                  dtype=np.float64)
    fpp = np.array([_np_lerp(log_E, log_E_grid, fpp_grid[i]) for i in range(len(elements))],
                   dtype=np.float64)
    return fp, fpp


def _np_lerp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    """Linear interp with extrapolation clamped at the end-points."""
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    # np.interp is linear-in-x; we want linear-in-log(x), but xp here is already log(E).
    return float(np.interp(x, xp, fp))


def _torch_lerp_along_grid(x_query, x_grid, y_grid):
    """Linear interpolation of ``y_grid[i]`` (shape (N, E_n)) along the energy
    axis at the (scalar or broadcastable) ``x_query``.  Returns shape (N,).
    Differentiable through ``x_query``."""
    import torch

    # x_query may be a 0-d tensor; clamp to grid range, then locate the bin.
    x_clamped = torch.clamp(x_query, min=x_grid[0], max=x_grid[-1])
    # bucket index: largest i such that x_grid[i] <= x_clamped
    idx = torch.searchsorted(x_grid, x_clamped.unsqueeze(-1)).squeeze(-1)
    idx = torch.clamp(idx, min=1, max=len(x_grid) - 1)
    i_lo = idx - 1
    i_hi = idx
    x_lo = x_grid[i_lo]
    x_hi = x_grid[i_hi]
    t = (x_clamped - x_lo) / (x_hi - x_lo)                                       # (), grad-safe
    y_lo = y_grid[:, i_lo]                                                        # (N,)
    y_hi = y_grid[:, i_hi]
    return y_lo + t * (y_hi - y_lo)


# -------------------------------------------------------- convenience helpers

def Z_for(element: str) -> int:
    """Return atomic number for normalized element symbol."""
    from .form_factors import _table as _cm_table  # reuse element list
    sym = _normalize_symbol(element)
    e = _cm_table().get(sym)
    if e is None:
        raise KeyError(f"unknown element {element!r}")
    return int(e["Z"])
