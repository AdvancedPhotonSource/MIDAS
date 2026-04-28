"""Cromer-Mann (IT92) X-ray atomic form factors.

f(s) = Σᵢ aᵢ exp(-bᵢ s²) + c    where s = sin(θ)/λ = 1/(2d)   [Å⁻¹]

Coefficients shipped in ``data/cromer_mann.json`` (gemmi IT92 export, neutral
atoms Z = 1..98).  Backend-agnostic: works on numpy arrays or torch tensors;
when torch is given, the result is differentiable.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = [
    "form_factor",
    "form_factor_batch",
    "available_elements",
    "coefficients",
]


# ---------------------------------------------------------------- table loading

@lru_cache(maxsize=1)
def _table() -> dict[str, dict]:
    raw = json.loads(files("midas_hkls").joinpath("data/cromer_mann.json").read_text())
    return raw


def available_elements() -> list[str]:
    return sorted(_table().keys())


_CHARGE_RE = re.compile(r"^([A-Z][a-z]?)(?:([0-9]+)?([+-]))?$")


def _normalize_symbol(symbol: str) -> str:
    """Map 'Fe', 'Fe2+', 'O2-', 'Fe(III)' → bare element 'Fe' / 'O' for IT92 lookup.

    IT92 (gemmi export) only provides neutral-atom coefficients.  Ions are
    folded onto the neutral entry with a future TODO to extend from cctbx.
    """
    s = symbol.strip()
    if not s:
        raise ValueError("empty element symbol")
    # strip Roman-numeral charge in parens, e.g. Fe(III)
    s = re.sub(r"\([IVX]+\)", "", s)
    m = _CHARGE_RE.match(s)
    if m is None:
        # last-ditch: take leading letters
        s = re.match(r"^[A-Z][a-z]?", s).group(0) if re.match(r"^[A-Z]", s) else s
        return s
    return m.group(1)


def coefficients(element: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (a, b, c) for an element, where a and b are length-4 arrays."""
    sym = _normalize_symbol(element)
    tbl = _table()
    if sym not in tbl:
        raise KeyError(f"no Cromer-Mann coefficients for element {element!r} (normalized {sym!r})")
    e = tbl[sym]
    return np.asarray(e["a"], dtype=np.float64), np.asarray(e["b"], dtype=np.float64), float(e["c"])


# --------------------------------------------------------------- backend probe

def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


# ----------------------------------------------------------------- public API

def form_factor(s2: Any, element: str) -> Any:
    """f(s) for a single element evaluated at scalar or array s² = sin²θ/λ²  [Å⁻²].

    ``s2`` may be a Python float, numpy array, or torch tensor.  The return
    type matches the input.  Torch tensors flow gradients through ``s2``.
    """
    a, b, c = coefficients(element)
    if _is_torch_tensor(s2):
        import torch
        device = s2.device
        dtype = s2.dtype
        a_t = torch.as_tensor(a, dtype=dtype, device=device)
        b_t = torch.as_tensor(b, dtype=dtype, device=device)
        c_t = torch.as_tensor(c, dtype=dtype, device=device)
        s2u = s2.unsqueeze(-1)            # (..., 1)
        return torch.sum(a_t * torch.exp(-b_t * s2u), dim=-1) + c_t

    arr = np.asarray(s2, dtype=np.float64)
    s2u = arr[..., None]
    return float((a * np.exp(-b * s2u)).sum() + c) if arr.ndim == 0 else (a * np.exp(-b * s2u)).sum(axis=-1) + c


def form_factor_batch(s2: Any, elements: Iterable[str]) -> Any:
    """f(s, atom_j) over many atoms.  Returns shape (..., N_atoms).

    ``elements`` is a length-N iterable of element symbols (one per atom);
    duplicates are allowed.  ``s2`` is broadcast against the atom axis.
    """
    el_list = list(elements)
    if not el_list:
        raise ValueError("elements must be non-empty")
    coefs = [coefficients(e) for e in el_list]
    a_stack = np.stack([c[0] for c in coefs])  # (N, 4)
    b_stack = np.stack([c[1] for c in coefs])  # (N, 4)
    c_stack = np.array([c[2] for c in coefs])  # (N,)

    if _is_torch_tensor(s2):
        import torch
        device = s2.device
        dtype = s2.dtype
        a_t = torch.as_tensor(a_stack, dtype=dtype, device=device)   # (N, 4)
        b_t = torch.as_tensor(b_stack, dtype=dtype, device=device)
        c_t = torch.as_tensor(c_stack, dtype=dtype, device=device)   # (N,)
        s2u = s2.unsqueeze(-1).unsqueeze(-1)                         # (..., 1, 1)
        terms = a_t * torch.exp(-b_t * s2u)                          # (..., N, 4)
        return terms.sum(dim=-1) + c_t                                # (..., N)

    arr = np.asarray(s2, dtype=np.float64)
    s2u = arr[..., None, None]                                        # (..., 1, 1)
    terms = a_stack * np.exp(-b_stack * s2u)                          # (..., N, 4)
    return terms.sum(axis=-1) + c_stack                               # (..., N)
