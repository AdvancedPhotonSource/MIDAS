"""Hooke's law stress wrapper.

Thin shim over :func:`midas_stress.hooke.hooke_stress` so the pipeline can
emit per-grain stress in lab and grain frames consistently with the rest of
the midas-* family. We do not duplicate the stiffness / Mandel-Voigt machinery
here.

Plan: paper Appendix A; equations 13–17.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from midas_stress.hooke import hooke_stress
from midas_stress.materials import get_stiffness, cubic_stiffness


def cauchy_stress(
    strain_tensor: torch.Tensor,
    stiffness: torch.Tensor,
    *,
    orient: Optional[torch.Tensor] = None,
    frame: str = "lab",
) -> torch.Tensor:
    """Return the per-grain Cauchy stress tensor ``(3, 3)``.

    Parameters
    ----------
    strain_tensor : torch.Tensor
        ``(3, 3)`` strain tensor (lab or grain frame matching ``frame``).
    stiffness : torch.Tensor
        ``(6, 6)`` Voigt-Mandel stiffness matrix.
    orient : torch.Tensor, optional
        ``(3, 3)`` orientation matrix; required if ``frame == "lab"`` and
        the stiffness is given in the crystal frame.
    frame : {"lab", "grain"}
        Which frame ``strain_tensor`` is expressed in. ``hooke_stress``
        rotates as needed.

    Returns
    -------
    torch.Tensor
        ``(3, 3)`` Cauchy stress.
    """
    return hooke_stress(strain_tensor, stiffness, orient=orient, frame=frame)


def resolve_stiffness(
    material_name: Optional[str] = None,
    stiffness_file: Optional[str] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
) -> Optional[torch.Tensor]:
    """Pick a 6×6 stiffness matrix.

    Returns ``None`` if no source is configured (caller will skip stress
    output). Otherwise returns a torch tensor on (device, dtype).

    Precedence:
      1. ``stiffness_file`` (CSV / .npy / .txt path).
      2. ``material_name`` lookup via ``midas_stress.materials.get_stiffness``.
    """
    dev = None if device is None else torch.device(device)
    if stiffness_file is not None:
        from pathlib import Path
        p = Path(stiffness_file)
        if p.suffix == ".npy":
            mat = np.load(p)
        else:
            mat = np.loadtxt(p)
        if mat.shape != (6, 6):
            raise ValueError(
                f"Stiffness file {p} must be 6x6; got {mat.shape}"
            )
        return torch.from_numpy(mat).to(device=dev, dtype=dtype)
    if material_name is not None:
        mat = get_stiffness(material_name)
        if isinstance(mat, np.ndarray):
            return torch.from_numpy(mat).to(device=dev, dtype=dtype)
        return mat.to(device=dev, dtype=dtype)
    return None
