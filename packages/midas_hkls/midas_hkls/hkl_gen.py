"""Generate sorted, deduplicated, allowed HKL lists with multiplicities."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .lattice import Lattice
from .space_group import SpaceGroup


@dataclass
class Reflection:
    h: int
    k: int
    l: int
    d_spacing: float
    two_theta_deg: float
    multiplicity: int
    ring_nr: int = 0


def _bragg_hkl_max(lattice: Lattice, d_min: float) -> Tuple[int, int, int]:
    """Find a coarse upper bound on |h|, |k|, |l| such that d_hkl ≥ d_min along
    each principal direction.  Conservative: uses reciprocal-lattice norms.
    """
    Gstar = lattice.reciprocal_metric_tensor()
    # d_min => |G| <= 1/d_min along each principal axis.
    bound = []
    for i in range(3):
        # |g*_i| projected gives upper bound on Miller index
        g_ii = Gstar[i, i]
        if g_ii <= 0:
            bound.append(20)  # fallback
        else:
            bound.append(int(np.ceil(1.0 / (np.sqrt(g_ii) * d_min))) + 1)
    return tuple(bound)  # type: ignore[return-value]


def generate_hkls(
    space_group: SpaceGroup,
    lattice: Lattice,
    *,
    wavelength_A: Optional[float] = None,
    d_min: Optional[float] = None,
    two_theta_max_deg: Optional[float] = None,
    d_max: Optional[float] = None,
) -> List[Reflection]:
    """Enumerate symmetry-unique reflections in a Bragg-allowed range, sorted by d-spacing descending.

    At least one of d_min / two_theta_max_deg must be given.  If `wavelength_A`
    is provided, two_theta is computed and `two_theta_max_deg` filters apply.
    """
    if d_min is None:
        if two_theta_max_deg is None or wavelength_A is None:
            raise ValueError("specify d_min OR (two_theta_max_deg + wavelength_A)")
        s = np.sin(np.deg2rad(two_theta_max_deg / 2.0))
        if s <= 0:
            raise ValueError("two_theta_max_deg must be positive")
        d_min = wavelength_A / (2.0 * s)
    if d_min <= 0:
        raise ValueError("d_min must be positive")

    H, K, L = _bragg_hkl_max(lattice, d_min)

    seen_asu: set[Tuple[int, int, int]] = set()
    out: List[Reflection] = []

    for h in range(-H, H + 1):
        for k in range(-K, K + 1):
            for l in range(-L, L + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                d = lattice.d_spacing(h, k, l)
                if not isfinite(d) or d < d_min:
                    continue
                if d_max is not None and d > d_max:
                    continue
                if space_group.is_systematically_absent(h, k, l):
                    continue

                rep = space_group.asu_representative(h, k, l)
                if rep in seen_asu:
                    continue
                seen_asu.add(rep)

                mult = space_group.multiplicity(*rep)
                tt = (
                    lattice.two_theta_deg(*rep, wavelength_A) if wavelength_A else float("nan")
                )
                if (
                    wavelength_A is not None
                    and two_theta_max_deg is not None
                    and isfinite(tt)
                    and tt > two_theta_max_deg
                ):
                    continue
                out.append(Reflection(rep[0], rep[1], rep[2], d, tt, mult))

    out.sort(key=lambda r: -r.d_spacing)
    for i, r in enumerate(out, 1):
        r.ring_nr = i
    return out


def reflections_to_dataframe(refs: Iterable[Reflection]):
    """Optional pandas helper. Keeps pandas a soft dependency."""
    import pandas as pd

    rows = [
        {
            "ring_nr": r.ring_nr,
            "h": r.h,
            "k": r.k,
            "l": r.l,
            "d_spacing": r.d_spacing,
            "two_theta_deg": r.two_theta_deg,
            "multiplicity": r.multiplicity,
        }
        for r in refs
    ]
    return pd.DataFrame(rows)
