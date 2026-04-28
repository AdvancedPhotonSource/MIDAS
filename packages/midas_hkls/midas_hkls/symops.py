"""Symmetry operations: integer Seitz matrices over the STBF=12 translation base.

A SymOp is (R, t) with R in Z^{3x3} and t in (Z mod STBF)^3 representing the
fractional translation t_i / STBF.  All composition / inversion / equality is
done in integer arithmetic so we never accidentally classify a screw as a pure
rotation due to floating-point fuzz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from .tables import STBF


@dataclass(frozen=True)
class SymOp:
    R: Tuple[int, ...]   # 9 ints, row-major
    t: Tuple[int, ...]   # 3 ints, mod STBF

    @classmethod
    def identity(cls) -> "SymOp":
        return cls((1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0))

    @classmethod
    def inversion(cls) -> "SymOp":
        return cls((-1, 0, 0, 0, -1, 0, 0, 0, -1), (0, 0, 0))

    @classmethod
    def from_R_t(cls, R: Sequence[int], t: Sequence[int]) -> "SymOp":
        return cls(tuple(int(x) for x in R), tuple(int(x) % STBF for x in t))

    def determinant(self) -> int:
        a, b, c, d, e, f, g, h, i = self.R
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    def trace(self) -> int:
        return self.R[0] + self.R[4] + self.R[8]

    def is_proper(self) -> bool:
        return self.determinant() == 1

    def order(self) -> int:
        """Rotation order from determinant + trace (standard crystallographic table)."""
        det = self.determinant()
        tr = self.trace()
        return _ORDER_TABLE[(det, tr)]

    def compose(self, other: "SymOp") -> "SymOp":
        Ra = np.array(self.R, dtype=int).reshape(3, 3)
        Rb = np.array(other.R, dtype=int).reshape(3, 3)
        ta = np.array(self.t, dtype=int)
        tb = np.array(other.t, dtype=int)
        R = Ra @ Rb
        t = (Ra @ tb + ta) % STBF
        return SymOp.from_R_t(R.flatten().tolist(), t.tolist())

    def inverse(self) -> "SymOp":
        Ra = np.array(self.R, dtype=int).reshape(3, 3)
        det = int(round(np.linalg.det(Ra)))
        if det not in (1, -1):
            raise ValueError(f"Non-unimodular rotation in symmetry op: det={det}")
        # Cofactor / det inverse for integer SO(3) / O(3).
        Rinv = np.round(np.linalg.inv(Ra) * det).astype(int) * det
        ta = np.array(self.t, dtype=int)
        tinv = (-(Rinv @ ta)) % STBF
        return SymOp.from_R_t(Rinv.flatten().tolist(), tinv.tolist())

    def apply_hkl(self, h: int, k: int, l: int) -> Tuple[int, int, int]:
        """Reciprocal-space transformation: h' = R^T h.

        Phase shift information lives in the translation but is not required
        for absence/equivalence enumeration on the rotation alone.
        """
        R = self.R
        # Applying R^T to (h,k,l): (R^T)_{ij} h_j = R_{ji} h_j
        h2 = R[0]*h + R[3]*k + R[6]*l
        k2 = R[1]*h + R[4]*k + R[7]*l
        l2 = R[2]*h + R[5]*k + R[8]*l
        return (h2, k2, l2)

    def to_xyz(self) -> str:
        """Render as Jones-faithful representation 'x+1/2,-y,z'."""
        coords = []
        for row in range(3):
            terms = []
            for col, var in enumerate("xyz"):
                v = self.R[3 * row + col]
                if v == 0:
                    continue
                sign = "+" if v > 0 else "-"
                mag = abs(v)
                core = var if mag == 1 else f"{mag}{var}"
                terms.append(f"{sign}{core}" if (terms or sign == "-") else core)
            tt = self.t[row]
            if tt:
                num, den = _reduce_fraction(tt, STBF)
                terms.append(f"{'+' if num > 0 else '-'}{abs(num)}/{den}" if terms else f"{num}/{den}")
            coords.append("".join(terms) if terms else "0")
        return ",".join(coords)


_ORDER_TABLE = {
    (1, 3): 1,    # identity
    (1, -1): 2,   # 2-fold
    (1, 0): 3,    # 3-fold
    (1, 1): 4,    # 4-fold
    (1, 2): 6,    # 6-fold
    (-1, -3): 2,  # inversion
    (-1, 1): 2,   # mirror
    (-1, 0): -6,  # -6
    (-1, -1): -4, # -4
    (-1, -2): -3, # -3
}


def _reduce_fraction(num: int, den: int) -> Tuple[int, int]:
    from math import gcd
    g = gcd(abs(num), den)
    return num // g, den // g


def expand_group(generators: Iterable[SymOp], lattice_translations: Iterable[Tuple[int, int, int]]) -> list[SymOp]:
    """Close the group under composition + lattice centering.

    Returns the full list of unique Seitz operators (rotation + translation mod
    STBF, with all centering vectors included).  Always contains the identity.
    """
    seen = set()
    out: list[SymOp] = []

    def _add(op: SymOp) -> bool:
        key = (op.R, op.t)
        if key in seen:
            return False
        seen.add(key)
        out.append(op)
        return True

    _add(SymOp.identity())
    for g in generators:
        _add(g)

    # Closure
    grew = True
    while grew:
        grew = False
        for a in list(out):
            for b in list(out):
                if _add(a.compose(b)):
                    grew = True
            if len(out) > 500:
                raise RuntimeError("Group did not close — bad Hall symbol?")

    # Apply lattice centering translations
    extra = []
    for op in out:
        for tr in lattice_translations:
            if all(v == 0 for v in tr):
                continue
            t2 = tuple((op.t[i] + tr[i]) % STBF for i in range(3))
            extra.append(SymOp(op.R, t2))
    for op in extra:
        _add(op)

    return out
