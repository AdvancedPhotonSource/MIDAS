"""Static tables ported verbatim from sginfo (Ralf W. Grosse-Kunstleve, 1994-96).

We use STBF=12 as the translation base so that all lattice / screw / glide
translations are exact integers. 1/2 -> 6, 1/3 -> 4, 1/4 -> 3, 1/6 -> 2, 1/12 -> 1.
"""
from __future__ import annotations

from typing import Tuple

STBF = 12  # translation base factor; matches sginfo's STBF


def _t(i: int) -> int:
    """sginfo macro T(i): i * (STBF / 12) — but STBF=12, so just i."""
    return i


# -------------------------------------------------------------------- lattices
# Centering operators in fractional coords (numerator over STBF).
# Each lattice contributes a number of translations (including the trivial 0).
LATTICE_TRANSLATIONS: dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "P": ((0, 0, 0),),
    "A": ((0, 0, 0), (0, 6, 6)),
    "B": ((0, 0, 0), (6, 0, 6)),
    "C": ((0, 0, 0), (6, 6, 0)),
    "I": ((0, 0, 0), (6, 6, 6)),
    "R": ((0, 0, 0), (8, 4, 4), (4, 8, 8)),  # obverse hexagonal R
    "S": ((0, 0, 0), (4, 4, 8), (8, 8, 4)),  # alternate
    "T": ((0, 0, 0), (4, 8, 4), (8, 4, 8)),  # alternate
    "F": ((0, 0, 0), (0, 6, 6), (6, 0, 6), (6, 6, 0)),
}

# -------------------------------------------------------- rotation matrix table
# Per-axis rotation operators used by Hall-symbol generators.  Format:
#   (Order, eigenvector, dir_code, 3x3-row-major)
# DirCode: '=' (principal), '"' (face-diagonal +), "'" (face-diagonal -),
#          '|' (hex 2 1 0), '\\' (hex 1 2 0), '*' (body-diagonal 1 1 1).
TAB_XTAL_ROT_MX: list[tuple[int, tuple[int, int, int], str, tuple[int, ...]]] = [
    (1, (0, 0, 0), ".",   (1, 0, 0,  0, 1, 0,  0, 0, 1)),
    (2, (0, 0, 1), "=",   (-1, 0, 0,  0, -1, 0,  0, 0, 1)),
    (2, (1, 0, 0), "=",   (1, -1, 0,  0, -1, 0,  0, 0, -1)),  # hex
    (2, (0, 1, 0), "=",   (-1, 0, 0, -1,  1, 0,  0, 0, -1)),  # hex
    (2, (1, 1, 0), '"',   (0, 1, 0,  1, 0, 0,  0, 0, -1)),
    (2, (1, -1, 0), "'",  (0, -1, 0, -1, 0, 0,  0, 0, -1)),
    (2, (2, 1, 0), "|",   (1, 0, 0,  1, -1, 0,  0, 0, -1)),  # hex
    (2, (1, 2, 0), "\\",  (-1, 1, 0,  0, 1, 0,  0, 0, -1)),  # hex
    (3, (0, 0, 1), "=",   (0, -1, 0,  1, -1, 0,  0, 0, 1)),
    (3, (1, 1, 1), "*",   (0, 0, 1,  1, 0, 0,  0, 1, 0)),
    (4, (0, 0, 1), "=",   (0, -1, 0,  1, 0, 0,  0, 0, 1)),
    (6, (0, 0, 1), "=",   (1, -1, 0,  1, 0, 0,  0, 0, 1)),
]

# 3-fold around (1,1,1) and its inverse (used for axis cycling).
RMX_3_111 = (0, 0, 1,  1, 0, 0,  0, 1, 0)
RMX_3I111 = (0, 1, 0,  0, 0, 1,  1, 0, 0)


# ----------------------------------------------------- Hall translation symbols
# Single-letter glide / centering translations: a, b, c, n, d, u, v, w.
HALL_TRANSLATIONS: dict[str, Tuple[int, int, int]] = {
    "a": (6, 0, 0),
    "b": (0, 6, 0),
    "c": (0, 0, 6),
    "n": (6, 6, 6),
    "d": (3, 3, 3),
    "u": (3, 0, 0),
    "v": (0, 3, 0),
    "w": (0, 0, 3),
}


# ---------------------------------------------------- crystal system / lattice
def crystal_system_for(num: int) -> str:
    if 1 <= num <= 2:    return "triclinic"
    if 3 <= num <= 15:   return "monoclinic"
    if 16 <= num <= 74:  return "orthorhombic"
    if 75 <= num <= 142: return "tetragonal"
    if 143 <= num <= 167: return "trigonal"
    if 168 <= num <= 194: return "hexagonal"
    if 195 <= num <= 230: return "cubic"
    raise ValueError(f"space group number out of range: {num}")


# Centrosymmetric Laue class for each space group (the Friedel-related point group).
def laue_class_for(num: int) -> str:
    if num <= 2:    return "-1"
    if num <= 15:   return "2/m"
    if num <= 74:   return "mmm"
    if num <= 88:   return "4/m"
    if num <= 142:  return "4/mmm"
    if num <= 148:  return "-3"
    if num <= 167:  return "-3m"
    if num <= 176:  return "6/m"
    if num <= 194:  return "6/mmm"
    if num <= 206:  return "m-3"
    return "m-3m"
