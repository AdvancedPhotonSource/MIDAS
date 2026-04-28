"""Hall-symbol parser.

Faithful port of sginfo's ParseHallSymbol (Ralf W. Grosse-Kunstleve, 1994-96).
The grammar is laid out in Hall (1981) "Space-group notation with an explicit
origin", Acta Cryst A37, 517-525.

Output: (lattice_centering_letter, generators, origin_shift_in_STBF) where
generators is a list of SymOp.  Lattice centering translations are NOT included
in `generators` — they get folded in by symops.expand_group().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .symops import SymOp
from .tables import (
    HALL_TRANSLATIONS,
    LATTICE_TRANSLATIONS,
    RMX_3_111,
    RMX_3I111,
    STBF,
    TAB_XTAL_ROT_MX,
)


class HallSymbolError(ValueError):
    pass


@dataclass
class _HG:
    improper: int = 0
    rotation: int = 0
    ref_axis: str = ""    # 'x' / 'y' / 'z' / ''
    dir_code: str = ""    # '=' / '"' / "'" / '|' / '\\' / '*' / ''
    screw: int = 0
    R: Optional[Tuple[int, ...]] = None
    t: List[int] = field(default_factory=lambda: [0, 0, 0])

    def reset(self) -> None:
        self.improper = 0
        self.rotation = 1
        self.ref_axis = ""
        self.dir_code = ""
        self.screw = 0
        self.R = None
        self.t = [0, 0, 0]


def _rotate_R(R: Tuple[int, ...], rmx: Tuple[int, ...], inv_rmx: Tuple[int, ...]) -> Tuple[int, ...]:
    """Conjugate R by rmx: R' = rmx * R * inv_rmx (used for axis cycling)."""
    A = np.array(R, dtype=int).reshape(3, 3)
    M = np.array(rmx, dtype=int).reshape(3, 3)
    Mi = np.array(inv_rmx, dtype=int).reshape(3, 3)
    return tuple((M @ A @ Mi).flatten().tolist())


def _lookup_rotation(hg: _HG, hg_index: int) -> bool:
    """Resolve hg.R from rotation order, ref_axis, dir_code (sginfo LookupRotMx)."""
    if hg.rotation <= 0:
        return False

    refaxis = hg.ref_axis
    dircode = hg.dir_code

    if hg.rotation == 1:
        refaxis = "o"
        dircode = "."
        n_next_basis = 0
    elif dircode == "*":
        if not refaxis:
            refaxis = "o"
        n_next_basis = 0
    else:
        if not dircode:
            dircode = "="
        if refaxis == "z":
            n_next_basis = 0
        elif refaxis == "x":
            n_next_basis = 1
        elif refaxis == "y":
            n_next_basis = 2
        else:
            return False

    for order, _eig, dc, rmx in TAB_XTAL_ROT_MX:
        if order != hg.rotation:
            continue
        if dc != dircode:
            continue
        f = 1 if hg.improper == 0 else -1
        R = tuple(v * f for v in rmx)
        for _ in range(n_next_basis):
            R = _rotate_R(R, RMX_3_111, RMX_3I111)
        hg.R = R
        return True
    return False


# Field-type sentinel constants (must be ordered to support sginfo's `<` comparisons)
FT_DELIMITER = 0
FT_IMPROPER = 1
FT_DIGIT = 2
FT_ROTATION = 3
FT_REFAXIS = 4
FT_DIRCODE = 5
FT_TRANSLATION = 6
FT_ORIGINSHIFT = 7


def parse_hall(hsym: str) -> Tuple[str, List[SymOp], Tuple[int, int, int]]:
    """Parse a Hall symbol → (centering, generators, origin_shift).

    `generators` are Seitz ops in the standard setting; `origin_shift` is
    in STBF/12 units (the sginfo convention for `(x y z)` shift values).
    Pure lattice centering translations are NOT in `generators` — apply them
    via `LATTICE_TRANSLATIONS[centering]` after expansion.
    """
    centric = False
    centering: Optional[str] = None
    generators: List[SymOp] = []
    origin_shift_12 = [0, 0, 0]

    hg = _HG(rotation=0, ref_axis="", dir_code="", screw=0)
    n_hg = 0
    clear_hg = True
    field_type = FT_DELIMITER
    previous_rotation = 0
    previous_ref_axis = ""
    sign_origin = 0
    origin_idx = 0
    origin_value = 0

    src = list(hsym) + [" "]  # sentinel, like sginfo's `do { ... } while (*hsym++);`
    i = 0
    while i < len(src):
        ch = src[i]
        # Normalize: '_', '.', '\t' → space (sginfo treats them as whitespace)
        if ch in ("_", ".", "\t", "\0"):
            ch = " "

        if centering is None:
            # We're still parsing the lattice prefix.
            if (not centric) and ch == "-":
                centric = True
            elif ch != " ":
                up = ch.upper()
                if up in ("P", "A", "B", "C", "I", "R", "S", "T", "F"):
                    centering = up
                else:
                    raise HallSymbolError(f"Illegal lattice code at position {i}: {ch!r}")
            i += 1
            continue

        # Inside the generator list.
        if field_type == FT_ORIGINSHIFT:
            # Grammar: '(' integer (whitespace integer){2} ')'
            if ch == ")":
                field_type = FT_DELIMITER
                i += 1
                continue
            if ch == " " or ch == ",":
                if sign_origin != 0 or origin_value != 0:
                    if origin_idx > 2:
                        raise HallSymbolError("Too many origin-shift values")
                    origin_shift_12[origin_idx] = (
                        (origin_value if sign_origin >= 0 else -origin_value) % 12
                    )
                    origin_idx += 1
                    sign_origin = 0
                    origin_value = 0
                i += 1
                continue
            if ch == "-":
                sign_origin = -1
                i += 1
                continue
            if ch == "+":
                sign_origin = 1
                i += 1
                continue
            if ch.isdigit():
                origin_value = origin_value * 10 + int(ch)
                if sign_origin == 0:
                    sign_origin = 1
                i += 1
                continue
            raise HallSymbolError(f"Illegal char in origin shift: {ch!r}")

        c = ch.lower()
        if c == "q":
            c = "'"
        elif c == "+":
            c = '"'

        previous_ft = field_type
        digit = rotation = 0
        ref_axis = ""
        dir_code = ""
        translation = None

        if c in HALL_TRANSLATIONS:
            translation = c
            field_type = FT_TRANSLATION
        else:
            if c == " ":
                field_type = FT_DELIMITER
            elif c == "-":
                field_type = FT_IMPROPER
            elif c.isdigit():
                digit = int(c)
                field_type = FT_DIGIT
            elif c in ("x", "y", "z"):
                ref_axis = c
                field_type = FT_REFAXIS
            elif c in ('"', "'", "*"):
                dir_code = c
                field_type = FT_DIRCODE
            elif c == "(":
                field_type = FT_ORIGINSHIFT
            else:
                raise HallSymbolError(f"Illegal character in Hall symbol at {i}: {ch!r}")

            if field_type == FT_DIGIT:
                if (not clear_hg) and hg.rotation > digit and hg.screw == 0 and hg.dir_code == "":
                    hg.screw = digit
                    field_type = FT_TRANSLATION
                elif digit == 5:
                    raise HallSymbolError("Illegal 5-fold rotation")
                else:
                    rotation = digit
                    field_type = FT_ROTATION

        # End-of-generator detection: per sginfo, emit when a delimiter, origin
        # shift, or "going backwards" in the field-type ordering arrives.
        if (
            (not clear_hg)
            and (
                field_type == FT_DELIMITER
                or field_type == FT_ORIGINSHIFT
                or field_type < previous_ft
                or (field_type == previous_ft and field_type != FT_TRANSLATION)
            )
            and not (field_type == FT_REFAXIS and hg.ref_axis == "" and previous_ft == FT_DIRCODE)
        ):
            # Resolve missing reference axis from positional defaults.
            if hg.ref_axis == "":
                if n_hg == 0:
                    hg.ref_axis = "z"
                else:
                    if hg.rotation == 2:
                        if previous_rotation in (2, 4):
                            hg.ref_axis = "x"
                        elif previous_rotation in (3, 6):
                            hg.ref_axis = previous_ref_axis
                            if hg.dir_code == "":
                                hg.dir_code = "'"
                    elif hg.rotation == 3:
                        if hg.dir_code == "":
                            hg.dir_code = "*"

            previous_ref_axis = hg.ref_axis
            previous_rotation = hg.rotation

            if not _lookup_rotation(hg, n_hg):
                raise HallSymbolError(
                    f"Illegal generator (rotation={hg.rotation}, axis={hg.ref_axis!r}, "
                    f"dir={hg.dir_code!r}) at position {i}"
                )

            # Apply screw translation along the principal axis.
            if hg.screw:
                axis_idx = "xyz".find(hg.ref_axis)
                if hg.dir_code or axis_idx < 0:
                    raise HallSymbolError("Screw on non-principal direction")
                hg.t[axis_idx] = (hg.t[axis_idx] + STBF * hg.screw // hg.rotation) % STBF

            # Reduce translations modulo STBF.
            hg.t = [v % STBF for v in hg.t]

            generators.append(SymOp(hg.R, tuple(hg.t)))
            n_hg += 1
            clear_hg = True

        if field_type not in (FT_DELIMITER, FT_ORIGINSHIFT):
            if clear_hg:
                hg = _HG(rotation=1)
                clear_hg = False

            if field_type == FT_IMPROPER:
                hg.improper = 1
            elif field_type == FT_ROTATION:
                hg.rotation = rotation
            elif field_type == FT_REFAXIS:
                hg.ref_axis = ref_axis
            elif field_type == FT_DIRCODE:
                hg.dir_code = dir_code
            elif field_type == FT_TRANSLATION and translation is not None:
                # Pure Hall-translation char (a/b/c/n/d/u/v/w).  When `translation`
                # is None the FT_TRANSLATION came from screw-digit reclassification
                # and is consumed later by the screw-along-axis logic.
                tx, ty, tz = HALL_TRANSLATIONS[translation]
                hg.t[0] = (hg.t[0] + tx) % STBF
                hg.t[1] = (hg.t[1] + ty) % STBF
                hg.t[2] = (hg.t[2] + tz) % STBF

        i += 1

    if centering is None:
        raise HallSymbolError("No lattice code in Hall symbol")

    # If the Hall symbol began with '-', append inversion as a generator.
    if centric:
        generators.insert(0, SymOp.inversion())

    return centering, generators, tuple(origin_shift_12)  # type: ignore[return-value]
