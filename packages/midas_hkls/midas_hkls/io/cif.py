"""CIF reader / writer for ``Crystal``.

Backend strategy:
    1. ``gemmi`` (preferred) — full-fat CIF1.1 parser with anisotropic ADPs,
       symop deduction, and SG-from-Hall fallbacks.
    2. ``pycifrw`` fallback — pure-Python, isotropic-ADP only.

Install with::

    pip install "midas-hkls[cif]"        # gemmi
    pip install "midas-hkls[cif-pure]"   # pycifrw fallback
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..crystal import Atom, Crystal, U_to_B
from ..lattice import Lattice
from ..space_group import SpaceGroup


__all__ = ["read_cif", "write_cif", "CifBackendError"]


class CifBackendError(ImportError):
    """Raised when no CIF backend is available."""


def _try_gemmi():
    try:
        import gemmi  # noqa: F401
        return gemmi
    except ImportError:
        return None


def _try_pycifrw():
    try:
        import CifFile  # noqa: F401  (pycifrw exposes itself as CifFile)
        return CifFile
    except ImportError:
        return None


# =============================================================== READ

def read_cif(path: str | Path, *, block: Optional[str] = None) -> Crystal:
    """Read a CIF file into a ``Crystal``.

    ``block`` selects a specific data block by name; default is the first
    block in the file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    gemmi = _try_gemmi()
    if gemmi is not None:
        return _read_via_gemmi(p, block, gemmi)

    cif = _try_pycifrw()
    if cif is not None:
        return _read_via_pycifrw(p, block, cif)

    raise CifBackendError(
        "No CIF backend installed.  Run `pip install midas-hkls[cif]` "
        "(gemmi) or `pip install midas-hkls[cif-pure]` (pycifrw)."
    )


def _read_via_gemmi(path: Path, block: Optional[str], gemmi) -> Crystal:
    ss = gemmi.read_small_structure(str(path))
    if block is not None and ss.name != block:
        # gemmi reads the first block by default; fall back to manual parse for explicit block selection.
        doc = gemmi.cif.read(str(path))
        sub = doc[block] if block in [b.name for b in doc] else None
        if sub is None:
            raise KeyError(f"block {block!r} not found in {path}")
        ss = gemmi.SmallStructure.from_block(sub)

    cell = ss.cell
    lat = Lattice(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)

    sg = _resolve_spacegroup_gemmi(ss, gemmi)

    atoms: list[Atom] = []
    for site in ss.sites:
        sym = site.type_symbol or site.element.name
        u_iso = float(site.u_iso) if site.u_iso else 0.0
        b_iso = U_to_B(u_iso)
        aniso = site.aniso
        u_aniso = None
        if aniso.nonzero():
            u_aniso = (
                float(aniso.u11), float(aniso.u22), float(aniso.u33),
                float(aniso.u12), float(aniso.u13), float(aniso.u23),
            )
        atoms.append(
            Atom(
                element=sym,
                fract=(float(site.fract.x), float(site.fract.y), float(site.fract.z)),
                occupancy=float(site.occ),
                B_iso=b_iso,
                U_aniso=u_aniso,
                label=site.label,
            )
        )
    return Crystal(lattice=lat, space_group=sg, atoms=atoms, name=ss.name or path.stem)


def _resolve_spacegroup_gemmi(ss, gemmi) -> SpaceGroup:
    """Map gemmi's space group to our SpaceGroup via Hall symbol when possible,
    falling back to SG number, then HM symbol.

    Critically, prefer ``ss.spacegroup.hall`` (the *resolved* group, including
    origin-choice and rhombohedral settings) over the raw text fields, which
    are often empty or carry only a ":2" suffix on the HM symbol.
    """
    resolved_hall = ss.spacegroup.hall.strip() if ss.spacegroup is not None else ""
    hall = (ss.spacegroup_hall or "").strip()
    number = int(ss.spacegroup_number or 0)
    hm = (ss.spacegroup_hm or "").strip()

    # 1. Resolved Hall symbol from gemmi's parsed SG object (handles :1/:2/:H/:R).
    for candidate in (resolved_hall, hall):
        if candidate:
            try:
                return SpaceGroup.from_hall(candidate)
            except ValueError:
                pass

    # 2. HM with explicit setting suffix → look up the matching extension.
    if hm:
        m = _HM_EXT_RE.search(hm)
        if m:
            ext = m.group(1)
            try:
                return SpaceGroup.from_number(number, extension=ext)
            except ValueError:
                pass
        try:
            return SpaceGroup.from_hm(hm)
        except ValueError:
            pass

    # 3. Bare SG number → first table entry (origin choice 1 by sginfo convention).
    if number:
        try:
            return SpaceGroup.from_number(number)
        except ValueError:
            pass
    raise ValueError(
        f"could not resolve space group from CIF (hall={hall!r}, number={number}, hm={hm!r})"
    )


import re
_HM_EXT_RE = re.compile(r":\s*([12HR])\s*$")


def _read_via_pycifrw(path: Path, block: Optional[str], CifFile) -> Crystal:
    cf = CifFile.ReadCif(str(path))
    block_name = block or list(cf.keys())[0]
    blk = cf[block_name]

    a = float(_strip_esd(blk["_cell_length_a"]))
    b = float(_strip_esd(blk["_cell_length_b"]))
    c = float(_strip_esd(blk["_cell_length_c"]))
    al = float(_strip_esd(blk["_cell_angle_alpha"]))
    be = float(_strip_esd(blk["_cell_angle_beta"]))
    ga = float(_strip_esd(blk["_cell_angle_gamma"]))
    lat = Lattice(a, b, c, al, be, ga)

    sg = _resolve_spacegroup_pycifrw(blk)

    # pycifrw exposes atom_site columns as parallel lists (case-insensitive keys).
    labels = _list_or_empty(blk, "_atom_site_label")
    if not labels:
        # try fallback: use type_symbol if labels missing
        labels = _list_or_empty(blk, "_atom_site_type_symbol")
    type_syms = _list_or_empty(blk, "_atom_site_type_symbol") or labels
    fxs = _list_or_empty(blk, "_atom_site_fract_x")
    fys = _list_or_empty(blk, "_atom_site_fract_y")
    fzs = _list_or_empty(blk, "_atom_site_fract_z")
    occs = _list_or_empty(blk, "_atom_site_occupancy")
    u_isos = _list_or_empty(blk, "_atom_site_U_iso_or_equiv")
    b_isos = _list_or_empty(blk, "_atom_site_B_iso_or_equiv")

    n = len(fxs)
    atoms: list[Atom] = []
    for i in range(n):
        label = labels[i] if i < len(labels) else ""
        sym = type_syms[i] if i < len(type_syms) else label
        x = float(_strip_esd(fxs[i]))
        y = float(_strip_esd(fys[i]))
        z = float(_strip_esd(fzs[i]))
        occ = float(_strip_esd(occs[i])) if i < len(occs) and occs[i] not in ("", "?") else 1.0
        if i < len(u_isos) and u_isos[i] not in ("", "?", None):
            b_iso = U_to_B(float(_strip_esd(u_isos[i])))
        elif i < len(b_isos) and b_isos[i] not in ("", "?", None):
            b_iso = float(_strip_esd(b_isos[i]))
        else:
            b_iso = 0.0
        atoms.append(
            Atom(element=sym, fract=(x, y, z), occupancy=occ, B_iso=b_iso, label=label)
        )
    return Crystal(lattice=lat, space_group=sg, atoms=atoms, name=block_name)


def _list_or_empty(blk, key: str) -> list[str]:
    """pycifrw is case-insensitive but only returns lists for looped items."""
    if key not in blk:
        return []
    val = blk[key]
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    return [str(val)]


def _strip_esd(v: str) -> str:
    """Drop CIF '(esd)' tail so '5.4112(2)' parses as 5.4112."""
    s = str(v).strip()
    if "(" in s:
        s = s.split("(")[0]
    return s


def _resolve_spacegroup_pycifrw(blk) -> SpaceGroup:
    for key in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
        if key in blk and blk[key] not in ("", "?"):
            try:
                return SpaceGroup.from_number(int(blk[key]))
            except ValueError:
                pass
    for key in ("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M"):
        if key in blk and blk[key] not in ("", "?"):
            try:
                return SpaceGroup.from_hm(str(blk[key]).strip().strip("'\""))
            except ValueError:
                pass
    for key in ("_space_group_name_Hall", "_symmetry_space_group_name_Hall"):
        if key in blk and blk[key] not in ("", "?"):
            try:
                return SpaceGroup.from_hall(str(blk[key]).strip().strip("'\""))
            except ValueError:
                pass
    raise ValueError("CIF has no resolvable space-group identifier")


# =============================================================== WRITE

def write_cif(crystal: Crystal, path: str | Path) -> None:
    """Write a Crystal to a CIF1.1 file.  Uses gemmi when available."""
    p = Path(path)
    gemmi = _try_gemmi()
    if gemmi is not None:
        return _write_via_gemmi(crystal, p, gemmi)
    return _write_minimal_cif(crystal, p)


def _write_via_gemmi(crystal: Crystal, path: Path, gemmi) -> None:
    ss = gemmi.SmallStructure()
    ss.name = crystal.name or path.stem
    L = crystal.lattice
    ss.cell = gemmi.UnitCell(L.a, L.b, L.c, L.alpha, L.beta, L.gamma)
    sg_hm = crystal.space_group.hm_symbol or f"#{crystal.space_group.number}"
    try:
        ss.spacegroup = gemmi.find_spacegroup_by_number(crystal.space_group.number)
    except Exception:
        pass
    ss.spacegroup_hm = sg_hm
    for a in crystal.atoms:
        site = ss.Site()
        site.label = a.label or a.element
        site.type_symbol = a.element
        site.fract = gemmi.Fractional(a.fract[0], a.fract[1], a.fract[2])
        site.occ = float(a.occupancy)
        site.u_iso = float(a.B_iso) / (8.0 * 3.141592653589793 ** 2)
        if a.U_aniso is not None:
            site.aniso = gemmi.SMat33d(
                a.U_aniso[0], a.U_aniso[1], a.U_aniso[2],
                a.U_aniso[3], a.U_aniso[4], a.U_aniso[5],
            )
        ss.add_site(site)
    block = ss.make_cif_block()
    block.write_file(str(path))


def _write_minimal_cif(crystal: Crystal, path: Path) -> None:
    """Pure-Python CIF1.1 emitter — covers lattice + SG + isotropic atom site loop."""
    L = crystal.lattice
    sg = crystal.space_group
    block_name = (crystal.name or path.stem).replace(" ", "_") or "crystal"
    lines = [
        f"data_{block_name}",
        f"_cell_length_a   {L.a:.6f}",
        f"_cell_length_b   {L.b:.6f}",
        f"_cell_length_c   {L.c:.6f}",
        f"_cell_angle_alpha {L.alpha:.6f}",
        f"_cell_angle_beta  {L.beta:.6f}",
        f"_cell_angle_gamma {L.gamma:.6f}",
        f"_space_group_IT_number {sg.number}",
        f"_space_group_name_Hall '{sg.hall_symbol}'",
        f"_space_group_name_H-M_alt '{sg.hm_symbol}'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "_atom_site_B_iso_or_equiv",
    ]
    for i, a in enumerate(crystal.atoms, 1):
        lab = a.label or f"{a.element}{i}"
        lines.append(
            f"{lab} {a.element} {a.fract[0]:.6f} {a.fract[1]:.6f} {a.fract[2]:.6f} "
            f"{a.occupancy:.4f} {a.B_iso:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")
