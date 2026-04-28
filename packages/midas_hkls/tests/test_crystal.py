"""Crystal/Atom symmetry expansion tests."""
from __future__ import annotations

import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, B_to_U, U_to_B


def test_b_u_round_trip():
    for B in (0.0, 0.5, 1.7, 5.0):
        assert U_to_B(B_to_U(B)) == pytest.approx(B)


def test_atom_validation():
    Atom("Fe", (0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        Atom("Fe", (0.1, 0.2))  # wrong length
    with pytest.raises(ValueError):
        Atom("Fe", (0.0, 0.0, 0.0), occupancy=2.0)  # out of range
    with pytest.raises(ValueError):
        Atom("Fe", (0.0, 0.0, 0.0), U_aniso=(0.01, 0.01, 0.01))  # wrong length


def test_ceo2_unit_cell_expansion():
    """CeO₂ Fm-3m: 4 Ce on (0,0,0) class, 8 O on (1/4,1/4,1/4) class."""
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=5.4112)
    ce = Atom("Ce", (0.0, 0.0, 0.0), B_iso=0.4)
    o = Atom("O", (0.25, 0.25, 0.25), B_iso=0.8)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[ce, o])
    uc = xt.unit_cell_atoms()
    n_ce = sum(1 for a in uc if a.element == "Ce")
    n_o = sum(1 for a in uc if a.element == "O")
    assert n_ce == 4
    assert n_o == 8


def test_si_diamond_unit_cell():
    """Si Fd-3m, origin-2: 8 Si in unit cell at (0,0,0) and (1/4,1/4,1/4) class."""
    sg = SpaceGroup.from_number(227)
    lat = Lattice.for_system("cubic", a=5.4309)
    si = Atom("Si", (0.0, 0.0, 0.0), B_iso=0.5)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[si])
    uc = xt.unit_cell_atoms()
    assert len(uc) == 8


def test_alpha_iron_unit_cell():
    """α-Fe Im-3m: 2 Fe per cell at (0,0,0) and (1/2,1/2,1/2)."""
    sg = SpaceGroup.from_number(229)
    lat = Lattice.for_system("cubic", a=2.866)
    fe = Atom("Fe", (0.0, 0.0, 0.0), B_iso=0.35)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[fe])
    uc = xt.unit_cell_atoms()
    assert len(uc) == 2


def test_dedupe_special_position():
    """An atom on a high-symmetry site (e.g. origin in Fm-3m) shouldn't be
    multiplied by the full group order; the dedupe should collapse it."""
    sg = SpaceGroup.from_number(225)  # order = 192 with centering
    lat = Lattice.for_system("cubic", a=5.0)
    atom = Atom("Cu", (0.0, 0.0, 0.0))
    uc = Crystal(lattice=lat, space_group=sg, atoms=[atom]).unit_cell_atoms()
    # Origin in Fm-3m has site symmetry 4/m -3 2/m → 4 copies in the F cell
    assert len(uc) == 4
