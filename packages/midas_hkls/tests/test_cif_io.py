"""CIF reader / writer round-trip tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, read_cif, write_cif


DATA = Path(__file__).parent / "data"


@pytest.fixture
def gemmi_only():
    pytest.importorskip("gemmi")


def test_read_ceo2(gemmi_only):
    xt = read_cif(DATA / "ceo2.cif")
    assert xt.space_group.number == 225
    assert xt.lattice.a == pytest.approx(5.4112)
    assert xt.lattice.alpha == pytest.approx(90.0)
    elements = sorted(a.element for a in xt.atoms)
    assert elements == ["Ce", "O"]


def test_read_si(gemmi_only):
    xt = read_cif(DATA / "si.cif")
    assert xt.space_group.number == 227
    assert len(xt.atoms) == 1
    assert xt.atoms[0].element == "Si"


def test_read_lab6(gemmi_only):
    xt = read_cif(DATA / "lab6.cif")
    assert xt.space_group.number == 221
    assert {a.element for a in xt.atoms} == {"La", "B"}


def test_read_alpha_fe(gemmi_only):
    xt = read_cif(DATA / "alpha_fe.cif")
    assert xt.space_group.number == 229
    uc = xt.unit_cell_atoms()
    assert len(uc) == 2  # bcc — 2 Fe per cell


def test_read_calcite(gemmi_only):
    xt = read_cif(DATA / "calcite.cif")
    assert xt.space_group.number == 167
    elems = {a.element for a in xt.atoms}
    assert elems == {"Ca", "C", "O"}
    assert xt.lattice.gamma == pytest.approx(120.0)


def test_round_trip_ceo2(gemmi_only, tmp_path):
    xt = read_cif(DATA / "ceo2.cif")
    out = tmp_path / "ceo2_out.cif"
    write_cif(xt, out)
    rt = read_cif(out)
    assert rt.space_group.number == xt.space_group.number
    assert rt.lattice.a == pytest.approx(xt.lattice.a, abs=1e-4)
    assert len(rt.atoms) == len(xt.atoms)
    for a, b in zip(sorted(xt.atoms, key=lambda a: a.label), sorted(rt.atoms, key=lambda a: a.label)):
        assert a.element == b.element
        for u, v in zip(a.fract, b.fract):
            assert u == pytest.approx(v, abs=1e-4)
        assert a.occupancy == pytest.approx(b.occupancy, abs=1e-4)
        assert a.B_iso == pytest.approx(b.B_iso, abs=1e-3)


def test_round_trip_calcite(gemmi_only, tmp_path):
    xt = read_cif(DATA / "calcite.cif")
    out = tmp_path / "calcite_out.cif"
    write_cif(xt, out)
    rt = read_cif(out)
    assert rt.space_group.number == 167
    assert rt.lattice.gamma == pytest.approx(120.0, abs=1e-4)
    assert len(rt.atoms) == 3


def test_pycifrw_reader_fallback(tmp_path):
    """Force gemmi off and read via pycifrw (when installed)."""
    pytest.importorskip("CifFile")
    from midas_hkls.io import cif as cifmod
    real = cifmod._try_gemmi
    cifmod._try_gemmi = lambda: None
    try:
        xt = cifmod.read_cif(DATA / "ceo2.cif")
        assert xt.space_group.number == 225
        assert xt.lattice.a == pytest.approx(5.4112)
        assert {a.element for a in xt.atoms} == {"Ce", "O"}
    finally:
        cifmod._try_gemmi = real


def test_pure_python_writer_fallback(tmp_path):
    """Force the gemmi-less write path by monkeypatching the importer."""
    from midas_hkls.io import cif as cifmod

    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=5.0)
    xt = Crystal(lattice=lat, space_group=sg, atoms=[Atom("Cu", (0.0, 0.0, 0.0))], name="test_cu")

    real = cifmod._try_gemmi
    cifmod._try_gemmi = lambda: None
    try:
        out = tmp_path / "cu.cif"
        write_cif(xt, out)
        text = out.read_text()
        assert "data_test_cu" in text
        assert "_cell_length_a" in text
        assert "Cu" in text
        assert "_atom_site_fract_x" in text
    finally:
        cifmod._try_gemmi = real
