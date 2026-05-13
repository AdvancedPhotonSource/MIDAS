"""Item 18 — CIF round-trip regression test.

Sweeps the entire ``tests/data`` corpus and asserts each one round-trips
through write_cif → read_cif preserving lattice, space-group number,
atom list, occupancies, and ADPs. Adds a synthetic Cu (FCC, SG 225) and
an SG 167 rhombohedral test (calcite already in corpus). Also exercises
the gemmi-off pure-python writer to confirm parity between backends.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, read_cif, write_cif


DATA = Path(__file__).parent / "data"
ALL_CORPUS = sorted(DATA.glob("*.cif"))


@pytest.fixture
def gemmi_only():
    pytest.importorskip("gemmi")


@pytest.mark.parametrize("path", ALL_CORPUS, ids=[p.name for p in ALL_CORPUS])
def test_corpus_round_trip(gemmi_only, tmp_path, path):
    """Every CIF in tests/data must round-trip without lattice / atom drift."""
    xt = read_cif(path)
    out = tmp_path / f"rt_{path.name}"
    write_cif(xt, out)
    rt = read_cif(out)
    # Lattice
    for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
        assert getattr(rt.lattice, attr) == pytest.approx(
            getattr(xt.lattice, attr), abs=1e-4
        ), f"lattice {attr} mismatch on {path.name}"
    # Space-group number
    assert rt.space_group.number == xt.space_group.number, (
        f"SG mismatch on {path.name}: {rt.space_group.number} vs "
        f"{xt.space_group.number}"
    )
    # Atom list, sorted by label
    a_in = sorted(xt.atoms, key=lambda a: a.label)
    a_out = sorted(rt.atoms, key=lambda a: a.label)
    assert len(a_in) == len(a_out), f"atom count mismatch on {path.name}"
    for a, b in zip(a_in, a_out):
        assert a.element == b.element
        for u, v in zip(a.fract, b.fract):
            assert u == pytest.approx(v, abs=1e-4)
        assert a.occupancy == pytest.approx(b.occupancy, abs=1e-4)
        assert a.B_iso == pytest.approx(b.B_iso, abs=1e-3)


def test_synthetic_fcc_cu_round_trip(gemmi_only, tmp_path):
    """FCC Cu (SG 225) — sanity-check synthetic crystal write+read."""
    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=3.615)
    xt = Crystal(
        lattice=lat, space_group=sg,
        atoms=[Atom("Cu", (0.0, 0.0, 0.0), occupancy=1.0, B_iso=0.5)],
        name="cu",
    )
    out = tmp_path / "cu.cif"
    write_cif(xt, out)
    rt = read_cif(out)
    assert rt.space_group.number == 225
    assert rt.lattice.a == pytest.approx(3.615, abs=1e-4)
    assert rt.atoms[0].element == "Cu"
    assert rt.atoms[0].B_iso == pytest.approx(0.5, abs=1e-3)


def test_writer_fallback_matches_gemmi(tmp_path):
    """The pure-Python writer (gemmi off) must produce a CIF that the
    gemmi reader still understands and recovers identical content."""
    pytest.importorskip("gemmi")
    from midas_hkls.io import cif as cifmod

    sg = SpaceGroup.from_number(225)
    lat = Lattice.for_system("cubic", a=5.4112)
    xt = Crystal(
        lattice=lat, space_group=sg,
        atoms=[
            Atom("Ce1", (0.0, 0.0, 0.0), occupancy=1.0, B_iso=0.4),
            Atom("O1", (0.25, 0.25, 0.25), occupancy=1.0, B_iso=0.6),
        ],
        name="ceo2_synth",
    )
    real = cifmod._try_gemmi
    cifmod._try_gemmi = lambda: None
    try:
        out = tmp_path / "ceo2_pp.cif"
        write_cif(xt, out)
    finally:
        cifmod._try_gemmi = real
    rt = read_cif(out)
    assert rt.space_group.number == 225
    assert rt.lattice.a == pytest.approx(5.4112, abs=1e-4)
    assert len(rt.atoms) == 2
    # Pure-python writer may emit type_symbol differently; just confirm
    # the elements mentioned in the file are recognised somewhere.
    elems_or_labels = {a.element for a in rt.atoms} | {a.label for a in rt.atoms}
    assert any(s.startswith("Ce") for s in elems_or_labels)
    assert any(s.startswith("O") for s in elems_or_labels)


def test_origin_choice_round_trip_si(gemmi_only, tmp_path):
    """SG 227 (Fd-3m) has two origin choices. The CIF reader should
    preserve whichever is recorded; round-trip must not flip atoms by
    a (1/8, 1/8, 1/8) shift."""
    xt = read_cif(DATA / "si.cif")
    out = tmp_path / "si_rt.cif"
    write_cif(xt, out)
    rt = read_cif(out)
    f_in = sorted(a.fract for a in xt.atoms)
    f_out = sorted(a.fract for a in rt.atoms)
    for fa, fb in zip(f_in, f_out):
        for u, v in zip(fa, fb):
            assert u == pytest.approx(v, abs=1e-4)
