"""Item 13 — TOPAS .inp adapter for midas_hkls."""
from __future__ import annotations

from pathlib import Path

import pytest

from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_hkls.io import write_topas_phase


def _ceo2() -> Crystal:
    return Crystal(
        lattice=Lattice.for_system("cubic", a=5.4112),
        space_group=SpaceGroup.from_number(225),
        atoms=[
            Atom("Ce", (0.0, 0.0, 0.0), occupancy=1.0, B_iso=0.4),
            Atom("O", (0.25, 0.25, 0.25), occupancy=1.0, B_iso=0.6),
        ],
        name="ceo2",
    )


def test_topas_rietveld_emits_lattice_and_sites(tmp_path: Path):
    out = tmp_path / "ceo2.inp"
    write_topas_phase(
        out, _ceo2(),
        phase_name="CeO2", wavelength_A=0.18,
        template_kind="rietveld",
    )
    text = out.read_text()
    assert "str" in text
    assert "phase_name \"CeO2\"" in text
    assert "space_group" in text
    # Lattice constants
    assert "5.411200" in text
    # Both atom sites present
    assert "site Ce num_posns" in text or "site Ce " in text
    assert "occ Ce 1.0000" in text
    assert "occ O 1.0000" in text
    assert "beq 0.4000" in text
    assert "beq 0.6000" in text


def test_topas_le_bail_skips_atoms(tmp_path: Path):
    out = tmp_path / "ceo2_lb.inp"
    write_topas_phase(
        out, _ceo2(),
        phase_name="CeO2-LB", wavelength_A=0.18,
        template_kind="le_bail",
    )
    text = out.read_text()
    assert "le_bail" in text
    assert "site" not in text


def test_topas_pawley(tmp_path: Path):
    out = tmp_path / "ceo2_p.inp"
    write_topas_phase(
        out, _ceo2(),
        phase_name="CeO2-P", wavelength_A=0.18,
        template_kind="pawley",
    )
    text = out.read_text()
    assert "pawley" in text


def test_topas_anomalous_appends_fp_fpp(tmp_path: Path):
    out = tmp_path / "ceo2_anom.inp"
    write_topas_phase(
        out, _ceo2(),
        phase_name="CeO2-anom", wavelength_A=1.7433,  # near Ce L3 edge
        include_anomalous=True,
        fp_table={"Ce": -8.5, "O": 0.05},
        fpp_table={"Ce": 3.4, "O": 0.03},
    )
    text = out.read_text()
    assert "fp -8.500000" in text
    assert "fpp 3.400000" in text


def test_topas_rejects_unknown_template(tmp_path: Path):
    with pytest.raises(ValueError):
        write_topas_phase(
            tmp_path / "bad.inp", _ceo2(),
            phase_name="x", wavelength_A=0.18,
            template_kind="lebail-fancy",
        )
