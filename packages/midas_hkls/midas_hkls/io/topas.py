"""TOPAS .inp phase-block adapter.

Emits a TOPAS-readable phase block (Rietveld / Le Bail / Pawley) that
can be ``#include``-d into a master ``.inp`` file. Captures lattice
parameters, space-group HM symbol, atom sites with occupancy and Biso,
and (optionally) anomalous f', f'' for resonance-aware refinement.

Used by Item 34 (σ → Rietveld demo) and Item 47 (σ → MAUD via MILK)
for cross-checking against the TOPAS / MAUD ecosystems.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..crystal import Crystal


_TEMPLATES = ("rietveld", "le_bail", "pawley")


def _hm_for_topas(crystal: Crystal) -> str:
    """TOPAS expects standard short HM with underscores stripped."""
    hm = crystal.space_group.hm_symbol or ""
    return hm.replace("_", "")


def write_topas_phase(
    path: str | Path,
    crystal: Crystal,
    *,
    phase_name: str,
    wavelength_A: float,
    two_theta_max_deg: float = 60.0,
    include_anomalous: bool = False,
    template_kind: str = "rietveld",
    fp_table: Optional[dict] = None,
    fpp_table: Optional[dict] = None,
) -> Path:
    """Emit a TOPAS .inp phase snippet.

    Parameters
    ----------
    path :
        Output ``.inp`` (or ``.inc``) file. Caller is expected to
        ``#include`` it from a master TOPAS run file.
    crystal :
        :class:`midas_hkls.Crystal`. Asymmetric unit only — TOPAS
        applies the space group operations.
    phase_name :
        Phase identifier; quoted in the ``str`` ... block.
    wavelength_A :
        Source wavelength (Å). Used to set ``lam`` if your master
        ``.inp`` doesn't supply one.
    two_theta_max_deg :
        Maximum 2θ to refine (informational only — the actual cutoff
        is set by the data block in TOPAS).
    include_anomalous :
        If True, embed ``fp <fp> fpp <fpp>`` per atom from the supplied
        ``fp_table`` / ``fpp_table`` dicts (element symbol → value).
    template_kind :
        ``"rietveld"`` (default) emits a full structural model;
        ``"le_bail"`` emits a Le-Bail-only block (intensity refined
        bin-by-bin, structure not used); ``"pawley"`` emits Pawley
        (intensity per HKL refined freely).
    fp_table, fpp_table :
        Optional dicts ``{element_symbol: value}``. Only consulted when
        ``include_anomalous=True``.
    """
    if template_kind not in _TEMPLATES:
        raise ValueError(
            f"template_kind must be one of {_TEMPLATES}, got {template_kind!r}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sg_hm = _hm_for_topas(crystal)
    lat = crystal.lattice
    with open(path, "w") as f:
        f.write(f"' TOPAS phase block — emitted by midas_hkls.io.topas\n")
        f.write(f"' phase: {phase_name}\n")
        f.write(f"' lambda_A: {wavelength_A:.6f}\n")
        f.write(f"' two_theta_max_deg: {two_theta_max_deg:.4f}\n")
        f.write(f"str\n")
        f.write(f"   phase_name \"{phase_name}\"\n")
        f.write(f"   space_group \"{sg_hm}\"\n")
        f.write(f"   a {lat.a:.6f} b {lat.b:.6f} c {lat.c:.6f}\n")
        f.write(f"   al {lat.alpha:.5f} be {lat.beta:.5f} ga {lat.gamma:.5f}\n")
        if template_kind == "le_bail":
            f.write(f"   le_bail\n")
            return path
        if template_kind == "pawley":
            f.write(f"   pawley\n")
            return path
        # Rietveld: full atom site list
        for atom in crystal.atoms:
            label = atom.label or atom.element
            x, y, z = atom.fract
            occ = atom.occupancy
            beq = atom.B_iso
            line = (
                f"   site {label} num_posns 0 "
                f"x {x:.6f} y {y:.6f} z {z:.6f} "
                f"occ {atom.element} {occ:.4f} beq {beq:.4f}"
            )
            if include_anomalous:
                fp = (fp_table or {}).get(atom.element, 0.0)
                fpp = (fpp_table or {}).get(atom.element, 0.0)
                line += f" fp {fp:.6f} fpp {fpp:.6f}"
            f.write(line + "\n")
    return path


__all__ = ["write_topas_phase"]
