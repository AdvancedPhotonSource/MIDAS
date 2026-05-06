"""Mode-dispatcher tests."""

from __future__ import annotations

import pytest

from midas_process_grains.modes import (
    VALID_MODES,
    apply_mode_defaults,
    misori_tol_rad,
)
from midas_process_grains.params import ProcessGrainsParams


def test_legacy_mode_uses_04_degrees_and_kenesei():
    p = ProcessGrainsParams()
    p = apply_mode_defaults(p, "legacy")
    assert p.MisoriTol == 0.4
    assert p.StrainMethod == "kenesei"


def test_paper_claim_mode_uses_001_degrees_and_jaccard_09():
    p = ProcessGrainsParams()
    p = apply_mode_defaults(p, "paper_claim")
    assert p.MisoriTol == 0.01
    assert p.JaccardTol == 0.9
    assert p.StrainMethod == "fable_beaudoin"


def test_spot_aware_mode_default_is_05_and_kenesei():
    """Default spot_aware uses MisoriTol=0.5° (gathers C-cluster equivalents
    correctly; 0.25° was too tight on real data and split real grains)."""
    p = ProcessGrainsParams()
    p = apply_mode_defaults(p, "spot_aware")
    assert p.MisoriTol == 0.5
    assert p.StrainMethod == "kenesei"


def test_spot_aware_mode_with_explicit_fable_keeps_fable():
    p = ProcessGrainsParams(StrainMethod="fable_beaudoin")
    p = apply_mode_defaults(p, "spot_aware")
    assert p.StrainMethod == "fable_beaudoin"


def test_alias_lstsq_resolves_to_kenesei():
    p = ProcessGrainsParams(StrainMethod="lstsq").validated()
    assert p.StrainMethod == "kenesei"


def test_alias_lattice_resolves_to_fable_beaudoin():
    p = ProcessGrainsParams(StrainMethod="lattice").validated()
    assert p.StrainMethod == "fable_beaudoin"


def test_strain_method_both_is_valid():
    p = ProcessGrainsParams(StrainMethod="both").validated()
    assert p.StrainMethod == "both"


def test_user_explicit_misori_tol_wins_over_mode_default():
    p = ProcessGrainsParams(MisoriTol=0.10)
    p = apply_mode_defaults(p, "spot_aware")
    assert p.MisoriTol == 0.10


def test_invalid_mode_raises():
    p = ProcessGrainsParams()
    with pytest.raises(ValueError, match="mode must be"):
        apply_mode_defaults(p, "freestyle")


def test_misori_tol_rad_converts_correctly():
    p = ProcessGrainsParams()
    p = apply_mode_defaults(p, "spot_aware")
    import math
    assert abs(misori_tol_rad(p) - math.radians(0.5)) < 1e-15


def test_misori_tol_rad_unresolved_raises():
    p = ProcessGrainsParams()
    with pytest.raises(ValueError, match="MisoriTol unresolved"):
        misori_tol_rad(p)
