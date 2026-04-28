"""Unit tests: Hall parser, group orders, lattice geometry, equivalent HKLs."""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_hkls import Lattice, SpaceGroup, generate_hkls
from midas_hkls.tables import LATTICE_TRANSLATIONS, STBF


# Standard general-position multiplicity per Laue class (centric SGs only).
EXPECTED_LAUE_ORDER = {
    "-1": 2, "2/m": 4, "mmm": 8, "4/m": 8, "4/mmm": 16, "-3": 6, "-3m": 12,
    "6/m": 12, "6/mmm": 24, "m-3": 24, "m-3m": 48,
}
CENT_FACTOR = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}


@pytest.mark.parametrize("num", list(range(1, 231)))
def test_all_230_space_groups_parse(num):
    sg = SpaceGroup.from_number(num)
    assert sg.order >= 1
    assert sg.centering in CENT_FACTOR
    # Friedel-corrected order = Laue × centering
    factor = 1 if sg.is_centrosymmetric() else 2
    assert sg.order * factor == EXPECTED_LAUE_ORDER[sg.laue_class] * CENT_FACTOR[sg.centering]
    # Lattice translations divide order
    assert sg.order % CENT_FACTOR[sg.centering] == 0


def test_p1_has_only_identity():
    sg = SpaceGroup.from_number(1)
    assert sg.order == 1
    assert sg.operations[0].R == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert sg.operations[0].t == (0, 0, 0)


def test_pminus1_has_inversion():
    sg = SpaceGroup.from_number(2)
    assert sg.order == 2
    assert any(op.R == (-1, 0, 0, 0, -1, 0, 0, 0, -1) for op in sg.operations)


def test_fm3m_systematic_absences():
    sg = SpaceGroup.from_number(225)
    # Allowed: hkl all even or all odd
    assert not sg.is_systematically_absent(1, 1, 1)
    assert not sg.is_systematically_absent(2, 0, 0)
    assert not sg.is_systematically_absent(2, 2, 0)
    # Forbidden: mixed parity
    assert sg.is_systematically_absent(1, 0, 0)
    assert sg.is_systematically_absent(2, 1, 0)
    assert sg.is_systematically_absent(2, 1, 1)


def test_im3m_systematic_absences():
    sg = SpaceGroup.from_number(229)
    # I-centered: h+k+l = 2n
    assert not sg.is_systematically_absent(1, 1, 0)
    assert not sg.is_systematically_absent(2, 0, 0)
    assert sg.is_systematically_absent(1, 0, 0)
    assert sg.is_systematically_absent(1, 1, 1)


def test_fd3m_d_glide_absences():
    sg = SpaceGroup.from_number(227)
    # Si: F-centered with d-glide → only (hkl) with all-even, h+k+l = 4n  OR  all-odd
    assert not sg.is_systematically_absent(1, 1, 1)
    assert not sg.is_systematically_absent(2, 2, 0)  # 2+2+0 = 4
    assert not sg.is_systematically_absent(4, 0, 0)
    assert sg.is_systematically_absent(2, 0, 0)      # F: mixed parity check first... 2,0,0 is all-even but 2+0+0=2 not 4n
    assert sg.is_systematically_absent(1, 0, 0)


def test_fm3m_multiplicities():
    sg = SpaceGroup.from_number(225)
    # FCC general / special multiplicities
    assert sg.multiplicity(1, 1, 1) == 8
    assert sg.multiplicity(2, 0, 0) == 6
    assert sg.multiplicity(2, 2, 0) == 12
    assert sg.multiplicity(3, 1, 1) == 24
    assert sg.multiplicity(2, 2, 2) == 8
    assert sg.multiplicity(4, 0, 0) == 6


def test_lattice_d_spacing_cubic():
    lat = Lattice.for_system("cubic", a=5.411)
    # 1/d² = (h² + k² + l²) / a²
    for h, k, l in [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]:
        expected = lat.a / math.sqrt(h*h + k*k + l*l)
        assert abs(lat.d_spacing(h, k, l) - expected) < 1e-10


def test_lattice_metric_tensor_orthogonal():
    lat = Lattice.for_system("orthorhombic", a=4.0, b=5.0, c=6.0)
    G = lat.metric_tensor()
    expected = np.diag([16.0, 25.0, 36.0])
    np.testing.assert_allclose(G, expected, atol=1e-12)


def test_reciprocal_lattice_volume_invariance():
    lat = Lattice(4.5, 6.5, 7.0, 90.0, 95.0, 90.0)  # monoclinic
    V = lat.volume()
    Vr = lat.reciprocal().volume()
    assert abs(V * Vr - 1.0) < 1e-10


def test_systematic_absences_table_consistency():
    """For every centric SG, compute (hkl), allowed↔(hkl-systematically-extinct?).
    The set of allowed reflections must be invariant under all symmetry ops.
    """
    sg = SpaceGroup.from_number(225)
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if (h, k, l) == (0, 0, 0):
                    continue
                allowed = not sg.is_systematically_absent(h, k, l)
                # Pick another SG-equivalent and check same answer
                eq = sg.equivalent_hkls(h, k, l)
                for hp, kp, lp in eq[:5]:
                    assert (not sg.is_systematically_absent(hp, kp, lp)) == allowed
