"""Tests for the crystal-system / SG / lookup-type mapping."""

from __future__ import annotations

import pytest

from midas_nf_preprocess.seed_orientations import (
    LOOKUP_TYPES,
    REPRESENTATIVE_SG,
    crystal_system_to_space_group,
    space_group_to_lookup_type,
    space_group_to_sym_order,
)


# ----- space_group_to_lookup_type ---------------------------------------------


@pytest.mark.parametrize(
    "sg,expected",
    [
        # Triclinic
        (1, "triclinic"),
        (2, "triclinic"),
        # Monoclinic
        (3, "monoclinic"),
        (15, "monoclinic"),
        # Orthorhombic
        (16, "orthorhombic"),
        (74, "orthorhombic"),
        # Tetragonal low/high
        (75, "tetragonal_low"),
        (88, "tetragonal_low"),
        (89, "tetragonal_high"),
        (142, "tetragonal_high"),
        # Trigonal subbuckets
        (143, "trigonal_low"),
        (148, "trigonal_low"),
        (155, "trigonal_type1"),
        (149, "trigonal_type2"),
        (162, "trigonal_type2"),
        # Hexagonal low/high
        (168, "hexagonal_low"),
        (176, "hexagonal_low"),
        (177, "hexagonal_high"),
        (194, "hexagonal_high"),
        # Cubic low/high
        (195, "cubic_low"),
        (206, "cubic_low"),
        (207, "cubic_high"),
        (225, "cubic_high"),
        (230, "cubic_high"),
    ],
)
def test_space_group_to_lookup_type(sg, expected):
    assert space_group_to_lookup_type(sg) == expected


def test_space_group_out_of_range():
    with pytest.raises(ValueError, match="\\[1, 230\\]"):
        space_group_to_lookup_type(0)
    with pytest.raises(ValueError, match="\\[1, 230\\]"):
        space_group_to_lookup_type(231)


# ----- crystal_system_to_space_group -----------------------------------------


@pytest.mark.parametrize(
    "name,sg",
    [
        ("triclinic", 1),
        ("monoclinic", 15),
        ("orthorhombic", 74),
        ("tetragonal", 142),
        ("trigonal", 155),
        ("hexagonal", 194),
        ("cubic", 225),
        ("CUBIC", 225),  # case-insensitive
    ],
)
def test_crystal_system_to_space_group(name, sg):
    assert crystal_system_to_space_group(name) == sg


def test_crystal_system_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        crystal_system_to_space_group("foo")


# ----- consistency / coverage -------------------------------------------------


def test_lookup_types_full_coverage():
    """Every lookup type must have a representative SG and sym order."""
    for lt in LOOKUP_TYPES:
        assert lt in REPRESENTATIVE_SG
        assert REPRESENTATIVE_SG[lt] >= 1
        assert REPRESENTATIVE_SG[lt] <= 230


@pytest.mark.parametrize("lt", LOOKUP_TYPES)
def test_representative_round_trip(lt):
    """Round-trip: lookup_type -> SG -> lookup_type."""
    sg = REPRESENTATIVE_SG[lt]
    assert space_group_to_lookup_type(sg) == lt


@pytest.mark.parametrize(
    "sg,expected_order",
    [
        (1, 2),     # triclinic
        (15, 4),    # monoclinic
        (74, 8),    # orthorhombic
        (142, 16),  # tetragonal_high
        (194, 24),  # hexagonal_high
        (225, 48),  # cubic_high
    ],
)
def test_space_group_to_sym_order(sg, expected_order):
    assert space_group_to_sym_order(sg) == expected_order
