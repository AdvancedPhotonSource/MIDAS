"""Shared fixtures for find_grains unit tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0xF1_AD)


def synth_orient_mat(rng: np.random.Generator) -> np.ndarray:
    """Pick a random rotation matrix as a row-major 9-vector."""
    # Random quaternion → rotation matrix.
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R.ravel()


def rotation_about_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Build a 9-vector OM that rotates by ``angle_deg`` about ``axis``."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    theta = np.deg2rad(angle_deg)
    c = np.cos(theta)
    s = np.sin(theta)
    ux, uy, uz = a
    R = np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
    ], dtype=np.float64)
    return R.ravel()


@pytest.fixture
def axis_angle_om():
    return rotation_about_axis
