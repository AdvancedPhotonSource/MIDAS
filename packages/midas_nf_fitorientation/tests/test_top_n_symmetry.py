"""Tests for the symmetry-aware Top-N uniqueness in :class:`TopNTracker`.

The bug we are fixing: when two candidate orientations are related by a
crystal symmetry op (e.g. a 90° z-rotation in a cubic crystal), the C
code's ``GetMisOrientationAngle`` returns ~0° because it reduces both
to the fundamental zone first. Our previous tracker used raw
quaternion misorientation and would accept both as "distinct"
solutions, double-counting the same physical orientation.
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_nf_fitorientation.fit_orientation import TopNTracker


def test_topn_collapses_cubic_symmetric_pairs():
    """Two Eulers related by a 90°-z rotation are symmetry-equivalent
    under SG 225 (FCC, point group m-3m). The tracker should refuse the
    second one even though their raw quaternion miso is 90°.
    """
    tracker = TopNTracker(
        n_saves=5,
        min_miso_deg=1.0,
        space_group=225,            # cubic
    )
    eul1 = torch.tensor([0.0, 0.0, 0.0])
    eul2 = torch.tensor([math.pi / 2, 0.0, 0.0])  # 90° about z

    tracker.offer(eul1, frac=0.95)
    tracker.offer(eul2, frac=0.94)

    assert len(tracker.entries) == 1
    assert tracker.entries[0][3] == pytest.approx(0.95)


def test_topn_keeps_genuinely_distinct_orientations():
    """An orientation more than ``min_miso_deg`` away from any existing
    entry must be kept, even under cubic symmetry."""
    tracker = TopNTracker(
        n_saves=5,
        min_miso_deg=2.0,
        space_group=225,
    )
    eul1 = torch.tensor([0.0, 0.0, 0.0])
    # 30° about Bunge phi1 — not a cubic symmetry, well above 2°.
    eul2 = torch.tensor([math.radians(30.0), 0.0, 0.0])

    tracker.offer(eul1, frac=0.95)
    tracker.offer(eul2, frac=0.93)
    assert len(tracker.entries) == 2
    # Sorted descending by frac
    assert tracker.entries[0][3] == pytest.approx(0.95)
    assert tracker.entries[1][3] == pytest.approx(0.93)


def test_topn_caps_at_n_saves():
    """When more than ``n_saves`` distinct orientations are offered,
    the lowest-frac entry is pushed out."""
    tracker = TopNTracker(
        n_saves=2,
        min_miso_deg=1.0,
        space_group=1,              # triclinic — no symmetry collapse
    )
    # Three Eulers far apart in raw misorientation; SG 1 has no
    # symmetry ops, so all three are "distinct".
    e = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([math.radians(20.0), 0.0, 0.0]),
        torch.tensor([math.radians(40.0), 0.0, 0.0]),
    ]
    tracker.offer(e[0], frac=0.90)
    tracker.offer(e[1], frac=0.95)
    tracker.offer(e[2], frac=0.85)
    assert len(tracker.entries) == 2
    # 0.95 first, then 0.90 (0.85 dropped)
    assert tracker.entries[0][3] == pytest.approx(0.95)
    assert tracker.entries[1][3] == pytest.approx(0.90)
