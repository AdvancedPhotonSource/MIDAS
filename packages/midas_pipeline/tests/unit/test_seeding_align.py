"""Tests for seeding/align.py (Stage A of merged-FF seeding).

The full ring-center alignment requires midas-calibrate-v2 + detector
images; it's scaffolded but not wired up in v1. The ``method='none'``
path is the production fallback for synthetic / drift-free data and
must work without external deps. We test that path plus the
not-implemented guards for the real algorithms.
"""

from __future__ import annotations

import pytest

from midas_pipeline.seeding.align import (
    AlignmentDiagnostics,
    align_per_scan,
)


def test_align_method_none_returns_zero_corrections(tmp_path):
    diags = align_per_scan(layer_dir=tmp_path, n_scans=4, method="none")
    assert len(diags) == 4
    for i, d in enumerate(diags):
        assert isinstance(d, AlignmentDiagnostics)
        assert d.scan_idx == i
        assert d.delta_bc_y == 0.0
        assert d.delta_bc_z == 0.0
        assert d.residual_px == 0.0
        assert d.method == "none"


def test_align_method_ring_center_raises_until_wired(tmp_path):
    with pytest.raises(NotImplementedError, match="ring-center"):
        align_per_scan(layer_dir=tmp_path, n_scans=4, method="ring-center")


def test_align_method_cross_correlation_raises_until_wired(tmp_path):
    with pytest.raises(NotImplementedError, match="cross-correlation"):
        align_per_scan(
            layer_dir=tmp_path, n_scans=4,
            method="cross-correlation",
        )


def test_align_reference_scan_sentinel_resolves_to_middle(tmp_path):
    """method='none' should still work with reference_scan=-1 (the sentinel)."""
    diags = align_per_scan(
        layer_dir=tmp_path, n_scans=6, method="none", reference_scan=-1,
    )
    # No assertion about reference_scan value internally, but the call
    # must not raise on the sentinel.
    assert len(diags) == 6
