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


# ---------------------------------------------------------------------------
# centroid method: per-scan median (y, z) drift from spot lists
# ---------------------------------------------------------------------------


def _write_scan_csv(layer_dir, scan_idx, n_spots, y_offset, z_offset):
    """Synthesise an InputAllExtraInfoFittingAll{scan_idx}.csv.

    Spots are placed at ``y = y_offset + idx``, ``z = z_offset + idx``
    so the median lands at ``y = y_offset + (n_spots-1)/2``.
    """
    import numpy as np
    arr = np.zeros((n_spots, 18))
    arr[:, 0] = np.arange(n_spots)                  # ID
    arr[:, 1] = 0.0                                 # omega
    arr[:, 2] = y_offset + np.arange(n_spots)       # y_det
    arr[:, 3] = z_offset + np.arange(n_spots)       # z_det
    arr[:, 4] = 1                                   # ring
    path = layer_dir / f"InputAllExtraInfoFittingAll{scan_idx}.csv"
    np.savetxt(path, arr, header=" ".join(f"col{c}" for c in range(18)),
               comments="")


def test_align_method_centroid_zero_drift(tmp_path):
    """Same spot pattern across scans → zero per-scan drift."""
    for i in range(4):
        _write_scan_csv(tmp_path, i, n_spots=5, y_offset=0.0, z_offset=0.0)
    diags = align_per_scan(
        layer_dir=tmp_path, n_scans=4, method="centroid", reference_scan=2,
    )
    assert len(diags) == 4
    for d in diags:
        assert d.delta_bc_y == 0.0
        assert d.delta_bc_z == 0.0
        assert d.method == "centroid"


def test_align_method_centroid_recovers_known_offset(tmp_path):
    """Inject known per-scan drift, check the centroid method recovers it."""
    # 5 spots per scan, scan i has y_offset=i (z_offset=0). Reference=scan 0.
    # Median(y_offset+0..4) = y_offset + 2. So delta_bc_y for scan i = i.
    for i in range(5):
        _write_scan_csv(tmp_path, i, n_spots=5, y_offset=float(i), z_offset=0.0)
    diags = align_per_scan(
        layer_dir=tmp_path, n_scans=5, method="centroid", reference_scan=0,
    )
    deltas_y = [d.delta_bc_y for d in diags]
    assert deltas_y == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_align_method_centroid_missing_scan_records_nan(tmp_path):
    """When a scan's spot file is missing, residual_px = NaN."""
    import math
    _write_scan_csv(tmp_path, 0, n_spots=3, y_offset=0.0, z_offset=0.0)
    # scan 1's file deliberately missing.
    _write_scan_csv(tmp_path, 2, n_spots=3, y_offset=5.0, z_offset=0.0)
    diags = align_per_scan(
        layer_dir=tmp_path, n_scans=3, method="centroid", reference_scan=0,
    )
    assert math.isnan(diags[1].residual_px)
    assert diags[1].delta_bc_y == 0.0


def test_align_method_centroid_missing_reference_raises(tmp_path):
    """No reference scan file → clear error message."""
    _write_scan_csv(tmp_path, 0, n_spots=3, y_offset=0.0, z_offset=0.0)
    # reference_scan=1 missing
    with pytest.raises(FileNotFoundError, match="reference"):
        align_per_scan(
            layer_dir=tmp_path, n_scans=3, method="centroid", reference_scan=1,
        )
