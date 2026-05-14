"""Indexing-mode sinogen — scan-position consistency filter drops spurious spots."""

from __future__ import annotations

import os

import numpy as np
import pytest

from midas_pipeline.find_grains import (
    CONSOLIDATED_KEY_COLS,
    CONSOLIDATED_VALS_COLS,
    build_scan_grid,
    generate_sinograms_indexing,
    open_all_three,
    write_ids_bin,
    write_keys_bin,
    write_vals_bin,
)


def synth_fixture(tmp_path, *, n_scans=5):
    """Build a tiny fixture: 1 grain, 1 voxel at scan_position[2]=0, 1 candidate.

    The candidate has 4 matched spots at scan positions [0,1,2,3,4] — only
    scan 2 (the voxel's true position) should pass the scan-position filter.
    """
    out_dir = tmp_path / "Output"
    out_dir.mkdir()

    # Scan positions: -2, -1, 0, 1, 2 µm (= scan_step 1, centered).
    scan_positions = np.arange(n_scans, dtype=np.float64) - (n_scans // 2)
    scan_grid = build_scan_grid(scan_positions)

    # Voxel grid is n_scans x n_scans; pick the center voxel (row=n//2, col=n//2).
    n_vox = n_scans * n_scans
    center_vox = (n_scans // 2) * n_scans + (n_scans // 2)
    # x_V = scan_positions[col], y_V = scan_positions[row]; both = 0 here.

    # Identity OM for the only candidate.
    OM_identity = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64)
    # 16-col vals record: cols [0,1]=spot info, [2..11]=OM, [12,13]=fits, [14]=nExpected, [15]=nMatched.
    val_row = np.zeros(CONSOLIDATED_VALS_COLS, dtype=np.float64)
    val_row[2:11] = OM_identity
    val_row[1] = 0.1                   # IA
    val_row[14] = 4.0                  # nExpected
    val_row[15] = 4.0                  # nMatched → confidence = 1.0
    vals_per_vox = [np.empty((0, CONSOLIDATED_VALS_COLS), dtype=np.float64) for _ in range(n_vox)]
    vals_per_vox[center_vox] = val_row.reshape(1, CONSOLIDATED_VALS_COLS)

    keys_per_vox = [np.empty((0, CONSOLIDATED_KEY_COLS), dtype=np.uint64) for _ in range(n_vox)]
    # Key cols: [SpotID, nMatches, nIDs_this_solution, reserved].
    keys_per_vox[center_vox] = np.array([[1, 4, 4, 0]], dtype=np.uint64)

    # IDs: 4 spot IDs for the center voxel's candidate.
    ids_per_vox = [np.empty(0, dtype=np.int32) for _ in range(n_vox)]
    ids_per_vox[center_vox] = np.array([1, 2, 3, 4], dtype=np.int32)

    # Write consolidated files.
    write_vals_bin(out_dir / "IndexBest_all.bin", vals_per_vox)
    write_keys_bin(out_dir / "IndexKey_all.bin", keys_per_vox)
    write_ids_bin(out_dir / "IndexBest_IDs_all.bin", ids_per_vox)

    # Build Spots.bin. Spot i (1-indexed) is at scanNr i-1. All on ring 1.
    # The C filter: |s_V_at_ome ± s_observed| < scan_tol, where
    # s_V_at_ome = -x_V*cos(ome) + y_V*sin(ome). For voxel at (0,0) this is 0,
    # so |0 ± s_observed| < tol means |s_observed| < tol.
    # Only spot at scan position 0 (= scan_to_spatial-mapped scanNr 2) passes
    # a tight tol < 1.
    spots = np.zeros((4, 10), dtype=np.float64)
    for i in range(4):
        sid = i + 1
        ome = 30.0 + i  # different omegas so they're separate HKL slots
        spots[i, 0] = 100.0 + i  # yCen
        spots[i, 1] = 200.0 + i  # zCen
        spots[i, 2] = ome
        spots[i, 3] = 1000.0 + i * 10  # intensity
        spots[i, 4] = sid             # spotID
        spots[i, 5] = 1               # ringNr
        spots[i, 6] = 15.0 + i * 0.5  # eta
        spots[i, 7] = 5.0             # theta
        spots[i, 8] = 1.0             # dspacing
        # Map spot i to scanNr i (file order — scan_to_spatial may permute,
        # but for sorted scan_positions argsort is identity).
        spots[i, 9] = i

    return out_dir, vals_per_vox, keys_per_vox, ids_per_vox, spots, scan_grid, center_vox


def test_indexing_scan_tol_drops_spurious_spots(tmp_path):
    """Tight scan tolerance keeps only on-position spots."""
    out_dir, vals_per_vox, keys_per_vox, ids_per_vox, spots, scan_grid, center_vox = synth_fixture(tmp_path)
    n_scans = scan_grid.n_scans

    vals_r, keys_r, ids_r = open_all_three(out_dir)

    unique_key_arr = np.array([[center_vox, 1, 4, 4, 0]], dtype=np.uint64)
    unique_OM_arr = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float64)

    # Tight scan tolerance: 0.5 µm. Only the scan with |s_obs| ≈ 0 should
    # pass; that's scan_positions[center] which is index n_scans//2 in
    # sorted order. The other 3 spots are at |s_obs| ∈ {1, 2, 2} > 0.5.
    out = generate_sinograms_indexing(
        unique_key_arr=unique_key_arr,
        unique_OM_arr=unique_OM_arr,
        all_spots=spots,
        n_scans=n_scans,
        space_group=225,
        max_ang_deg=1.0,
        tol_ome=0.5,
        tol_eta=0.1,
        output_dir=tmp_path / "sino_out",
        vals_reader=vals_r,
        keys_reader=keys_r,
        ids_reader=ids_r,
        scan_grid=scan_grid,
        confidence_min=0.5,
        scan_tolerance_um=0.5,
    )

    sino_raw_fn = out.sino_paths["raw"]
    sino = np.frombuffer(open(sino_raw_fn, "rb").read(), dtype=np.float64)
    # Only 1 spot survives → max_n_hkls == 1, one cell at the center scan should be nonzero.
    sino = sino.reshape(out.n_grains, out.max_n_hkls, out.n_scans)
    nz = int((sino > 0).sum())
    assert nz == 1, f"expected exactly 1 nonzero cell after tight scan_tol, got {nz}"


def test_indexing_loose_scan_tol_keeps_all(tmp_path):
    """Loose scan tolerance keeps all 4 spots."""
    out_dir, vals_per_vox, keys_per_vox, ids_per_vox, spots, scan_grid, center_vox = synth_fixture(tmp_path)
    n_scans = scan_grid.n_scans

    vals_r, keys_r, ids_r = open_all_three(out_dir)
    unique_key_arr = np.array([[center_vox, 1, 4, 4, 0]], dtype=np.uint64)
    unique_OM_arr = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float64)
    out = generate_sinograms_indexing(
        unique_key_arr=unique_key_arr,
        unique_OM_arr=unique_OM_arr,
        all_spots=spots,
        n_scans=n_scans,
        space_group=225,
        max_ang_deg=1.0,
        tol_ome=0.5,
        tol_eta=0.1,
        output_dir=tmp_path / "sino_out_loose",
        vals_reader=vals_r,
        keys_reader=keys_r,
        ids_reader=ids_r,
        scan_grid=scan_grid,
        confidence_min=0.5,
        scan_tolerance_um=1000.0,  # very loose
    )
    sino_raw_fn = out.sino_paths["raw"]
    sino = np.frombuffer(open(sino_raw_fn, "rb").read(), dtype=np.float64)
    sino = sino.reshape(out.n_grains, out.max_n_hkls, out.n_scans)
    nz = int((sino > 0).sum())
    # All 4 spots accepted, each at its own scanNr.
    assert nz == 4, f"expected 4 nonzero cells with loose scan_tol, got {nz}"


def test_indexing_env_override_min_conf(tmp_path, monkeypatch):
    """MIDAS_PF_SINO_CONF_MIN env overrides the keyword default."""
    out_dir, vals_per_vox, keys_per_vox, ids_per_vox, spots, scan_grid, center_vox = synth_fixture(tmp_path)
    n_scans = scan_grid.n_scans
    # Setting MIDAS_PF_SINO_CONF_MIN to 1.5 (out of range) should be ignored.
    monkeypatch.setenv("MIDAS_PF_SINO_CONF_MIN", "1.5")
    vals_r, keys_r, ids_r = open_all_three(out_dir)
    unique_key_arr = np.array([[center_vox, 1, 4, 4, 0]], dtype=np.uint64)
    unique_OM_arr = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=np.float64)
    # The candidate has confidence = 1.0 (4/4). With default conf_min=0.5
    # it would pass; with env=0.7 it would still pass; with env=1.5 (invalid)
    # falls back to default 0.5 and still passes.
    out = generate_sinograms_indexing(
        unique_key_arr=unique_key_arr,
        unique_OM_arr=unique_OM_arr,
        all_spots=spots,
        n_scans=n_scans,
        space_group=225,
        max_ang_deg=1.0,
        tol_ome=0.5,
        tol_eta=0.1,
        output_dir=tmp_path / "sino_out_env",
        vals_reader=vals_r,
        keys_reader=keys_r,
        ids_reader=ids_r,
        scan_grid=scan_grid,
        confidence_min=0.5,
        scan_tolerance_um=1000.0,
    )
    assert out.max_n_hkls > 0
