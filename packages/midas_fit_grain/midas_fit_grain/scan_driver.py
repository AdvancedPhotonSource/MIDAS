"""Per-voxel scanning refinement orchestrator (P6 batch driver).

Thin shim that mirrors the structural pattern of
``midas_index.indexer.run_scanning``:

  1. Read ``positions.csv`` → 1-D Y scan positions (µm).
  2. Read ``IndexBest_all.bin`` (the consolidated output emitted by
     ``midas_index.run_scanning``) → per-voxel candidate records.
  3. For each voxel:
     a. Pick the top candidate (highest completeness ratio) from the
        record block.
     b. Stamp ``cfg.scan_pos_tol_um`` and ``cfg.beam_size_um`` so the
        scan-aware filter + position-mode logic in
        ``refine_grain`` takes effect.
     c. Call ``refine_grain`` with the C-parity contract
        (``position_mode="fixed"``, ``mode="all_at_once"``).
  4. Emit per-voxel result CSVs into ``<results_dir>/Result_OrientPos_voxel_N.csv``.

Notes
-----
- The per-voxel work is independent, so this loop is trivially
  parallelisable; we keep it serial in v1 for parity with the
  ``Indexer.run_scanning`` reference implementation.
- The full residual emission (per-spot mapping etc.) is delegated to
  the existing ``refine_grain`` plumbing; this driver only adds the
  voxel-iteration shell + CSV writer.
- Differentiability: ``refine_grain`` is torch-native end-to-end; the
  voxel loop preserves that contract per voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch


@dataclass
class ScanVoxelResult:
    """Per-voxel refinement summary."""

    voxel_idx: int
    n_solutions_in: int          # candidates the indexer produced
    final_loss: float            # post-refine residual sum-of-squares
    n_matched: int               # observed spots matched after refinement
    converged: bool
    position_um: np.ndarray      # (3,) post-refine position
    euler_rad: np.ndarray        # (3,) post-refine Euler angles
    lattice: np.ndarray          # (6,) post-refine lattice
    csv_path: Path


def _read_index_best_all(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Parse IndexBest_all.bin into (n_sol_arr, [per-voxel records])."""
    raw = path.read_bytes()
    n_voxels = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    cursor = 4
    n_sol_arr = np.frombuffer(raw[cursor:cursor + 4 * n_voxels],
                              dtype=np.int32).copy()
    cursor += 4 * n_voxels
    cursor += 8 * n_voxels      # off_arr — recomputable from cumulative
    vals = np.frombuffer(raw[cursor:], dtype=np.float64).reshape(-1, 16).copy()
    blocks: List[np.ndarray] = []
    pos = 0
    for n in n_sol_arr:
        blocks.append(vals[pos:pos + int(n)].copy())
        pos += int(n)
    return n_sol_arr, blocks


def _top_candidate(block: np.ndarray) -> Optional[np.ndarray]:
    """Pick the candidate with highest completeness (col 15 / col 14)."""
    if block.shape[0] == 0:
        return None
    n_expected = np.maximum(block[:, 14], 1.0)
    completeness = block[:, 15] / n_expected
    return block[int(np.argmax(completeness))]


def _euler_zxz_to_om_np(eulers_rad: np.ndarray) -> np.ndarray:
    """Active ZXZ rotation, matches midas_stress.orientation convention."""
    p1, p, p2 = eulers_rad
    c1, s1 = np.cos(p1), np.sin(p1)
    c2, s2 = np.cos(p2), np.sin(p2)
    cp, sp = np.cos(p), np.sin(p)
    return np.array([
        [c1*c2 - s1*s2*cp, -c1*s2 - s1*c2*cp,  s1*sp],
        [s1*c2 + c1*s2*cp, -s1*s2 + c1*c2*cp, -c1*sp],
        [s2*sp,             c2*sp,              cp  ],
    ])


def _euler_zxz_to_quat_np(eulers_rad: np.ndarray) -> np.ndarray:
    """ZXZ Euler → unit quaternion (w, x, y, z)."""
    p1, p, p2 = eulers_rad
    cp1, sp1 = np.cos(p1/2), np.sin(p1/2)
    cp2, sp2 = np.cos(p2/2), np.sin(p2/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    w = cp1*cp*cp2 - sp1*cp*sp2
    x = cp1*sp*cp2 + sp1*sp*sp2
    y = sp1*sp*cp2 - cp1*sp*sp2
    z = cp1*cp*sp2 + sp1*cp*cp2
    return np.array([w, x, y, z])


def _write_voxel_csv(path: Path, voxel_idx: int,
                     pos_um: np.ndarray, euler_rad: np.ndarray,
                     lattice: np.ndarray, n_matched: int,
                     final_loss: float, *,
                     n_expected: int = 0,
                     pos_err_um: float = 0.0,
                     ome_err_deg: float = 0.0,
                     internal_angle_deg: float = 0.0) -> None:
    """Emit one per-voxel result CSV in the 43-col layout consumed by
    ``midas_pipeline.stages.consolidation_pf``.

    Layout matches the legacy ``FitOrStrainsScanningOMP`` writer:

        col 0    : SpotID (= voxel_idx; the legacy file used the matched
                   spot ID, but PF consolidation only reads col 26 +
                   2..10 + 11..13 + 15..20 + 26 + 36..38, so SpotID is
                   a label).
        col 1-9  : OM (row-major 3×3) computed from euler.
        col 10   : SpotID (= voxel_idx).
        col 11-13: posX posY posZ (µm).
        col 14   : SpotID (= voxel_idx).
        col 15-20: a b c alpha beta gamma.
        col 21   : SpotID (= voxel_idx).
        col 22   : PosErr (µm).
        col 23   : OmeErr (deg).
        col 24   : InternalAngle (deg).
        col 25   : Radius (= 1.0 placeholder).
        col 26   : Completeness = n_matched / max(n_expected, 1).
        col 27-35: Strain E11..E33 (zeros; our refiner doesn't compute strain).
        col 36-38: Euler angles (Eul1, Eul2, Eul3) — Python convention.
        col 39-42: Quaternion (w, x, y, z).

    Empty / failed voxels write a header-only file so the consolidator's
    per-voxel scan still gets a hit but ``_row_is_accepted`` rejects it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    om = _euler_zxz_to_om_np(np.asarray(euler_rad, dtype=np.float64).ravel())
    quat = _euler_zxz_to_quat_np(np.asarray(euler_rad, dtype=np.float64).ravel())
    sid = float(voxel_idx)
    completeness = float(n_matched) / max(int(n_expected), 1)
    pos = np.asarray(pos_um, dtype=np.float64).ravel()
    lat = np.asarray(lattice, dtype=np.float64).ravel()
    eul = np.asarray(euler_rad, dtype=np.float64).ravel()
    vals = [
        sid,                              # 0
        *om.ravel().tolist(),             # 1-9
        sid,                              # 10
        pos[0], pos[1], pos[2],           # 11-13
        sid,                              # 14
        lat[0], lat[1], lat[2], lat[3], lat[4], lat[5],  # 15-20
        sid,                              # 21
        float(pos_err_um),                # 22
        float(ome_err_deg),               # 23
        float(internal_angle_deg),        # 24
        1.0,                              # 25 Radius placeholder
        completeness,                     # 26
        *[0.0] * 9,                       # 27-35 strain E11..E33
        eul[0], eul[1], eul[2],           # 36-38
        quat[0], quat[1], quat[2], quat[3],  # 39-42
    ]
    header = (
        "SpotID O11 O12 O13 O21 O22 O23 O31 O32 O33 "
        "SpotID x y z "
        "SpotID a b c alpha beta gamma "
        "SpotID PosErr OmeErr InternalAngle Radius Completeness "
        "E11 E12 E13 E21 E22 E23 E31 E32 E33 "
        "Eul1 Eul2 Eul3 Quat1 Quat2 Quat3 Quat4\n"
    )
    line = " ".join(f"{v:.9f}" if not isinstance(v, int) else str(v) for v in vals)
    with path.open("w") as f:
        f.write(header)
        f.write(line + "\n")


def refine_scanning_block(
    cfg,                                # FitConfig (avoid circular import)
    *,
    index_best_all: str | Path,
    positions_csv: str | Path,
    results_dir: str | Path,
    model,                              # HEDMForwardModel
    obs,                                # ObservedSpots
    pred_ring_slot: torch.Tensor,
    voxel_block_nr: int = 0,
    voxel_n_blocks: int = 1,
    on_voxel: Optional[callable] = None,   # callback(voxel_idx, ScanVoxelResult)
) -> List[ScanVoxelResult]:
    """Per-voxel scan-aware refinement orchestrator.

    Returns the per-voxel results for the requested shard. Voxels with
    no indexer candidates are skipped (no CSV written).
    """
    from .config import FitConfig    # local import to keep module light
    from .refine import refine_grain

    assert isinstance(cfg, FitConfig)
    index_best_all = Path(index_best_all)
    positions_csv = Path(positions_csv)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scan_positions = np.loadtxt(positions_csv).astype(np.float64)
    if scan_positions.ndim == 0:
        scan_positions = scan_positions.reshape(1)
    n_scans = scan_positions.size
    if n_scans < 2:
        raise ValueError(
            f"refine_scanning_block requires n_scans >= 2; got {n_scans}. "
            "Use refine_block / refine_grain for the FF single-scan case."
        )

    _, blocks = _read_index_best_all(index_best_all)
    n_vox = len(blocks)
    if n_vox != n_scans * n_scans:
        raise ValueError(
            f"voxel count mismatch: IndexBest_all has {n_vox}, but "
            f"positions.csv implies {n_scans * n_scans} ({n_scans}^2)."
        )

    # Same Cartesian-product layout as Indexer.run_scanning + the C
    # IndexerScanningOMP at lines 1667-1683.
    idx = np.arange(n_vox)
    i_idx = idx // n_scans                # row (y)
    j_idx = idx % n_scans                 # col (x)
    voxel_xy_table = np.stack(
        [scan_positions[j_idx], scan_positions[i_idx]], axis=-1,
    )

    # Voxel sharding.
    if voxel_n_blocks < 1 or voxel_block_nr < 0 or voxel_block_nr >= voxel_n_blocks:
        raise ValueError(
            f"invalid voxel sharding: block={voxel_block_nr}, n={voxel_n_blocks}"
        )
    block_size = (n_vox + voxel_n_blocks - 1) // voxel_n_blocks
    v_start = voxel_block_nr * block_size
    v_end = min(v_start + block_size, n_vox)

    # Stamp the scan-aware kwargs on the cfg once. ``refine_grain``
    # reads ``cfg.scan_pos_tol_um`` / ``cfg.position_mode`` /
    # ``cfg.beam_size_um`` and only activates when scan_pos_tol_um > 0.
    if cfg.scan_pos_tol_um <= 0:
        cfg.scan_pos_tol_um = 1.5    # production default, plan §1b

    out: List[ScanVoxelResult] = []
    for v in range(v_start, v_end):
        block = blocks[v]
        cand = _top_candidate(block)
        if cand is None:
            continue                                                    # no indexer hit
        om = cand[2:11].reshape(3, 3)
        # Convert OM → Euler (use a torch-native path so the refiner
        # accepts the seed cleanly).
        from midas_stress.orientation import orient_mat_to_euler
        euler = np.asarray(orient_mat_to_euler(om.ravel().tolist()),
                           dtype=np.float64)

        init_pos = torch.tensor(
            [voxel_xy_table[v, 0], voxel_xy_table[v, 1], 0.0],
            dtype=torch.float64,
        )
        init_eul = torch.tensor(euler, dtype=torch.float64)
        init_lat = torch.tensor(cfg.LatticeConstant, dtype=torch.float64)

        result = refine_grain(
            cfg, model=model, obs=obs,
            init_position=init_pos,
            init_euler=init_eul,
            init_lattice=init_lat,
            pred_ring_slot=pred_ring_slot,
        )

        csv_path = results_dir / f"Result_OrientPos_voxel_{v}.csv"
        # Extract completeness denominator + per-spot residual stats from
        # the final match. ``n_t_frac`` is the n_expected denominator
        # the C refiner uses for Completeness.
        try:
            n_expected = int(result.match.n_t_frac[0].item())
        except (AttributeError, IndexError, TypeError):
            n_expected = max(int(result.n_matched), 1)
        try:
            res = result.per_spot_residuals.detach().cpu().numpy()
            pos_err_um = float(np.sqrt(np.mean(res[:, :3] ** 2))) if res.size else 0.0
            ome_err_deg = (
                float(np.sqrt(np.mean(res[:, 3] ** 2)))
                if res.size and res.shape[1] > 3 else 0.0
            )
        except (AttributeError, IndexError, TypeError):
            pos_err_um = 0.0
            ome_err_deg = 0.0
        _write_voxel_csv(
            csv_path, voxel_idx=v,
            pos_um=result.position.detach().cpu().numpy(),
            euler_rad=result.euler.detach().cpu().numpy(),
            lattice=result.lattice.detach().cpu().numpy(),
            n_matched=result.n_matched,
            final_loss=result.final_loss,
            n_expected=n_expected,
            pos_err_um=pos_err_um,
            ome_err_deg=ome_err_deg,
        )
        vr = ScanVoxelResult(
            voxel_idx=v,
            n_solutions_in=int(block.shape[0]),
            final_loss=float(result.final_loss),
            n_matched=int(result.n_matched),
            converged=bool(result.converged),
            position_um=result.position.detach().cpu().numpy(),
            euler_rad=result.euler.detach().cpu().numpy(),
            lattice=result.lattice.detach().cpu().numpy(),
            csv_path=csv_path,
        )
        if on_voxel is not None:
            on_voxel(v, vr)
        out.append(vr)
    return out
