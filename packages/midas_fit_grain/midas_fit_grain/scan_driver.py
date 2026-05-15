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


def _write_voxel_csv(path: Path, voxel_idx: int,
                     pos_um: np.ndarray, euler_rad: np.ndarray,
                     lattice: np.ndarray, n_matched: int,
                     final_loss: float) -> None:
    """Emit one ``Result_OrientPos_voxel_N.csv`` line.

    Column order (matches the legacy ``FitOrStrainsScanningOMP`` text
    output, minus the per-spot block which downstream PF consolidation
    rebuilds from the optimizer's match table):

        voxelNr, posX, posY, posZ,
        eulerX_rad, eulerY_rad, eulerZ_rad,
        a, b, c, alpha, beta, gamma,
        n_matched, final_loss
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "voxelNr posX posY posZ "
        "eulerX_rad eulerY_rad eulerZ_rad "
        "a b c alpha beta gamma "
        "n_matched final_loss\n"
    )
    fields = [
        str(int(voxel_idx)),
        *(f"{v:.9f}" for v in pos_um.ravel()),
        *(f"{v:.9f}" for v in euler_rad.ravel()),
        *(f"{v:.9f}" for v in lattice.ravel()),
        str(int(n_matched)),
        f"{final_loss:.9e}",
    ]
    with path.open("w") as f:
        f.write(header)
        f.write(" ".join(fields) + "\n")


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
        _write_voxel_csv(
            csv_path, voxel_idx=v,
            pos_um=result.position.detach().cpu().numpy(),
            euler_rad=result.euler.detach().cpu().numpy(),
            lattice=result.lattice.detach().cpu().numpy(),
            n_matched=result.n_matched,
            final_loss=result.final_loss,
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
