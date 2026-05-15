"""Public library API: the `Indexer` class.

Two construction paths:
  - `Indexer.from_param_file("paramstest.txt", device=...)` — file-driven.
  - `Indexer(params, device=..., dtype=...)` — programmatic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from .device import apply_cpu_threads, resolve_device, resolve_dtype
from .params import IndexerParams

if TYPE_CHECKING:
    from .result import IndexerResult


class Indexer:
    """Top-level entry point. Wraps the full pipeline."""

    def __init__(
        self,
        params: IndexerParams,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        self.params = params
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(self.device, dtype)
        self._observations: dict | None = None

    @classmethod
    def from_param_file(
        cls,
        path: str | os.PathLike,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "Indexer":
        from .io.params import read_params

        return cls(read_params(path), device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Loading observations (file-driven or programmatic)
    # ------------------------------------------------------------------

    def load_observations(
        self,
        cwd: str | Path | None = None,
        *,
        spots: np.ndarray | torch.Tensor | None = None,
        bins: tuple[np.ndarray, np.ndarray] | None = None,
        hkls: tuple[np.ndarray, np.ndarray] | None = None,
        spot_ids: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """Load Spots.bin, Data.bin, nData.bin, hkls.csv, SpotsToIndex.csv.

        File-driven: pass `cwd` (the directory containing the binaries; this
        defaults to `dirname(OutputFolder)` per IndexerOMP.c:2230). All other
        kwargs override the on-disk file with explicit data (useful for
        synthetic / unit-test cases).
        """
        from .io import (
            read_bins,
            read_grains_csv,
            read_hkls_csv,
            read_spots,
            read_spots_to_index_csv,
            write_spots_to_index_csv,
        )
        from .io.binary import read_bins_scanning

        if cwd is None:
            cwd = os.path.dirname(self.params.OutputFolder.rstrip("/")) or "."
        cwd = Path(cwd)

        if spots is None:
            _, spots = read_spots(cwd)
        if bins is None:
            # PF / scanning fixtures emit Data.bin + nData.bin as int64 with
            # (spot_id, scan_nr) pairs / (count, offset) pairs respectively
            # (SaveBinDataScanning.c:672-700). FF fixtures use int32.
            # Disambiguate via the Spots.bin column count: 10-col = PF.
            spots_arr = np.asarray(spots)
            n_cols = spots_arr.shape[1] if spots_arr.ndim == 2 else 0
            if n_cols >= 10:
                bins = read_bins_scanning(cwd)
            else:
                bins = read_bins(cwd)
        if hkls is None:
            hkls = read_hkls_csv("hkls.csv", ring_numbers=self.params.RingNumbers)

        if spot_ids is None:
            sti = cwd / "SpotsToIndex.csv"
            if not sti.exists() and self.params.isGrainsInput:
                # Mode A: derive SpotsToIndex.csv from Grains.csv
                grains_path = self.params.GrainsFileName
                if not Path(grains_path).is_absolute():
                    grains_path = str(cwd / grains_path)
                grains = read_grains_csv(grains_path)
                # Default mode-A row layout: (newID=grainID, origID=grainID)
                pairs = [(int(g), int(g)) for g in grains["ids"]]
                write_spots_to_index_csv(sti, pairs)
            spot_ids = read_spots_to_index_csv(sti)

        self._observations = {
            "spots": np.asarray(spots),
            "bin_data": np.asarray(bins[0]),
            "bin_ndata": np.asarray(bins[1]),
            "hkls_real": np.asarray(hkls[0]),
            "hkls_int": np.asarray(hkls[1]),
            "spot_ids": np.asarray(spot_ids).astype(np.int64),
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        block_nr: int = 0,
        n_blocks: int = 1,
        n_spots_to_index: int | None = None,
        num_procs: int = 1,
        seed_group_size: int | None = None,
    ) -> "IndexerResult":
        """Run the indexer on `[block_nr/n_blocks]` of the seed list."""
        from .pipeline import IndexerContext, run_block

        if self._observations is None:
            self.load_observations()
        obs = self._observations
        assert obs is not None

        apply_cpu_threads(num_procs, self.device)

        ctx = IndexerContext(
            params=self.params,
            hkls_real=obs["hkls_real"],
            hkls_int=obs["hkls_int"],
            obs=obs["spots"],
            bin_data=obs["bin_data"],
            bin_ndata=obs["bin_ndata"],
            device=self.device,
            dtype=self.dtype,
        )

        spot_ids = torch.as_tensor(obs["spot_ids"], dtype=torch.int64)
        if n_spots_to_index is not None:
            spot_ids = spot_ids[:n_spots_to_index]
        return run_block(ctx, spot_ids, block_nr=block_nr, n_blocks=n_blocks,
                         seed_group_size=seed_group_size)

    # ------------------------------------------------------------------
    # Scan-aware run (pf-HEDM)
    # ------------------------------------------------------------------

    def run_scanning(
        self,
        scan_positions: np.ndarray | torch.Tensor,
        *,
        out_path: str | Path,
        n_spots_to_index: int | None = None,
        num_procs: int = 1,
        seed_group_size: int | None = None,
        voxel_block_nr: int = 0,
        voxel_n_blocks: int = 1,
    ) -> int:
        """Run the per-voxel scanning indexer (pf-HEDM mode).

        Iterates over the (n_scans × n_scans) voxel grid built as the
        Cartesian product of ``scan_positions`` (1-D Y values, µm). For
        each voxel, sets the scan-aware kwargs on ``IndexerContext`` and
        runs the full seed pipeline; collects each voxel's solutions
        into a list. After the loop, writes the consolidated
        ``IndexBest_all.bin`` per the C
        ``IndexerScanningOMP``/``IndexerConsolidatedIO.h`` byte layout.

        Notes
        -----
        - The voxel grid layout matches
          ``IndexerScanningOMP.c:1667-1683``: ``grid[i*nScans + j] =
          (scan_positions[j], scan_positions[i])`` — Cartesian product
          of sorted 1-D Y positions (scan-axis is Y only per P0 audit).
        - ``params.scan_pos_tol_um`` and
          ``params.friedel_symmetric_scan_filter`` drive the per-voxel
          filter inside ``compare_spots``. ``scan_pos_tol_um == 0`` ⇒
          filter inactive (degenerates to FF behavior per voxel, useful
          for sanity).
        - For very large grids the per-voxel cost is significant — call
          with ``voxel_block_nr/voxel_n_blocks > 1`` to shard.

        Returns
        -------
        int
            Number of voxels processed (== ``end - start`` over the
            sharded range).
        """
        from .pipeline import IndexerContext, run_block
        from .io.consolidated import write_index_best_all

        # 1. Validate scan positions BEFORE touching the context — we'd
        # rather fail with a clear ValueError than dive into the
        # pipeline's IndexerContext constructor which may need
        # configured params (EtaBinSize etc.).
        scan_positions_t = torch.as_tensor(
            np.asarray(scan_positions), dtype=self.dtype, device=self.device,
        ).view(-1)
        n_scans = int(scan_positions_t.numel())
        if n_scans < 2:
            raise ValueError(
                f"run_scanning requires n_scans >= 2; got {n_scans}. "
                "Use run() for the FF (single-scan) case."
            )

        if self._observations is None:
            self.load_observations()
        obs = self._observations
        assert obs is not None
        apply_cpu_threads(num_procs, self.device)

        # 2. Build context.
        ctx = IndexerContext(
            params=self.params,
            hkls_real=obs["hkls_real"],
            hkls_int=obs["hkls_int"],
            obs=obs["spots"],
            bin_data=obs["bin_data"],
            bin_ndata=obs["bin_ndata"],
            device=self.device,
            dtype=self.dtype,
        )
        ctx.scan_positions = scan_positions_t

        # 3. Build voxel grid: nVox = nScans * nScans, voxel_xy[v] =
        # (scan_positions[v % nScans], scan_positions[v // nScans]).
        # Matches IndexerScanningOMP.c:1667-1683.
        # i-axis = "row" (y); j-axis = "col" (x). v = i * nScans + j.
        idx = torch.arange(n_scans * n_scans, device=self.device)
        i_idx = idx // n_scans
        j_idx = idx % n_scans
        voxel_xy_table = torch.stack(
            [scan_positions_t[j_idx], scan_positions_t[i_idx]], dim=-1,
        )                                     # (nVox, 2)
        n_vox = int(voxel_xy_table.shape[0])

        # 4. Voxel sharding (used for cluster runs).
        if voxel_n_blocks < 1 or voxel_block_nr < 0 or voxel_block_nr >= voxel_n_blocks:
            raise ValueError(
                f"invalid voxel sharding: block={voxel_block_nr}, n={voxel_n_blocks}"
            )
        block_size = (n_vox + voxel_n_blocks - 1) // voxel_n_blocks
        v_start = voxel_block_nr * block_size
        v_end = min(v_start + block_size, n_vox)

        # 5. Initial spot-ids list (same for every voxel; the scan filter
        # decides per-voxel which spots are admissible).
        spot_ids = torch.as_tensor(obs["spot_ids"], dtype=torch.int64)
        if n_spots_to_index is not None:
            spot_ids = spot_ids[:n_spots_to_index]

        # 5b. Per-seed scan-aware pre-filter (mirrors IndexerScanningOMP.c:1786-1793).
        # Builds (omega_rad, scan_y_obs) for every seed once, before the voxel
        # loop. Per voxel we then compute s_proj = x*sin(omega) + y*cos(omega)
        # vectorised over seeds, keep seeds with |s_proj - scan_y_obs| <= tol.
        # Without this pre-filter, every voxel re-runs ALL seeds through the
        # full orientation grid + compare_spots, even those that the C
        # reference would have rejected in O(1) at the outer loop — dominant
        # perf hotspot in the per-voxel solve.
        obs_np = np.asarray(obs["spots"])
        if obs_np.shape[1] < 10:
            raise ValueError(
                "scanning indexer needs 10-col Spots.bin (PF layout); got "
                f"{obs_np.shape[1]} cols. Check Spots.bin emitter."
            )
        obs_id_to_row = {int(v): i for i, v in enumerate(obs_np[:, 4].astype(np.int64))}
        seed_ids_np = spot_ids.cpu().numpy().astype(np.int64)
        seed_obs_rows = np.array(
            [obs_id_to_row.get(int(sid), -1) for sid in seed_ids_np], dtype=np.int64,
        )
        seed_has_obs = seed_obs_rows >= 0
        # Use row 0 as a safe placeholder for unmatched seeds; mask anyway.
        seed_obs_rows_safe = np.where(seed_has_obs, seed_obs_rows, 0)
        seed_omega_deg = obs_np[seed_obs_rows_safe, 2]
        seed_scan_nr = obs_np[seed_obs_rows_safe, 9].astype(np.int64)
        seed_omega_rad_np = np.deg2rad(seed_omega_deg)
        seed_sin_ome = np.sin(seed_omega_rad_np)
        seed_cos_ome = np.cos(seed_omega_rad_np)
        # ypos[seed_scan_nr] — np ndarray (n_seeds,)
        scan_positions_np = scan_positions_t.cpu().numpy().astype(np.float64)
        n_scans_pos = scan_positions_np.size
        seed_scan_nr_clamped = np.clip(seed_scan_nr, 0, n_scans_pos - 1)
        seed_scan_y_obs = scan_positions_np[seed_scan_nr_clamped]
        scan_pos_tol = float(ctx.scan_pos_tol_um)
        friedel_sym = bool(ctx.friedel_symmetric_scan_filter)
        pre_filter_enabled = scan_pos_tol > 0
        voxel_xy_np = voxel_xy_table.cpu().numpy().astype(np.float64)

        # 6. Per-voxel loop. Collect each voxel's seed results into a
        # (n_solutions, 16) float64 record block matching the C
        # IndexerScanningOMP consolidated layout.
        per_voxel_records: list[np.ndarray] = [
            np.zeros((0, 16), dtype=np.float64) for _ in range(n_vox)
        ]
        for v in range(v_start, v_end):
            ctx.current_voxel_xy = voxel_xy_table[v]
            voxel_seeds = spot_ids
            if pre_filter_enabled:
                vx, vy = voxel_xy_np[v]
                s_proj = vx * seed_sin_ome + vy * seed_cos_ome
                diff = np.abs(s_proj - seed_scan_y_obs)
                ok = diff <= scan_pos_tol
                if friedel_sym:
                    diff_friedel = np.abs(s_proj + seed_scan_y_obs)
                    ok = ok | (diff_friedel <= scan_pos_tol)
                ok = ok & seed_has_obs
                if not ok.any():
                    continue  # voxel has no seeds; record block stays empty
                voxel_seeds = torch.as_tensor(
                    seed_ids_np[ok], dtype=torch.int64,
                )
            voxel_result = run_block(
                ctx, voxel_seeds,
                block_nr=0, n_blocks=1,
                seed_group_size=seed_group_size,
            )
            per_voxel_records[v] = _seeds_to_record_block(voxel_result)

        # 7. Write consolidated output.
        write_index_best_all(out_path, per_voxel_records)
        return v_end - v_start


def _seeds_to_record_block(result) -> np.ndarray:
    """Convert an IndexerResult into the (n_solutions, 16) record layout.

    Column map (matches the IndexerScanningOMP.c writer / pf_MIDAS.py
    parser semantics):

        col 0  : seed spot id (mirrors the C convention)
        col 1  : avg internal-angle score (radians)
        col 2-10: 9-element orientation matrix (row-major 3×3)
        col 11 : posX (sample-frame x, µm) — from best_pos[0]
        col 12 : posY (sample-frame y, µm) — from best_pos[1]
        col 13 : posZ (sample-frame z, µm) — from best_pos[2]
        col 14 : nExpected (total predicted spots, denominator)
        col 15 : nMatches (matched predicted spots, numerator)

    Empty seeds (no matches) are dropped. One row per accepted seed.
    """
    rows: list[list[float]] = []
    for s in result.seeds:
        if s.n_matches <= 0:
            continue
        om = s.best_or_mat.detach().cpu().numpy().reshape(9)
        pos = s.best_pos.detach().cpu().numpy().reshape(3)
        rows.append([
            float(s.spot_id),                          # 0
            float(s.avg_ia),                           # 1
            *[float(x) for x in om],                   # 2-10
            float(pos[0]), float(pos[1]), float(pos[2]),  # 11-13
            float(s.n_t_spots),                        # 14: nExpected
            float(s.n_matches),                        # 15: nMatches
        ])
    if not rows:
        return np.zeros((0, 16), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
