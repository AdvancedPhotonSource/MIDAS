"""Public library API: the `Indexer` class.

Two construction paths:
  - `Indexer.from_param_file("paramstest.txt", device=...)` — file-driven.
  - `Indexer(params, device=..., dtype=...)` — programmatic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

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

        if cwd is None:
            cwd = os.path.dirname(self.params.OutputFolder.rstrip("/")) or "."
        cwd = Path(cwd)

        if spots is None:
            _, spots = read_spots(cwd)
        if bins is None:
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
        return run_block(ctx, spot_ids, block_nr=block_nr, n_blocks=n_blocks)
