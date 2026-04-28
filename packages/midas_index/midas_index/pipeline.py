"""High-level pipeline orchestration.

Mirrors `FF_HEDM/src/IndexerOMP.c::main` flow:

  1. Parse argv -> (param_file, block_nr, n_blocks, n_spots_to_index, num_procs)
  2. ReadParams(param_file)                              -> IndexerParams
  3. read hkls.csv                                        -> hkls table
  4. read Bins.bin / nData.bin                            -> binned spot index
  5. read Spots.bin                                       -> ObsSpotsLab
  6. if isGrainsInput: build SpotsToIndex.csv from Grains.csv (mode A)
     load SpotsToIndex.csv                                -> seed spot IDs
  7. compute startRowNr, endRowNr from block sharding
  8. for spot_id in seeds[startRowNr:endRowNr]:
         process_seed(...)                                <- the per-seed kernel
         write best result to IndexBest.bin
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .compute import (
    binning,
    forward_adapter,
    matching,
    orientation_grid,
    position_grid,
    reduce as reduce_,
    seeds as seeds_module,
)
from .params import IndexerParams
from .result import IndexerResult, SeedResult

if TYPE_CHECKING:
    pass

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Pre-computed indexer context (per-Indexer; reused across seeds)
# ---------------------------------------------------------------------------


class IndexerContext:
    """Shared, immutable context built once per `Indexer.run`.

    Holds device-resident tensors for hkls, observed spots, bin index, margin
    LUTs, and the forward adapter. One per Indexer invocation.
    """

    def __init__(
        self,
        params: IndexerParams,
        hkls_real: np.ndarray | torch.Tensor,
        hkls_int: np.ndarray | torch.Tensor,
        obs: np.ndarray | torch.Tensor,
        bin_data: np.ndarray | torch.Tensor,
        bin_ndata: np.ndarray | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.params = params
        self.device = device
        self.dtype = dtype

        self.hkls_real = torch.as_tensor(np.asarray(hkls_real), device=device, dtype=dtype)
        self.hkls_int = torch.as_tensor(np.asarray(hkls_int), device=device, dtype=torch.long)
        self.obs = torch.as_tensor(np.asarray(obs), device=device, dtype=dtype)
        self.bin_data = torch.as_tensor(np.asarray(bin_data), device=device, dtype=torch.int32)
        self.bin_ndata = torch.as_tensor(np.asarray(bin_ndata), device=device, dtype=torch.int32)

        # Per-ring cached HKL row index. For seed processing we need to find
        # the HKL Cartesian (g1, g2, g3) for a given ring number. C code stores
        # this in `RingHKL[ringnr]` (last-seen wins per IndexerOMP.c:2202-2205).
        # We replicate by iterating in order; later rings overwrite if duplicate.
        self.ring_hkl: dict[int, torch.Tensor] = {}
        self.ring_ttheta: dict[int, float] = {}
        for i in range(self.hkls_real.shape[0]):
            rn = int(self.hkls_real[i, 3].item())
            self.ring_hkl[rn] = self.hkls_real[i, :3].clone()
            self.ring_ttheta[rn] = float(self.hkls_real[i, 5].item() * 2 * RAD2DEG)

        # Bin geometry
        self.n_eta_bins = int(math.ceil(360.0 / params.EtaBinSize))
        self.n_ome_bins = int(math.ceil(360.0 / params.OmeBinSize))

        # Margin LUTs
        self.eta_margins = matching.build_eta_margins(
            ring_radii=params.RingRadii,
            margin_eta=params.MarginEta,
            stepsize_orient_deg=params.StepsizeOrient,
            device=device, dtype=dtype,
        )
        self.ome_margins = matching.build_ome_margins(
            margin_ome=params.MarginOme,
            stepsize_orient_deg=params.StepsizeOrient,
            device=device, dtype=dtype,
        )

        self.rings_to_reject = torch.tensor(
            params.RingsToReject if params.RingsToReject else [],
            device=device, dtype=torch.int64,
        )

        # Forward adapter (constructs HEDMForwardModel internally)
        self.adapter = forward_adapter.IndexerForwardAdapter(
            params=params,
            hkls_real=self.hkls_real,
            hkls_int=self.hkls_int,
            device=device,
            dtype=dtype,
        )

    def find_obs_row_by_id(self, spot_id: int) -> int:
        """Return the row index of the observed spot whose column 4 equals `spot_id`."""
        col = self.obs[:, 4]
        idx = torch.where(col == float(spot_id))[0]
        if idx.numel() == 0:
            return -1
        return int(idx[0].item())


# ---------------------------------------------------------------------------
# Per-seed kernel
# ---------------------------------------------------------------------------


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v).clamp_min(1e-30)


def _spot_to_gv(
    distance: float, y0: float, z0: float, omega_deg: float,
    *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """G-vector for a Bragg spot at (Lsd, y0, z0) measured at omega.

    Mirrors `MakeUnitLength` + `spot_to_gv` from `FF_HEDM/src/IndexerOMP.c:726`,
    used as the seed's `hklnormal` in GenerateCandidateOrientationsF (line 1849).
    """
    L = math.sqrt(distance * distance + y0 * y0 + z0 * z0)
    xn = distance / L
    yn = y0 / L
    zn = z0 / L
    g1r = -1.0 + xn
    g2r = yn
    co = math.cos(-omega_deg * DEG2RAD)
    so = math.sin(-omega_deg * DEG2RAD)
    g1 = g1r * co - g2r * so
    g2 = g1r * so + g2r * co
    g3 = zn
    return torch.tensor([g1, g2, g3], device=device, dtype=dtype)


def process_seed(
    spot_id: int, ctx: IndexerContext,
) -> SeedResult | None:
    """Run the full per-seed indexing kernel for a single spot ID."""
    p = ctx.params
    obs_row = ctx.find_obs_row_by_id(spot_id)
    if obs_row < 0:
        return None

    seed_obs = ctx.obs[obs_row]                            # (9,)
    ys = float(seed_obs[0].item())
    zs = float(seed_obs[1].item())
    seed_omega = float(seed_obs[2].item())
    seed_ring_rad = float(seed_obs[3].item())              # observed radial position
    seed_eta = float(seed_obs[6].item())
    seed_ring_nr = int(seed_obs[5].item())

    if seed_ring_nr not in ctx.ring_hkl:
        return None
    hkl = ctx.ring_hkl[seed_ring_nr]                       # (3,)
    ring_rad_user = p.get_ring_radius(seed_ring_nr)        # canonical ring radius from paramstest
    ring_rad = ring_rad_user if ring_rad_user > 0 else seed_ring_rad
    ttheta = ctx.ring_ttheta[seed_ring_nr]

    # 1. Seed candidates (y0, z0)
    if p.UseFriedelPairs == 1:
        seed_yz = seeds_module.generate_ideal_spots_friedel(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
            ring_nr=seed_ring_nr, ring_rad=ring_rad,
            rsample=p.Rsample, hbeam=p.Hbeam,
            ome_tol=p.MarginOme, radius_tol=p.MarginRad,
            obs_spots=ctx.obs, device=ctx.device, dtype=ctx.dtype,
        )
    elif p.UseFriedelPairs == 2:
        seed_yz = seeds_module.generate_ideal_spots_friedel_mixed(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta, omega_deg=seed_omega,
            ring_nr=seed_ring_nr, ring_rad=ring_rad, lsd=p.Distance,
            rsample=p.Rsample, hbeam=p.Hbeam,
            step_size_pos=p.StepsizePos,
            ome_tol=p.MarginOme, radial_tol=p.MarginRad, eta_tol_um=p.MarginEta,
            obs_spots=ctx.obs, device=ctx.device, dtype=ctx.dtype,
        )
    else:
        seed_yz = seeds_module.generate_ideal_spots(
            ys=ys, zs=zs, ttheta_deg=ttheta, eta_deg=seed_eta,
            ring_rad=ring_rad, rsample=p.Rsample, hbeam=p.Hbeam,
            step_size=p.StepsizePos, device=ctx.device, dtype=ctx.dtype,
        )
    if seed_yz.shape[0] == 0:
        return None

    # 2. Build the global cartesian product of (y0_z0) × orientations × positions
    #    in one shot, so the forward sim and match run as a single batched
    #    tensor op (instead of one Python iteration per y0_z0 candidate).
    hkl_int_for_ring = tuple(
        ctx.hkls_int[ctx.hkls_int[:, 3] == seed_ring_nr][0, :3].tolist()
    )

    R_chunks: list[torch.Tensor] = []
    pos_chunks: list[torch.Tensor] = []
    for k in range(seed_yz.shape[0]):
        y0 = float(seed_yz[k, 0].item())
        z0 = float(seed_yz[k, 1].item())
        plane_normal = _spot_to_gv(
            p.Distance, y0, z0, seed_omega,
            device=ctx.device, dtype=ctx.dtype,
        )
        Rs = orientation_grid.generate_candidate_orientations(
            hkl=hkl, plane_normal=plane_normal,
            stepsize_orient_deg=p.StepsizeOrient,
            ring_nr=seed_ring_nr, space_group=p.SpaceGroup,
            hkl_int=hkl_int_for_ring, abcabg=p.LatticeConstant,
        )
        if Rs.shape[0] == 0:
            continue
        positions_k, _ = position_grid.build_position_grid(
            seed_y0=torch.tensor([y0], device=ctx.device, dtype=ctx.dtype),
            seed_z0=torch.tensor([z0], device=ctx.device, dtype=ctx.dtype),
            ys=ys, zs=zs, omega_deg=seed_omega,
            distance=p.Distance, r_sample=p.Rsample, step_size=p.StepsizePos,
        )
        if positions_k.shape[0] == 0:
            continue

        n_or = Rs.shape[0]
        n_pos = positions_k.shape[0]
        N_k = n_or * n_pos
        R_chunks.append(
            Rs.unsqueeze(1).expand(n_or, n_pos, 3, 3).reshape(N_k, 3, 3)
        )
        pos_chunks.append(
            positions_k.unsqueeze(0).expand(n_or, n_pos, 3).reshape(N_k, 3)
        )

    if not R_chunks:
        return None

    R_all = torch.cat(R_chunks, dim=0)            # (N_total, 3, 3)
    pos_all = torch.cat(pos_chunks, dim=0)        # (N_total, 3)
    N = R_all.shape[0]

    # 3. Single batched forward + match across all candidate tuples.
    theor, valid = ctx.adapter.simulate(R_all, pos_all, lattice=None)
    ref_rad = torch.full((N,), ring_rad, device=ctx.device, dtype=ctx.dtype)
    result = matching.compare_spots(
        theor=theor, valid=valid, obs=ctx.obs,
        bin_data=ctx.bin_data, bin_ndata=ctx.bin_ndata,
        ref_rad=ref_rad,
        margin_rad=p.MarginRad, margin_radial=p.MarginRadial,
        eta_margins=ctx.eta_margins, ome_margins=ctx.ome_margins,
        eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize,
        n_eta_bins=ctx.n_eta_bins, n_ome_bins=ctx.n_ome_bins,
        rings_to_reject=ctx.rings_to_reject,
        distance=p.Distance, pos=pos_all,
    )

    # 4. Reduce: per-seed best tuple via packed-score argmax.
    keys = reduce_.pack_score(result.frac_matches, result.avg_ia)
    if keys.numel() == 0:
        return None
    idx = int(reduce_.best_tuple(keys).item())

    # 5. Compose SeedResult for the winning tuple.
    n_t = int(valid[idx].sum().item())
    n_t_frac = n_t
    if ctx.rings_to_reject.numel() > 0:
        in_reject = (
            theor[idx, :, 9].long().unsqueeze(-1) == ctx.rings_to_reject
        ).any(dim=-1)
        n_t_frac = int((valid[idx] & ~in_reject).sum().item())
    return SeedResult(
        spot_id=spot_id,
        best_or_mat=R_all[idx].detach().clone(),
        best_pos=pos_all[idx].detach().clone(),
        n_matches=int(result.n_matches[idx].item()),
        n_t_spots=n_t,
        n_t_frac_calc=n_t_frac,
        frac_matches=float(result.frac_matches[idx].item()),
        avg_ia=float(result.avg_ia[idx].item()),
        matched_ids=result.matched_obs_id[idx][result.matched[idx]].clone(),
    )


# ---------------------------------------------------------------------------
# Block driver
# ---------------------------------------------------------------------------


def run_block(
    ctx: IndexerContext,
    spot_ids: torch.Tensor,           # int64 (n_total_seeds,)
    block_nr: int,
    n_blocks: int,
) -> IndexerResult:
    """Process one block's slice of spot_ids and return seed results.

    Mirrors IndexerOMP.c:2287-2347. Per-block sharding:
        startRowNr = ceil(n_total / n_blocks) * block_nr
        endRowNr   = min(ceil(n_total / n_blocks) * (block_nr+1) - 1, n_total - 1)
    """
    n_total = int(spot_ids.numel())
    block_size = math.ceil(n_total / max(1, n_blocks))
    start = block_size * block_nr
    end_inclusive = min(block_size * (block_nr + 1) - 1, n_total - 1)
    if start > end_inclusive or start >= n_total:
        return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=[])

    seeds_out: list[SeedResult] = []
    for i in range(start, end_inclusive + 1):
        sid = int(spot_ids[i].item())
        seed = process_seed(sid, ctx)
        if seed is not None:
            seeds_out.append(seed)

    return IndexerResult(block_nr=block_nr, n_blocks=n_blocks, seeds=seeds_out)
