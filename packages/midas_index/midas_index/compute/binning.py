"""Binned-spot index lookups.

Mirrors `GetBin` from `FF_HEDM/src/IndexerOMP.c:115`. Given (ring_nr, eta,
omega) of a theoretical spot, computes a flat bin index and gathers the
candidate observed-spot row IDs from `data`/`ndata` arrays.

Vectorized: works on arbitrary leading-dim tensors of theoretical spots and
returns (n_in_bin, data_offset) per element. The actual gather of candidate
rows is left to `compare_spots` (it depends on the matching strategy —
dense vs. jagged).
"""

from __future__ import annotations

import torch


def get_bin_indices(
    ring_nr: torch.Tensor,        # int (..)
    eta_deg: torch.Tensor,        # float (..)
    omega_deg: torch.Tensor,      # float (..)
    eta_bin_size: float,
    ome_bin_size: float,
    n_eta_bins: int,
    n_ome_bins: int,
) -> torch.Tensor:
    """Compute flat bin indices `pos = (ring-1) * (n_eta * n_ome) + iEta * n_ome + iOme`.

    Mirrors C: `iEta = floor((180 + eta) / EtaBinSize)`, `iOme` likewise.
    """
    i_ring = (ring_nr.to(torch.int64) - 1)
    i_eta = torch.floor((180.0 + eta_deg) / eta_bin_size).to(torch.int64)
    i_ome = torch.floor((180.0 + omega_deg) / ome_bin_size).to(torch.int64)
    return i_ring * (n_eta_bins * n_ome_bins) + i_eta * n_ome_bins + i_ome


def lookup_bin_counts(
    pos: torch.Tensor,            # int64 (..,)
    ndata: torch.Tensor,          # int32 (2*n_bins,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (n_in_bin, data_offset) for each pos.

    `ndata` layout is interleaved: `[count_0, offset_0, count_1, offset_1, ...]`.
    """
    pos = pos.to(torch.int64)
    n_per = ndata[pos * 2].to(torch.int64)
    offset = ndata[pos * 2 + 1].to(torch.int64)
    return n_per, offset
