"""Build the (Data.bin, nData.bin) bin index from a flat observed-spots table.

Mirrors `SaveBinData.c` from FF_HEDM. The key insight: the indexer's
`CompareSpots` only checks the SINGLE bin a theoretical spot lands in. To
allow margin-overlap matching, the binner replicates each observed spot
across all bins reachable within its per-spot eta/omega margins:

    omemargin = MarginOme + 0.5 * StepsizeOrient / |sin(eta_deg)|
    etamargin = rad2deg * atan(MarginEta / RingRadii[ring]) + 0.5 * StepsizeOrient

Bin layout (mirrors GetBin in IndexerOMP.c:115):
    pos = (ring_nr - 1) * (n_eta_bins * n_ome_bins) + iEta * n_ome_bins + iOme
    iEta = floor((180 + eta_deg) / EtaBinSize)
    iOme = floor((180 + omega_deg) / OmeBinSize)

Without this spreading, two spots within MarginOme of each other but
straddling a bin boundary would never match.
"""

from __future__ import annotations

import math

import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def build_bin_index(
    obs: np.ndarray,                 # (n_obs, 9) float64
    *,
    eta_bin_size: float,
    ome_bin_size: float,
    n_rings: int,
    margin_eta: float = 0.0,         # um — eta tolerance (acts as etamargin0)
    margin_ome: float = 0.0,         # deg
    stepsize_orient: float = 0.0,    # deg — also rotation step in C
    ring_radii: dict[int, float] | None = None,
    n_eta_bins: int | None = None,
    n_ome_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (data, ndata) int32 arrays.

    When `margin_eta` / `margin_ome` / `stepsize_orient` / `ring_radii` are
    given, spots are replicated across margin-overlapping bins. With all
    margins zero (the legacy non-spreading mode), each spot lives in only
    its primary bin.

    Parameters
    ----------
    obs : (n_obs, 9) float64
        Observed spots. Col 5 = ring_nr, col 6 = eta_deg, col 2 = omega_deg.
    eta_bin_size, ome_bin_size : float
        Bin widths in degrees.
    n_rings : int
        Number of rings (max ring_nr in the dataset).
    margin_eta : float
        IndexerParams.MarginEta (um). Spread radius in eta is
        `rad2deg * atan(MarginEta / RingRadii[ring]) + 0.5 * stepsize_orient`.
    margin_ome : float
        IndexerParams.MarginOme (deg). Spread radius in omega is
        `margin_ome + 0.5 * stepsize_orient / |sin(eta_deg)|`.
    stepsize_orient : float
        IndexerParams.StepsizeOrient (deg). Adds a half-step buffer to both margins.
    ring_radii : dict[int, float]
        Per-ring radius lookup. Required when `margin_eta > 0`.

    Returns
    -------
    data : np.ndarray int32 — flat list of spot rows (each spot may appear
           in multiple bins).
    ndata : np.ndarray int32 (2 * n_total_bins,) — interleaved (count, offset).
    """
    if n_eta_bins is None:
        n_eta_bins = int(math.ceil(360.0 / eta_bin_size))
    if n_ome_bins is None:
        n_ome_bins = int(math.ceil(360.0 / ome_bin_size))
    n_total_bins = n_rings * n_eta_bins * n_ome_bins

    n_obs = obs.shape[0]
    ring_nr_all = np.round(obs[:, 5]).astype(np.int64)
    eta_all = obs[:, 6]
    ome_all = obs[:, 2]

    rows_per_bin: list[list[int]] = [[] for _ in range(n_total_bins)]

    for r in range(n_obs):
        ring_nr = int(ring_nr_all[r])
        if ring_nr < 1 or ring_nr > n_rings:
            continue
        if ring_radii is not None and ring_nr not in ring_radii:
            continue
        if ring_radii is not None and ring_radii[ring_nr] == 0:
            continue
        eta = float(eta_all[r])
        ome = float(ome_all[r])

        # Per-spot omega margin in degrees
        sin_eta = max(abs(math.sin(eta * DEG2RAD)), 1e-30)
        omemargin = margin_ome + 0.5 * stepsize_orient / sin_eta
        # Per-spot eta margin in degrees
        if ring_radii is not None and margin_eta > 0:
            etamargin = (
                RAD2DEG * math.atan(margin_eta / ring_radii[ring_nr])
                + 0.5 * stepsize_orient
            )
        else:
            etamargin = 0.5 * stepsize_orient

        i_eta_min = int(math.floor((180.0 + eta - etamargin) / eta_bin_size))
        i_eta_max = int(math.floor((180.0 + eta + etamargin) / eta_bin_size))
        i_ome_min = int(math.floor((180.0 + ome - omemargin) / ome_bin_size))
        i_ome_max = int(math.floor((180.0 + ome + omemargin) / ome_bin_size))

        for i_eta_raw in range(i_eta_min, i_eta_max + 1):
            i_eta = i_eta_raw % n_eta_bins
            if i_eta < 0:
                i_eta += n_eta_bins
            for i_ome_raw in range(i_ome_min, i_ome_max + 1):
                i_ome = i_ome_raw % n_ome_bins
                if i_ome < 0:
                    i_ome += n_ome_bins
                pos = (ring_nr - 1) * (n_eta_bins * n_ome_bins) + i_eta * n_ome_bins + i_ome
                rows_per_bin[pos].append(r)

    counts = np.array([len(v) for v in rows_per_bin], dtype=np.int32)
    total = int(counts.sum())
    data = np.empty(total, dtype=np.int32)
    offsets = np.zeros(n_total_bins, dtype=np.int32)
    cursor = 0
    for b in range(n_total_bins):
        offsets[b] = cursor
        if counts[b]:
            data[cursor:cursor + counts[b]] = rows_per_bin[b]
            cursor += counts[b]

    ndata = np.empty(2 * n_total_bins, dtype=np.int32)
    ndata[0::2] = counts
    ndata[1::2] = offsets

    return data, ndata
