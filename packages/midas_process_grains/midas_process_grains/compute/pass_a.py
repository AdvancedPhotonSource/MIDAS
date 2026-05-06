"""Pass A — post-Phase-1 spot-overlap merge of cluster reps.

Replaces the C ProcessGrains "Pass A" dedup (`misori < 0.1° AND Δpos < 5 µm`,
hardcoded at FF_HEDM/src/ProcessGrains.c:869) with a spot-overlap-based
merge: two Phase-1 cluster reps are merged when their misorientation is
small AND their resolved-SpotID Jaccard is high.

Why: position is unreliable for many indexer/refiner outputs (e.g. C-LMO).
Spot overlap uses what the indexer matched (robust) instead of what it
fitted (noisy).

Algorithm
---------
1. For each Phase-1 cluster, pick a rep (the min-IA seed) and pull its
   IBF col-0 SpotID list (~120 SpotIDs per grain at 7-ring FCC).
2. Build inverted index: SpotID → list of cluster IDs claiming it.
3. Generate candidate cluster pairs by walking the inverted index
   (any pair sharing ≥ 1 SpotID).
4. Vectorised filter: keep pairs with misorientation < tol_rad.
5. Per surviving pair, compute spot-set Jaccard; keep if ≥ jaccard_tol.
6. Connected components on the merge graph → new "super-cluster" labels.
7. Relabel ``cluster.labels`` and rebuild ``members_by_label``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from midas_stress.orientation import misorientation_quat_batch


def _candidate_pairs_from_spot_overlap(
    rep_spot_arrays: list,
) -> np.ndarray:
    """For each spot, emit all pairs of cluster reps claiming it.

    rep_spot_arrays[c] = sorted unique np.int64 SpotIDs for cluster c.

    Returns (m, 2) int64 of unique candidate cluster pairs (i < j).
    """
    spot_to_clusters: Dict[int, list] = defaultdict(list)
    for c, spots in enumerate(rep_spot_arrays):
        for s in spots:
            spot_to_clusters[int(s)].append(c)

    pair_chunks = []
    for cs in spot_to_clusters.values():
        if len(cs) < 2:
            continue
        cs_arr = np.unique(np.asarray(cs, dtype=np.int64))
        if cs_arr.size < 2:
            continue
        i_idx, j_idx = np.triu_indices(cs_arr.size, k=1)
        chunk = np.stack([cs_arr[i_idx], cs_arr[j_idx]], axis=1)
        pair_chunks.append(chunk)

    if not pair_chunks:
        return np.empty((0, 2), dtype=np.int64)

    all_pairs = np.concatenate(pair_chunks, axis=0)
    encoded = all_pairs[:, 0] * np.int64(2**31) + all_pairs[:, 1]
    unique_encoded = np.unique(encoded)
    a = unique_encoded // np.int64(2**31)
    b = unique_encoded % np.int64(2**31)
    return np.stack([a, b], axis=1)


def _jaccard_per_pair(
    pairs: np.ndarray,
    rep_spot_sets: list,
) -> np.ndarray:
    """Compute Jaccard for each (c1, c2) pair, given each cluster's spot set."""
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for i in range(pairs.shape[0]):
        c1, c2 = int(pairs[i, 0]), int(pairs[i, 1])
        s1 = rep_spot_sets[c1]
        s2 = rep_spot_sets[c2]
        if not s1 or not s2:
            out[i] = 0.0
            continue
        inter = len(s1 & s2)
        if inter == 0:
            out[i] = 0.0
        else:
            union = len(s1) + len(s2) - inter
            out[i] = inter / union
    return out


def merge_clusters_by_spot_overlap(
    *,
    cluster_labels: np.ndarray,           # (n_seeds,) int64 from Phase 1
    n_phase1_clusters: int,
    members_by_label: Dict[int, np.ndarray],
    quats_per_seed: np.ndarray,           # (n_seeds, 4) FZ-reduced
    ias_per_seed: np.ndarray,             # (n_seeds,) internal angle
    ibf_alive_col0: np.ndarray,           # (n_alive, n_hkls) int64 SpotIDs
    ibf_global_to_local: np.ndarray,      # (n_seeds,) global pos → row in ibf_alive_col0
    space_group: int,
    misori_tol_rad: float,
    jaccard_tol: float,
    progress: bool = True,
) -> Tuple[np.ndarray, int, Dict[int, np.ndarray]]:
    """Perform Pass A. Returns (new_labels, n_super_clusters, new_members_by_label)."""
    t0 = time.time()

    if n_phase1_clusters < 2:
        return cluster_labels.copy(), n_phase1_clusters, dict(members_by_label)

    # ---- 1. rep selection per cluster ---------------------------------------
    cluster_rep = np.full(n_phase1_clusters, -1, dtype=np.int64)
    for lab, members in members_by_label.items():
        rep_local = int(np.argmin(ias_per_seed[members]))
        cluster_rep[lab] = int(members[rep_local])

    # rep spot lists (sorted unique, positive only)
    rep_spot_arrays = []
    rep_spot_sets = []
    for c in range(n_phase1_clusters):
        rep = int(cluster_rep[c])
        if rep < 0:
            rep_spot_arrays.append(np.empty(0, dtype=np.int64))
            rep_spot_sets.append(set())
            continue
        local = int(ibf_global_to_local[rep])
        spots = ibf_alive_col0[local]
        spots = spots[spots > 0]
        spots_unique = np.unique(spots).astype(np.int64)
        rep_spot_arrays.append(spots_unique)
        rep_spot_sets.append(set(spots_unique.tolist()))

    if progress:
        print(f"[passA] {n_phase1_clusters:,} cluster reps, "
              f"avg {sum(len(s) for s in rep_spot_sets)/max(1,n_phase1_clusters):.1f} spots/rep "
              f"[{time.time()-t0:.1f}s]", flush=True)

    # ---- 2. candidate pairs from spot overlap -------------------------------
    candidate_pairs = _candidate_pairs_from_spot_overlap(rep_spot_arrays)
    if progress:
        print(f"[passA] {candidate_pairs.shape[0]:,} candidate pairs from spot overlap "
              f"[{time.time()-t0:.1f}s]", flush=True)

    if candidate_pairs.shape[0] == 0:
        return cluster_labels.copy(), n_phase1_clusters, dict(members_by_label)

    # ---- 3. vectorised misorientation filter --------------------------------
    rep_quats = quats_per_seed[cluster_rep]                        # (n_clusters, 4)
    qa_t = torch.from_numpy(np.ascontiguousarray(rep_quats[candidate_pairs[:, 0]]))
    qb_t = torch.from_numpy(np.ascontiguousarray(rep_quats[candidate_pairs[:, 1]]))
    misori = misorientation_quat_batch(qa_t, qb_t, space_group)
    if hasattr(misori, "detach"):
        misori = misori.detach().cpu().numpy()
    misori = np.asarray(misori, dtype=np.float64)
    keep_misori = misori < misori_tol_rad
    pairs_after_misori = candidate_pairs[keep_misori]

    if progress:
        print(f"[passA] {pairs_after_misori.shape[0]:,} pairs pass misori<"
              f"{math.degrees(misori_tol_rad):.2f}° "
              f"[{time.time()-t0:.1f}s]", flush=True)

    if pairs_after_misori.shape[0] == 0:
        return cluster_labels.copy(), n_phase1_clusters, dict(members_by_label)

    # ---- 4. Jaccard filter (per-pair set-intersection) ----------------------
    jaccards = _jaccard_per_pair(pairs_after_misori, rep_spot_sets)
    keep_jac = jaccards >= jaccard_tol
    final_edges = pairs_after_misori[keep_jac]

    if progress:
        print(f"[passA] {final_edges.shape[0]:,} edges pass Jaccard ≥ {jaccard_tol} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    if final_edges.shape[0] == 0:
        return cluster_labels.copy(), n_phase1_clusters, dict(members_by_label)

    # ---- 5. connected components on the merge graph -------------------------
    n = n_phase1_clusters
    src = np.concatenate([final_edges[:, 0], np.arange(n, dtype=np.int64)])
    dst = np.concatenate([final_edges[:, 1], np.arange(n, dtype=np.int64)])
    ones = np.ones(src.size, dtype=np.int8)
    A = coo_matrix((ones, (src, dst)), shape=(n, n))
    n_super, super_label = connected_components(A, directed=False, return_labels=True)
    super_label = super_label.astype(np.int64)

    if progress:
        print(f"[passA] CC: {n_phase1_clusters:,} → {n_super:,} super-clusters "
              f"[{time.time()-t0:.1f}s]", flush=True)

    # ---- 6. relabel cluster.labels and members_by_label ---------------------
    new_labels = np.full_like(cluster_labels, -1, dtype=np.int64)
    alive = cluster_labels >= 0
    new_labels[alive] = super_label[cluster_labels[alive]]

    # Compact super-labels to 0..n_super-1 in their natural order (already 0..n_super-1).
    new_members_by_label: Dict[int, np.ndarray] = {}
    sort_order = np.argsort(new_labels, kind="stable")
    sorted_labels = new_labels[sort_order]
    first_alive = int(np.searchsorted(sorted_labels, 0, side="left"))
    sorted_labels = sorted_labels[first_alive:]
    sort_order = sort_order[first_alive:]
    if sorted_labels.size > 0:
        label_changes = np.flatnonzero(np.diff(sorted_labels) != 0) + 1
        label_breaks = np.concatenate([[0], label_changes, [sorted_labels.size]])
        for k in range(label_breaks.size - 1):
            lo, hi = int(label_breaks[k]), int(label_breaks[k + 1])
            lab = int(sorted_labels[lo])
            new_members_by_label[lab] = sort_order[lo:hi]

    return new_labels, int(n_super), new_members_by_label
