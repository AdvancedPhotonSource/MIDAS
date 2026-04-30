#!/usr/bin/env python
"""Confidence-margin-gated spatial regularization of a pf-HEDM grain-ID map.

For each voxel V we already have:
- A current grain assignment g_old (e.g. argmax of bayesian posterior).
- A per-grain orientation confidence orient[g, V] (the highest indexer
  candidate completeness whose OM is within max_ang of grain g).

Algorithm
---------
For each voxel V with a current assignment:
1. Find top-1 grain g1 (highest orient) and top-2 grain g2.
2. margin(V) = orient[g1, V] - orient[g2, V].
3. If margin >= tau: voxel has unique high-confidence orientation evidence
   for g1 -- keep g_old (do nothing).
4. Else (margin < tau, voxel is "ambiguous"):
   - Candidate set C(V) = {g : orient[g, V] >= orient[g1, V] - tau}.
   - Look at the 8-conn neighborhood (radius r) and count current
     assignments. Vote winner = grain in C(V) with most neighbors.
   - Tie-break by orient[g, V].
   - If winner != g_old, swap.

We never invent assignments for voxels that the indexer didn't index, and
we never override voxels whose top orientation strictly dominates -- the
only edits are at confidence ties, where the indexer alone cannot decide
and spatial neighborhood is the legitimate tiebreaker.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from calcMiso import GetMisOrientationAngleOMBatch  # noqa: E402


def read_indexbest(consol_path):
    with open(consol_path, 'rb') as f:
        nVox = np.frombuffer(f.read(4), dtype=np.int32)[0]
        nSolArr = np.frombuffer(f.read(4 * nVox), dtype=np.int32)
        offArr = np.frombuffer(f.read(8 * nVox), dtype=np.int64)
        header = 4 + 4 * nVox + 8 * nVox
        allVals = np.frombuffer(f.read(), dtype=np.double)
    return nVox, nSolArr, offArr, header, allVals


def compute_orient_score(topdir, sgnum, nScans, nGrs,
                        max_ang_deg=1.0, min_conf=0.0):
    """Per-grain per-voxel orient_score, shape (nGrs, nScans, nScans)."""
    nVox, nSolArr, offArr, header, allVals = read_indexbest(
        os.path.join(topdir, 'Output', 'IndexBest_all.bin'))
    n_cols = 16

    cand_OMs, cand_confs, cand_voxels = [], [], []
    for v in range(nVox):
        if nSolArr[v] == 0:
            continue
        dataOff = int((offArr[v] - header) // 8)
        sols = allVals[dataOff:dataOff + nSolArr[v] * n_cols].reshape(
            nSolArr[v], n_cols)
        confs = sols[:, 15] / np.maximum(sols[:, 14], 1)
        keep = confs >= min_conf
        if not keep.any():
            continue
        for ci in np.where(keep)[0]:
            cand_OMs.append(sols[ci, 2:11])
            cand_confs.append(confs[ci])
            cand_voxels.append(v)
    if not cand_OMs:
        return np.zeros((nGrs, nScans, nScans), dtype=np.float32)

    cand_OMs = np.asarray(cand_OMs)
    cand_confs = np.asarray(cand_confs)
    cand_voxels = np.asarray(cand_voxels)
    grain_OMs = np.genfromtxt(
        os.path.join(topdir, 'UniqueOrientations.csv'), delimiter=' ')[:, 5:14]
    max_ang_rad = np.deg2rad(max_ang_deg)

    orient = np.zeros((nGrs, nScans * nScans), dtype=np.float32)
    for g in range(nGrs):
        angs = GetMisOrientationAngleOMBatch(
            cand_OMs, np.tile(grain_OMs[g], (len(cand_OMs), 1)), sgnum).flatten()
        ok = angs < max_ang_rad
        if not ok.any():
            continue
        for cand_idx in np.where(ok)[0]:
            v = cand_voxels[cand_idx]
            c = cand_confs[cand_idx]
            if c > orient[g, v]:
                orient[g, v] = c
    return orient.reshape(nGrs, nScans, nScans)


def regularize(grid_in, orient, tau, radius=1, max_passes=4):
    """Confidence-margin-gated neighborhood majority vote.

    Returns (new_grid, margin, ambiguous_mask, n_changed_per_pass).
    """
    nGrs, H, W = orient.shape
    grid = grid_in.copy().astype(int)

    sorted_idx = np.argsort(orient, axis=0)[::-1]                  # (nGrs, H, W)
    sorted_conf = np.take_along_axis(orient, sorted_idx, axis=0)
    c1 = sorted_conf[0]
    c2 = sorted_conf[1] if nGrs > 1 else np.zeros_like(c1)
    margin = c1 - c2

    # Only revisit voxels that already have an assignment AND are ambiguous.
    eligible = (margin < tau) & (grid_in >= 0) & (c1 > 0)

    n_changed_history = []
    for _ in range(max_passes):
        new_grid = grid.copy()
        n_changed = 0
        rows, cols = np.where(eligible)
        for r, c in zip(rows, cols):
            top_c = c1[r, c]
            cand_mask = orient[:, r, c] >= (top_c - tau)
            cand = np.where(cand_mask)[0]
            if len(cand) <= 1:
                continue
            r0, r1 = max(0, r - radius), min(H, r + radius + 1)
            c0, c1c = max(0, c - radius), min(W, c + radius + 1)
            patch = grid[r0:r1, c0:c1c].flatten()
            patch = patch[patch >= 0]
            if len(patch) == 0:
                continue
            counts = np.zeros(nGrs, dtype=int)
            for g in cand:
                counts[g] = int((patch == g).sum())
            best_count = counts.max()
            if best_count == 0:
                continue
            tied = np.where(counts == best_count)[0]
            if len(tied) == 1:
                winner = int(tied[0])
            else:
                winner = int(tied[np.argmax(orient[tied, r, c])])
            if winner != grid[r, c]:
                new_grid[r, c] = winner
                n_changed += 1
        n_changed_history.append(n_changed)
        if n_changed == 0:
            break
        grid = new_grid
    return grid, margin, eligible, n_changed_history


def _read_param(param_path, key):
    with open(param_path) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == key:
                return parts[1]
    return None


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-topdir', required=True,
                   help='pf-HEDM result dir (contains Output/, Recons/, UniqueOrientations.csv)')
    p.add_argument('-paramFile', required=True,
                   help='pf-HEDM parameter file (SpaceGroup, nScans entries)')
    p.add_argument('-tau', type=float, default=0.05,
                   help='Confidence-margin threshold for ambiguity (default 0.05)')
    p.add_argument('-radius', type=int, default=1, help='Neighborhood radius (1=3x3, 2=5x5)')
    p.add_argument('-passes', type=int, default=6, help='Max regularization passes')
    p.add_argument('-maxAng', type=float, default=1.0,
                   help='Max angular distance (deg) for orient_score grain matching')
    p.add_argument('-outTif', default=None)
    p.add_argument('-outPng', default=None)
    args = p.parse_args()

    sgnum = int(_read_param(args.paramFile, 'SpaceGroup'))
    nScans = int(_read_param(args.paramFile, 'nScans'))
    print(f'sgnum={sgnum}  nScans={nScans}  tau={args.tau}  radius={args.radius}')

    grain_OMs = np.genfromtxt(
        os.path.join(args.topdir, 'UniqueOrientations.csv'), delimiter=' ')
    if grain_OMs.ndim == 1:
        grain_OMs = grain_OMs[None, :]
    nGrs = grain_OMs.shape[0]
    print(f'nGrs={nGrs}')

    orient = compute_orient_score(args.topdir, sgnum, nScans, nGrs,
                                  max_ang_deg=args.maxAng, min_conf=0.0)
    print(f'orient shape={orient.shape}  max={orient.max():.3f}  '
          f'>0 voxels per grain: {[int((orient[g]>0).sum()) for g in range(nGrs)]}')

    grid_path = os.path.join(args.topdir, 'Recons/Full_recon_max_project_grID.tif')
    grid = np.array(Image.open(grid_path)).astype(int)
    counts_before = [int((grid == g).sum()) for g in range(nGrs)]
    unassigned_before = int((grid < 0).sum())
    print(f'before: per-grain {counts_before}  unassigned {unassigned_before}')

    new_grid, margin, ambig, hist = regularize(
        grid, orient, args.tau, args.radius, args.passes)
    counts_after = [int((new_grid == g).sum()) for g in range(nGrs)]
    n_changed = int((new_grid != grid).sum())
    print(f'after : per-grain {counts_after}  changed {n_changed}  passes {hist}')
    print(f'ambiguous voxels (margin<tau, assigned): {int(ambig.sum())}')

    # Connectivity
    from scipy.ndimage import label
    print('Connectivity (largest component / total per grain):')
    for g in range(nGrs):
        m_b = (grid == g)
        m_a = (new_grid == g)
        if not m_a.any():
            continue
        _, n_b = label(m_b, structure=np.ones((3, 3), int))
        lab_a, n_a = label(m_a, structure=np.ones((3, 3), int))
        sz_a = np.bincount(lab_a.flatten())[1:]
        big = sz_a.max() if len(sz_a) else 0
        print(f'  G{g}: {m_b.sum()}->{m_a.sum()}  comps {n_b}->{n_a}  '
              f'largest {big} ({100*big/max(m_a.sum(),1):.0f}%)')

    if args.outTif:
        Image.fromarray(new_grid.astype(np.int32)).save(args.outTif)
        print(f'saved {args.outTif}')
    if args.outPng:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        palette = ['#202020', '#e41a1c', '#377eb8', '#4daf4a',
                   '#984ea3', '#ff7f00', '#ffff33', '#a65628']
        cmap = ListedColormap(palette[:nGrs + 1])
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(grid, cmap=cmap, vmin=-1, vmax=nGrs - 1, interpolation='nearest')
        axs[0].set_title(f'before  (changed={n_changed})')
        axs[1].imshow(new_grid, cmap=cmap, vmin=-1, vmax=nGrs - 1, interpolation='nearest')
        axs[1].set_title(f'after  tau={args.tau}  r={args.radius}')
        im = axs[2].imshow(margin, cmap='RdYlGn', vmin=0, vmax=0.3, interpolation='nearest')
        axs[2].contour(ambig.astype(int), levels=[0.5], colors='black', linewidths=0.4)
        axs[2].set_title('orient margin (top1-top2); black=ambiguous')
        plt.colorbar(im, ax=axs[2], shrink=0.8)
        plt.tight_layout()
        plt.savefig(args.outPng, dpi=110, bbox_inches='tight')
        print(f'saved {args.outPng}')
