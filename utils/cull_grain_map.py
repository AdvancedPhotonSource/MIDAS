#!/usr/bin/env python
"""Two independent post-processing options for pf-HEDM grain ID maps.

OPTION A (--mode size):
    Drop connected components per grain whose size is below `min_size`.
    Justification: a 1-2 voxel grain inside a 4000-voxel grain is below the
    physical resolution of a 1 µm scan. Defensible "below noise floor" cull,
    not a reassignment.

OPTION B (--mode shape):
    For each voxel V assigned to grain g, check whether the per-grain
    tomographic density `recon[g, V]` (independent shape evidence) is at
    least `shape_floor * P_q` of grain g's density distribution at its own
    voxels (e.g. floor=0.10, q=0.50 means "must be >= 10% of grain g's
    median assigned-voxel density"). Voxels failing the check are unassigned.

Both modes only DROP voxels; neither invents new assignments.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import label


def cull_small_components(grid, n_grains, min_size):
    """Drop connected components < min_size (per grain)."""
    out = grid.copy()
    n_dropped = 0
    n_dropped_by_grain = []
    for g in range(n_grains):
        m = (grid == g)
        if not m.any():
            n_dropped_by_grain.append(0)
            continue
        lab, _ = label(m, structure=np.ones((3, 3), int))
        sizes = np.bincount(lab.flatten())
        for cid in range(1, len(sizes)):
            if sizes[cid] < min_size:
                drop = (lab == cid)
                out[drop] = -1
                n_dropped += int(drop.sum())
        n_dropped_by_grain.append(int(((grid == g) & (out == -1)).sum()))
    return out, n_dropped, n_dropped_by_grain


def cull_by_shape_evidence(grid, recons, shape_floor, q):
    """Drop voxels where recon[g, V] is below shape_floor × P_q of grain g's
    assigned-voxel density distribution. Each grain's threshold is
    independent (small grains compete fairly).
    """
    n_grains = recons.shape[0]
    out = grid.copy()
    n_dropped_by_grain = []
    thresholds = []
    for g in range(n_grains):
        m = (grid == g)
        if not m.any():
            thresholds.append(np.nan)
            n_dropped_by_grain.append(0)
            continue
        densities_assigned = recons[g][m]
        ref = float(np.quantile(densities_assigned, q))
        thr = shape_floor * ref
        thresholds.append(thr)
        weak = m & (recons[g] < thr)
        out[weak] = -1
        n_dropped_by_grain.append(int(weak.sum()))
    return out, sum(n_dropped_by_grain), n_dropped_by_grain, thresholds


def per_grain_stats(grid, n_grains):
    rows = []
    for g in range(n_grains):
        m = (grid == g)
        if not m.any():
            rows.append((g, 0, 0, 0, 0.0))
            continue
        lab, nc = label(m, structure=np.ones((3, 3), int))
        sz = np.bincount(lab.flatten())[1:]
        big = int(sz.max()) if len(sz) else 0
        rows.append((g, int(m.sum()), nc, big, big / max(int(m.sum()), 1)))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-topdir', required=True,
                   help='pf-HEDM result dir (contains Recons/Full_recon_max_project_grID.tif)')
    p.add_argument('-mode', required=True, choices=['size', 'shape'],
                   help='size = drop small connected components; '
                        'shape = drop voxels with weak per-grain density')
    p.add_argument('-minSize', type=int, default=3,
                   help='[size mode] Drop components with fewer voxels than this. Default 3.')
    p.add_argument('-shapeFloor', type=float, default=0.10,
                   help='[shape mode] Density floor as fraction of grain quantile (default 0.10)')
    p.add_argument('-shapeQ', type=float, default=0.50,
                   help='[shape mode] Reference quantile of per-grain density (default 0.50 = median)')
    p.add_argument('-outTif', default=None)
    p.add_argument('-outPng', default=None)
    args = p.parse_args()

    grid_path = os.path.join(args.topdir, 'Recons/Full_recon_max_project_grID.tif')
    grid = np.array(Image.open(grid_path)).astype(int)

    # Detect n_grains by reading Recons/recon_grNr_*.tif
    recon_files = sorted(
        f for f in os.listdir(os.path.join(args.topdir, 'Recons'))
        if f.startswith('recon_grNr_') and f.endswith('.tif'))
    n_grains = len(recon_files)
    print(f'grid {grid.shape}, n_grains={n_grains}')

    print('\nBefore:')
    for g, vox, nc, big, frac in per_grain_stats(grid, n_grains):
        print(f'  G{g}: vox={vox:5d}  comps={nc:3d}  largest={big:5d} ({frac*100:.0f}%)')
    print(f'  unassigned: {(grid<0).sum()}')

    if args.mode == 'size':
        new_grid, n_total, n_by_g = cull_small_components(grid, n_grains, args.minSize)
        title = f'cull_small (min_size={args.minSize})'
        print(f'\nDropped {n_total} voxels (per grain: {n_by_g})')
    else:  # shape
        recons = np.stack([np.array(Image.open(os.path.join(args.topdir, 'Recons', f))).astype(float)
                           for f in recon_files], axis=0)
        new_grid, n_total, n_by_g, thr = cull_by_shape_evidence(
            grid, recons, args.shapeFloor, args.shapeQ)
        thr_str = ', '.join(f'G{g}={thr[g]:.2f}' for g in range(n_grains))
        title = f'cull_shape (floor={args.shapeFloor} × P{int(args.shapeQ*100)}; thresholds: {thr_str})'
        print(f'\nThresholds (shape_floor × P{int(args.shapeQ*100)}): {thr}')
        print(f'Dropped {n_total} voxels (per grain: {n_by_g})')

    print('\nAfter:')
    for g, vox, nc, big, frac in per_grain_stats(new_grid, n_grains):
        print(f'  G{g}: vox={vox:5d}  comps={nc:3d}  largest={big:5d} ({frac*100:.0f}%)')
    print(f'  unassigned: {(new_grid<0).sum()}')

    if args.outTif:
        Image.fromarray(new_grid.astype(np.int32)).save(args.outTif)
        print(f'\nsaved {args.outTif}')

    if args.outPng:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        palette = ['#202020', '#e41a1c', '#377eb8', '#4daf4a',
                   '#984ea3', '#ff7f00', '#ffff33', '#a65628']
        cmap = ListedColormap(palette[:n_grains + 1])
        fig, axs = plt.subplots(1, 2, figsize=(13, 6))
        axs[0].imshow(grid, cmap=cmap, vmin=-1, vmax=n_grains - 1, interpolation='nearest')
        axs[0].set_title('before')
        axs[1].imshow(new_grid, cmap=cmap, vmin=-1, vmax=n_grains - 1, interpolation='nearest')
        axs[1].set_title(title)
        plt.tight_layout()
        plt.savefig(args.outPng, dpi=110, bbox_inches='tight')
        print(f'saved {args.outPng}')
