#!/usr/bin/env python
"""Independent cross-validation of pf-HEDM per-voxel grain assignments.

For each voxel V (assigned to grain g by the indexer + bayesian fusion):
1. Take the indexer's MATCHED spot IDs for V (from IndexBest_IDs_all.bin).
2. For each matched spot, compute the per-spot scan-position residual:
       s_expected_V = x_V * sin(omega) + y_V * cos(omega)   (indexer convention)
       residual_i   = | s_expected_V - s_observed |
   where s_observed = positions[spotScanNum].
3. Score voxel by the FRACTION of its matched spots whose residual is within
   a TIGHT cross-validation tolerance (e.g. 0.2 µm) — much tighter than the
   indexer's accept tolerance (0.5 µm = BeamSize/2).

A voxel that the indexer assigned via spurious matches will have most of its
spots near the loose tolerance edge (residual ~0.4-0.5 µm) and FEW within
the strict tolerance. A real voxel will have most spots within the strict
tolerance (residual ~0-0.2 µm).

This is INDEPENDENT of the indexer's accept criterion: we're not redoing
the same check, we're partitioning the indexer-accepted spots into "strong"
vs "borderline" matches based on residual quality.

Voxels below `frac_strong_min` get unassigned; the grain map is otherwise
preserved (no re-assignment, no fabrication).
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


# Spots.bin columns: [x, y, omega, intensity, spotID, ringNum, eta, theta,
#                     dspacing, scanNum]  (10 doubles per spot)
SPOTS_COLS = 10


def read_spots_bin(path):
    """Memory-map Spots.bin. Returns (n_spots, omega_arr, spotID_arr,
    scanNum_arr) — all aligned by spot index in the file."""
    raw = np.memmap(path, dtype=np.float64, mode='r').reshape(-1, SPOTS_COLS)
    return raw.shape[0], raw[:, 2].copy(), raw[:, 4].astype(np.int64).copy(), \
           raw[:, 9].astype(np.int64).copy()


def read_indexbest_ids(path):
    """IndexBest_IDs_all.bin layout (mirrors IndexBest_all.bin):
        int32 nVox
        int32[nVox] nIDsPerVox  -- total matched IDs across all solutions
        int64[nVox] offByte     -- byte offset of this voxel's first ID
        int32[remaining] all matched IDs concatenated.
    """
    with open(path, 'rb') as f:
        nVox = np.frombuffer(f.read(4), dtype=np.int32)[0]
        nArr = np.frombuffer(f.read(4 * nVox), dtype=np.int32).copy()
        offArr = np.frombuffer(f.read(8 * nVox), dtype=np.int64).copy()
        header = 4 + 4 * nVox + 8 * nVox
        allIDs = np.frombuffer(f.read(), dtype=np.int32).copy()
    return nVox, nArr, offArr, header, allIDs


def read_indexbest_vals(path):
    with open(path, 'rb') as f:
        nVox = np.frombuffer(f.read(4), dtype=np.int32)[0]
        nSolArr = np.frombuffer(f.read(4 * nVox), dtype=np.int32).copy()
        offArr = np.frombuffer(f.read(8 * nVox), dtype=np.int64).copy()
        header = 4 + 4 * nVox + 8 * nVox
        allVals = np.frombuffer(f.read(), dtype=np.float64).copy()
    return nVox, nSolArr, offArr, header, allVals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-topdir', required=True)
    p.add_argument('-paramFile', required=True,
                   help='paramFile path (used to read SpaceGroup, nScans, BeamSize)')
    p.add_argument('-tolStrict', type=float, default=0.20,
                   help='Strict cross-validation residual tolerance (µm). Default 0.20.')
    p.add_argument('-fracStrongMin', type=float, default=0.50,
                   help='Minimum fraction of matched spots that must pass the strict '
                        'tolerance. Voxels below this are treated as spurious. Default 0.50.')
    p.add_argument('-outTif', default=None,
                   help='Validated grain ID TIF (voxels failing validation become -1)')
    p.add_argument('-outPng', default=None)
    p.add_argument('-diagCsv', default=None,
                   help='Optional CSV: row, col, gid, nMatched, nStrict, frac_strong, mean_resid')
    args = p.parse_args()

    topdir = args.topdir

    # Read params we need
    nScans = None
    with open(args.paramFile) as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == 'nScans':
                nScans = int(parts[1])
                break
    print(f'nScans={nScans}')

    # Load spot data
    n_spots, omega_arr, spotID_arr, scanNum_arr = read_spots_bin(
        os.path.join(topdir, 'Spots.bin'))
    print(f'Spots.bin: {n_spots} spots')

    # Map spotID -> spot row index. spotIDs are 1-based in MIDAS.
    max_id = int(spotID_arr.max())
    id_to_row = -np.ones(max_id + 2, dtype=np.int64)
    id_to_row[spotID_arr] = np.arange(n_spots, dtype=np.int64)

    # Read positions.csv
    pos = np.loadtxt(os.path.join(topdir, 'positions.csv'))
    pos_sorted = np.sort(pos)
    print(f'positions: {len(pos)} entries (range {pos.min()} to {pos.max()})')

    # Load consolidated files
    nVox_v, nSolArr, offArr_v, header_v, allVals = read_indexbest_vals(
        os.path.join(topdir, 'Output', 'IndexBest_all.bin'))
    nVox_i, nIDsArr, offArr_i, header_i, allIDs = read_indexbest_ids(
        os.path.join(topdir, 'Output', 'IndexBest_IDs_all.bin'))
    assert nVox_v == nVox_i, f'voxel count mismatch {nVox_v} vs {nVox_i}'
    nVox = nVox_v
    print(f'nVoxels in indexer output: {nVox}')

    # Load current grain map
    grid = np.array(Image.open(
        os.path.join(topdir, 'Recons/Full_recon_max_project_grID.tif'))).astype(int)
    print(f'grid shape {grid.shape}')

    # Walk voxels: for each voxel that has a top solution AND is in grid,
    # extract its top-solution matched spot IDs and run the residual check.
    n_vals_cols = 16
    diag = []
    new_grid = grid.copy()

    # Build voxNr -> (row, col) mapping. pf_MIDAS uses voxNr = row*nScans + col,
    # with row/col in [0, nScans). The grid rendering matches gid[row, col].
    n_dropped = 0
    n_dropped_by_grain = [0] * 5

    for v in range(nVox):
        if nSolArr[v] == 0:
            continue
        row = v // nScans
        col = v % nScans
        if row >= grid.shape[0] or col >= grid.shape[1]:
            continue
        g = int(grid[row, col])
        if g < 0:
            continue

        # Top solution is at index 0 in this voxel's solution block
        dataOff = int((offArr_v[v] - header_v) // 8)
        sol = allVals[dataOff:dataOff + n_vals_cols]
        nMatched = int(sol[15])
        if nMatched == 0:
            continue

        # IDs for top solution = first nMatched of this voxel's concatenated list
        idOff = int((offArr_i[v] - header_i) // 4)
        ids_top = allIDs[idOff:idOff + nMatched]

        # Compute V's spatial position. C indexer convention (line 1680):
        #     voxNr = i * nScans + j  (i=row, j=col)
        #     xThis = ypos_sorted[i]   (row -> x)
        #     yThis = ypos_sorted[j]   (col -> y)
        x_V = pos_sorted[row]   # row index    -> x
        y_V = pos_sorted[col]   # column index -> y

        # Look up each spot's omega and scanNum, compute residual
        rows = id_to_row[ids_top]
        valid = rows >= 0
        if not valid.any():
            continue
        omes = omega_arr[rows[valid]]
        scans = scanNum_arr[rows[valid]]
        s_obs = pos[scans]
        ome_rad = np.deg2rad(omes)
        s_exp = x_V * np.sin(ome_rad) + y_V * np.cos(ome_rad)
        residuals = np.abs(s_exp - s_obs)

        nStrict = int((residuals < args.tolStrict).sum())
        nTotal = int(valid.sum())
        frac_strong = nStrict / max(nTotal, 1)
        mean_resid = float(residuals.mean()) if nTotal else float('nan')

        diag.append((row, col, g, nTotal, nStrict, frac_strong, mean_resid))

        if frac_strong < args.fracStrongMin:
            new_grid[row, col] = -1
            n_dropped += 1
            n_dropped_by_grain[g] += 1

    diag = np.array(diag) if diag else np.zeros((0, 7))
    print(f'\nProcessed {len(diag)} voxels with valid top solutions.')
    if len(diag):
        fs = diag[:, 5]
        mr = diag[:, 6]
        print(f'frac_strong distribution: median={np.median(fs):.3f}  '
              f'q10={np.quantile(fs,0.10):.3f}  q25={np.quantile(fs,0.25):.3f}  '
              f'q75={np.quantile(fs,0.75):.3f}')
        print(f'mean residual (µm) distribution: median={np.median(mr):.3f}  '
              f'q10={np.quantile(mr,0.10):.3f}  q90={np.quantile(mr,0.90):.3f}')
        for g in range(5):
            mask = diag[:, 2] == g
            if mask.any():
                print(f'  G{g}: N={int(mask.sum())}  '
                      f'frac_strong median={np.median(diag[mask,5]):.3f}  '
                      f'mean_resid median={np.median(diag[mask,6]):.3f} µm')

    print(f'\nDropped {n_dropped} voxels (per grain: {n_dropped_by_grain})')

    # After-stats
    from scipy.ndimage import label
    print('\nAfter validation:')
    for g in range(5):
        m_b = (grid == g)
        m_a = (new_grid == g)
        if not m_b.any():
            continue
        lab_a, nc_a = label(m_a, structure=np.ones((3, 3), int))
        sz = np.bincount(lab_a.flatten())[1:] if m_a.any() else np.array([])
        big = int(sz.max()) if len(sz) else 0
        print(f'  G{g}: vox {int(m_b.sum())}->{int(m_a.sum())}  '
              f'comps {nc_a}  largest {big} ({100*big/max(int(m_a.sum()),1):.0f}%)')
    print(f'  unassigned: {int((grid<0).sum())}->{int((new_grid<0).sum())}')

    if args.outTif:
        Image.fromarray(new_grid.astype(np.int32)).save(args.outTif)
        print(f'\nsaved {args.outTif}')

    if args.diagCsv and len(diag):
        np.savetxt(args.diagCsv, diag, fmt='%d %d %d %d %d %.4f %.4f',
                   header='row col gid nMatched nStrict frac_strong mean_resid', comments='')
        print(f'saved {args.diagCsv}')

    if args.outPng:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#202020', '#e41a1c', '#377eb8',
                               '#4daf4a', '#984ea3', '#ff7f00'])

        # Build per-voxel frac_strong map for diagnostic panel
        fs_map = np.full(grid.shape, np.nan)
        mr_map = np.full(grid.shape, np.nan)
        for r, c, _, _, _, fs_v, mr_v in diag:
            fs_map[int(r), int(c)] = fs_v
            mr_map[int(r), int(c)] = mr_v

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(grid, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        axs[0].set_title('before')
        axs[1].imshow(new_grid, cmap=cmap, vmin=-1, vmax=4, interpolation='nearest')
        axs[1].set_title(f'after (tolStrict={args.tolStrict}, fracMin={args.fracStrongMin})')
        im2 = axs[2].imshow(fs_map, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        axs[2].set_title('frac_strong (resid<tolStrict)')
        plt.colorbar(im2, ax=axs[2], shrink=0.8)
        im3 = axs[3].imshow(mr_map, cmap='magma_r', vmin=0, vmax=0.5, interpolation='nearest')
        axs[3].set_title('mean residual (µm)')
        plt.colorbar(im3, ax=axs[3], shrink=0.8)
        plt.tight_layout()
        plt.savefig(args.outPng, dpi=110, bbox_inches='tight')
        print(f'saved {args.outPng}')


if __name__ == '__main__':
    main()
