"""
MIDAS Tomography Reconstruction Python Library

Three entry points wrapping the MIDAS_TOMO C binary (Gridrec FBP):

  run_tomo()              — from raw projections (dark + 2 whites + images)
  run_tomo_from_sinos()   — from pre-formed sinograms (areSinos=1)
  run_tomo_cleanup_sweep()— tune Vo stripe-removal parameters; picks the
                            best config by visualisation + a numeric ring
                            metric, returns the recommended (snr, la, sm).

Usage examples
--------------
Plain reconstruction::

    from midas_tomo_python import run_tomo
    recon = run_tomo(data, dark, whites, '/tmp/work', thetas, shifts=1.0)

Cleanup tuning before the full sweep::

    from midas_tomo_python import run_tomo_cleanup_sweep, run_tomo
    tune = run_tomo_cleanup_sweep(data, dark, whites, '/tmp/work', thetas)
    cfg = tune['best_config']  # {'snr': 3.0, 'la': 31, 'sm': 11}
    recon = run_tomo(
        data, dark, whites, '/tmp/work', thetas,
        shifts=[-25, 25.1, 0.1],
        doStripeRemoval=1, stripeSnr=cfg['snr'],
        stripeLaSize=cfg['la'], stripeSmSize=cfg['sm'],
    )

From sinograms::

    from midas_tomo_python import run_tomo_from_sinos
    recon = run_tomo_from_sinos(sino_2d, '/tmp/work', thetas)
"""

import collections.abc
import os
import subprocess
import sys
import time
from math import ceil, log2

import numpy as np


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _next_power_of_2(n):
    """Return the smallest power of 2 >= *n*."""
    if n <= 0:
        return 1
    return 1 << int(ceil(log2(n))) if n > 1 else 1


def _find_tomo_exe(useGPU=False):
    """Locate the MIDAS_TOMO binary (or MIDAS_TOMO_GPU if useGPU=True)."""
    binary_name = 'MIDAS_TOMO_GPU' if useGPU else 'MIDAS_TOMO'
    try:
        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'utils')
        if utils_dir not in sys.path:
            sys.path.append(utils_dir)
        import midas_config
        path = os.path.join(midas_config.MIDAS_TOMO_BIN_DIR, binary_name)
        if os.path.isfile(path):
            return path
        # Fall back to CPU binary if GPU not found
        if useGPU:
            print(f'Warning: {binary_name} not found, falling back to MIDAS_TOMO')
            return os.path.join(midas_config.MIDAS_TOMO_BIN_DIR, 'MIDAS_TOMO')
        return path
    except ImportError:
        path = os.path.expanduser(f"~/opt/MIDAS/TOMO/bin/{binary_name}")
        if os.path.isfile(path):
            return path
        if useGPU:
            print(f'Warning: {binary_name} not found, falling back to MIDAS_TOMO')
            return os.path.expanduser("~/opt/MIDAS/TOMO/bin/MIDAS_TOMO")
        return path


def _write_thetas(thetas, path):
    """Write one angle per line to *path*."""
    with open(path, 'w') as f:
        for theta in thetas:
            f.write(f'{theta}\n')


def _parse_shift_arg(shifts):
    """Return (shift_string_for_config, nrShifts)."""
    if not isinstance(shifts, collections.abc.Sequence):
        return f'{shifts} {shifts} 1', 1
    nrShifts = round(abs((shifts[1] - shifts[0])) / shifts[2]) + 1
    return f'{shifts[0]} {shifts[1]} {shifts[2]}', nrShifts


def _read_recon(outfnstr, nrShifts, nrSlices, xDimNew):
    """Read the reconstruction binary written by MIDAS_TOMO."""
    outfn = (f'{outfnstr}_NrShifts_{str(nrShifts).zfill(3)}'
             f'_NrSlices_{str(nrSlices).zfill(5)}'
             f'_XDim_{str(xDimNew).zfill(6)}'
             f'_YDim_{str(xDimNew).zfill(6)}_float32.bin')
    recon = np.fromfile(outfn, dtype=np.float32,
                        count=nrSlices * nrShifts * xDimNew * xDimNew)
    return recon.reshape((nrShifts, nrSlices, xDimNew, xDimNew)), outfn


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def run_tomo(data, dark, whites, workingdir, thetas,
             shifts=0.0, filterNr=2, doLog=1, extraPad=0,
             autoCentering=1, numCPUs=40, doCleanup=1, ringRemoval=0,
             doStripeRemoval=0, stripeSnr=3.0, stripeLaSize=61,
             stripeSmSize=21, useGPU=False, fftwBridge=False):
    """Reconstruct from raw projection data.

    Parameters
    ----------
    data : ndarray, shape (nrThetas+2, nrSlices, xDim)
        Raw projections.  The first two frames are tilt-corrected copies
        that the C code skips (nrThetas -= 2 internally).
    dark : ndarray, shape (nrSlices, xDim) or (xDim,)
        Dark-field image.
    whites : ndarray, shape (2, nrSlices, xDim) or (2, xDim)
        Two white-field images.
    workingdir : str
        Directory for temporary files and output.
    thetas : 1D array
        Rotation angles in degrees.
    shifts : float or [start, end, interval]
        Shift values for rotation-axis centering.
    filterNr : int
        0=none, 1=Shepp-Logan, 2=Hann (default), 3=Hamming, 4=Ramp.
    doLog : int
        1 to take -log of transmission, 0 to use intensities directly.
    extraPad : int
        0=half padding, 1=one-half padding.
    autoCentering : int
        1 to auto-center rotation axis.
    numCPUs : int
        Number of parallel threads.
    doCleanup : int
        1 to remove temporary files after reconstruction.
    ringRemoval : float
        Ring-removal coefficient (0 to disable).
    doStripeRemoval : int
        1 to enable Vo et al. 2018 stripe removal, 0 to disable.
    stripeSnr : float
        SNR threshold for stripe detection (default 3.0).
    stripeLaSize : int
        Median filter window for large stripes (default 61, must be odd).
    stripeSmSize : int
        Median filter window for small stripes (default 21, must be odd).
    useGPU : bool
        If True, use GPU-accelerated reconstruction (MIDAS_TOMO_GPU binary).
    fftwBridge : bool
        If True (and useGPU=True), use CPU FFTW for FFTs to ensure
        byte-identical output to the CPU path (slower).

    Returns
    -------
    recon : ndarray, shape (nrShifts, nrSlices, xDimNew, xDimNew)
        Reconstructed slices.  xDimNew is the next power of 2 >= xDim
        (doubled if extraPad=1).
    """
    start_time = time.time()
    os.makedirs(workingdir, exist_ok=True)

    nrThetas, nrSlices, xDim = data.shape
    data = data.astype(np.float32)

    # --- Pad odd slices to even (MIDAS_TOMO requirement) ---
    originalNSlices = nrSlices
    if nrSlices % 2 != 0:
        data = np.concatenate([data, data[:, -1:, :]], axis=1)
        nrSlices = data.shape[1]

    # Write binary: dark, whites, data(uint16)
    infn = os.path.join(workingdir, 'input.bin')
    with open(infn, 'wb') as f:
        dark.astype(np.float32).tofile(f)
        whites.astype(np.float32).tofile(f)
        data.astype(np.uint16).tofile(f)

    # The C code subtracts 2 from nrThetas (dark+whites preamble)
    nrThetas -= 2

    outfnstr = os.path.join(workingdir, 'output')
    xDimNew = _next_power_of_2(xDim)
    if extraPad == 1:
        xDimNew *= 2

    # Write thetas
    thetasFN = os.path.join(workingdir, 'midastomo_thetas.txt')
    _write_thetas(thetas, thetasFN)

    # Write config
    shift_str, nrShifts = _parse_shift_arg(shifts)
    configFN = os.path.join(workingdir, 'midastomo.par')
    with open(configFN, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 0\n')
        f.write(f'detXdim {xDim}\n')
        f.write(f'detYdim {nrSlices}\n')
        f.write(f'thetaFileName {thetasFN}\n')
        f.write(f'shiftValues {shift_str}\n')
        f.write(f'filter {filterNr}\n')
        f.write(f'ringRemovalCoefficient {ringRemoval}\n')
        f.write(f'doLog {doLog}\n')
        f.write('slicesToProcess -1\n')
        f.write(f'ExtraPad {extraPad}\n')
        f.write(f'AutoCentering {autoCentering}\n')
        if doStripeRemoval:
            f.write(f'doStripeRemoval 1\n')
            f.write(f'stripeSnr {stripeSnr}\n')
            f.write(f'stripeLaSize {stripeLaSize}\n')
            f.write(f'stripeSmSize {stripeSmSize}\n')

    print(f'Time elapsed in preprocessing: {time.time() - start_time:.3f}s.')

    # Run MIDAS_TOMO
    tomo_exe = _find_tomo_exe(useGPU=useGPU)
    cmd = [tomo_exe, configFN, str(numCPUs)]
    if useGPU:
        cmd.append('--gpu')
        if fftwBridge:
            cmd.append('--fftw-bridge')
    subprocess.run(cmd, check=True)

    # Read result
    start_time = time.time()
    recon, outfn = _read_recon(outfnstr, nrShifts, nrSlices, xDimNew)

    # Truncate back to original slice count
    recon = recon[:, :originalNSlices, :, :]

    if doCleanup:
        for fn in [outfn, configFN, thetasFN, infn,
                   os.path.join(workingdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'),
                   os.path.join(workingdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt')]:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    print(f'Time elapsed in postprocessing: {time.time() - start_time:.3f}s.')
    return recon


# ──────────────────────────────────────────────────────────────
# Cleanup-parameter sweep
# ──────────────────────────────────────────────────────────────

def _odd(n):
    """Round *n* up to the next odd positive integer (min 3)."""
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


def default_cleanup_grid(detector_xdim):
    """Return the built-in cleanup grid scaled to the detector width.

    Four configs covering common cases:
      0. baseline (no cleanup) — anchors the visual / metric comparison
      1. moderate (snr=3.0, la=w/4, sm=w/12)
      2. broader  (snr=3.0, la=w/3, sm=w/9)
      3. aggressive SNR (snr=1.5, la=w/4, sm=w/12)

    For a 128-px detector this gives la∈{31, 41}, sm∈{11, 15}; for a 2048-px
    detector la∈{511, 681}, sm∈{171, 227}. Empirically these bracket the
    sensible regime for Vo stripe removal.
    """
    w = int(detector_xdim)
    la_mid = _odd(w / 4)
    la_big = _odd(w / 3)
    sm_mid = _odd(w / 12)
    sm_big = _odd(w / 9)
    return [
        {'snr': 0.0, 'la': 0, 'sm': 0},
        {'snr': 3.0, 'la': la_mid, 'sm': sm_mid},
        {'snr': 3.0, 'la': la_big, 'sm': sm_big},
        {'snr': 1.5, 'la': la_mid, 'sm': sm_mid},
    ]


def ring_metric(img):
    """Stdev of the first-difference of an azimuthal-mean radial profile.

    Real sample features make the radial profile vary smoothly; concentric
    ring artefacts create sharp spikes at fixed radii, raising the stdev of
    the radial first-difference. Lower is better.
    """
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.indices((ny, nx))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = min(cy, cx) - 2
    profile = np.array(
        [img[(r >= b) & (r < b + 1)].mean() for b in range(rmax + 1)]
    )
    return float(np.std(np.diff(profile)))


def _load_cleanup_grid_file(path):
    """Parse a cleanup grid file: one ``snr la sm`` per line, ``#`` comments."""
    configs = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            configs.append({
                'snr': float(parts[0]),
                'la': int(parts[1]),
                'sm': int(parts[2]),
            })
    if not configs:
        raise ValueError(f'No valid configs in cleanup grid file {path}')
    return configs


def run_tomo_cleanup_sweep(data, dark, whites, workingdir, thetas,
                           cleanup_configs=None, shift=0.0,
                           tuning_slices=None, filterNr=2, doLog=1,
                           extraPad=0, autoCentering=1, numCPUs=40,
                           doCleanup=1, ringRemoval=0):
    """Sweep over Vo stripe-removal configs and recommend the best one.

    The function calls MIDAS_TOMO once (using the new ``stripeConfigFile``
    sweep mode) to reconstruct a thin slab at a single fixed shift across
    every cleanup configuration. Output side-effects in *workingdir*:

      cleanup_tuning_montage.png      - slice montage across configs
      cleanup_tuning_scores.txt       - per-config ring metric table
      cleanup_tuning_recommended.txt  - one line: ``snr la sm`` for the pick

    Parameters
    ----------
    data, dark, whites : as for :func:`run_tomo`.
    workingdir : str
        Destination directory; created if missing.
    thetas : 1-D array
        Rotation angles in degrees.
    cleanup_configs : list[dict] | str | None
        - list of ``{'snr': float, 'la': int, 'sm': int}`` dicts, or
        - path to a text file with one ``snr la sm`` per line, or
        - ``None`` to use :func:`default_cleanup_grid` sized to the
          detector. A config with ``snr <= 0`` is treated as the
          no-cleanup baseline.
    shift : float
        Single rotation-axis shift to use during tuning. Pick the value
        you already believe is near optimal; cleanup tuning is largely
        independent of small shift errors.
    tuning_slices : list[int] | None
        Indices of slices to reconstruct for the tuning montage. ``None``
        picks 4 slices spread around the middle of the stack. Fewer
        tuning slices is faster.
    filterNr, doLog, extraPad, autoCentering, ringRemoval :
        Same semantics as :func:`run_tomo`.
    numCPUs : int
        OpenMP thread count passed to ``MIDAS_TOMO``.
    doCleanup : int
        1 to remove the temporary binary / parameter / config files after
        the sweep completes.

    Returns
    -------
    dict with keys:
        ``configs`` (list[dict]),
        ``recons`` (ndarray of shape
            ``(n_cfg, n_tuning_slices, xDimNew, xDimNew)``),
        ``ring_metric`` (ndarray of shape ``(n_cfg,)``; mean over tuning
            slices, lower is better),
        ``best_idx`` (int — winning index, baseline excluded if it is not
            the only good option),
        ``best_config`` (dict).
    """
    os.makedirs(workingdir, exist_ok=True)

    # Resolve configs
    if cleanup_configs is None:
        cleanup_configs = default_cleanup_grid(data.shape[2])
    elif isinstance(cleanup_configs, str):
        cleanup_configs = _load_cleanup_grid_file(cleanup_configs)
    if len(cleanup_configs) < 2:
        raise ValueError(
            'run_tomo_cleanup_sweep requires at least 2 configurations; '
            'for a single fixed config use run_tomo() directly.'
        )

    nrThetas, nrSlices, xDim = data.shape
    # Pick tuning slices
    if tuning_slices is None:
        mid = nrSlices // 2
        tuning_slices = sorted(
            {max(0, min(nrSlices - 1, mid + d)) for d in (-3, -1, 1, 3)}
        )
    tuning_slices = sorted(int(s) for s in tuning_slices)
    if len(tuning_slices) % 2 != 0:
        # MIDAS_TOMO needs even slices in single-shift sweep
        tuning_slices = tuning_slices + [tuning_slices[-1]]

    # Write raw binary (same layout as run_tomo)
    infn = os.path.join(workingdir, 'cleanup_tuning_input.bin')
    with open(infn, 'wb') as f:
        dark.astype(np.float32).tofile(f)
        whites.astype(np.float32).tofile(f)
        data.astype(np.uint16).tofile(f)

    thetas_fn = os.path.join(workingdir, 'cleanup_tuning_thetas.txt')
    _write_thetas(thetas, thetas_fn)

    grid_fn = os.path.join(workingdir, 'cleanup_tuning_grid.txt')
    with open(grid_fn, 'w') as f:
        f.write('# snr  la_size  sm_size  (snr<=0 means baseline)\n')
        for c in cleanup_configs:
            f.write(f'{c["snr"]:.4f}  {int(c["la"])}  {int(c["sm"])}\n')

    slices_fn = os.path.join(workingdir, 'cleanup_tuning_slices.txt')
    with open(slices_fn, 'w') as f:
        for s in tuning_slices:
            f.write(f'{s}\n')

    outfnstr = os.path.join(workingdir, 'cleanup_tuning_output')
    xDimNew = _next_power_of_2(xDim)
    if extraPad == 1:
        xDimNew *= 2

    par_fn = os.path.join(workingdir, 'cleanup_tuning.par')
    # The C engine's multi-shift inner loop pairs shifts. For tuning we want
    # one effective shift, so we ask for two near-identical shifts and discard
    # the second when reading the cube. This costs an extra ~1× per slice but
    # avoids a separate single-shift code path.
    nrShifts = 2
    shift_lo = float(shift)
    shift_hi = float(shift) + 0.1
    shift_step = 0.1
    with open(par_fn, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 0\n')
        f.write(f'detXdim {xDim}\n')
        # C engine uses det_ydim for "all slices" expansion; pass total here.
        # When a slicesToProcess file is given, n_slices == len(slices_file).
        f.write(f'detYdim {nrSlices}\n')
        f.write(f'thetaFileName {thetas_fn}\n')
        f.write(f'shiftValues {shift_lo} {shift_hi} {shift_step}\n')
        f.write(f'filter {filterNr}\n')
        f.write(f'ringRemovalCoefficient {ringRemoval}\n')
        f.write(f'doLog {doLog}\n')
        f.write(f'slicesToProcess {slices_fn}\n')
        f.write(f'ExtraPad {extraPad}\n')
        f.write(f'AutoCentering {autoCentering}\n')
        f.write('doStripeRemoval 1\n')
        f.write(f'stripeConfigFile {grid_fn}\n')

    t0 = time.time()
    subprocess.run([_find_tomo_exe(), par_fn, str(numCPUs)], check=True)
    print(f'cleanup-sweep MIDAS_TOMO took {time.time() - t0:.2f}s')

    # Read multi-cleanup output cube. nrShifts==2 here (see comment above);
    # we keep only the first shift's results — the second is a near-duplicate
    # forced by the C engine's pair-of-shifts inner loop.
    nCfg = len(cleanup_configs)
    nTuneSlices = len(tuning_slices)
    outfn = (f'{outfnstr}_NrCleanup_{str(nCfg).zfill(3)}'
             f'_NrShifts_{str(nrShifts).zfill(3)}'
             f'_NrSlices_{str(nTuneSlices).zfill(5)}'
             f'_XDim_{str(xDimNew).zfill(6)}'
             f'_YDim_{str(xDimNew).zfill(6)}_float32.bin')
    cube = np.fromfile(
        outfn, dtype=np.float32,
        count=nCfg * nrShifts * nTuneSlices * xDimNew * xDimNew
    ).reshape((nCfg, nrShifts, nTuneSlices, xDimNew, xDimNew))
    cube = cube[:, 0, :, :, :]  # (nCfg, nTuneSlices, X, Y) — first shift only

    # Ring metric: per-config mean over tuning slices
    rm = np.zeros(nCfg, dtype=np.float64)
    for ci in range(nCfg):
        per_slice = [ring_metric(cube[ci, si]) for si in range(nTuneSlices)]
        rm[ci] = float(np.mean(per_slice))

    # Pick the best: lowest ring metric, but if the baseline (snr<=0) is
    # somehow tied with cleaned configs, prefer the cleaned one (rings rarely
    # vanish without any filter — equal score there means none of the configs
    # changed the data, not that cleanup is harmful).
    order = np.argsort(rm)
    best_idx = int(order[0])
    if cleanup_configs[best_idx]['snr'] <= 0 and nCfg > 1:
        # If a cleaned config is within 1% of baseline, prefer the cleaned one
        baseline = rm[best_idx]
        for cand in order[1:]:
            if cleanup_configs[cand]['snr'] > 0 and rm[cand] <= baseline * 1.01:
                best_idx = int(cand)
                break

    # Write scores
    scores_fn = os.path.join(workingdir, 'cleanup_tuning_scores.txt')
    with open(scores_fn, 'w') as f:
        f.write('cleanup_idx\tsnr\tla\tsm\tring_metric\tbest\n')
        for ci, c in enumerate(cleanup_configs):
            mark = 'BEST' if ci == best_idx else ''
            f.write(f'{ci}\t{c["snr"]:.4f}\t{c["la"]}\t{c["sm"]}\t'
                    f'{rm[ci]:.6e}\t{mark}\n')
    print(f'Saved {scores_fn}')

    # Recommended config (one line for easy --cleanup parsing)
    rec_fn = os.path.join(workingdir, 'cleanup_tuning_recommended.txt')
    bc = cleanup_configs[best_idx]
    with open(rec_fn, 'w') as f:
        f.write(f'{bc["snr"]:.4f} {bc["la"]} {bc["sm"]}\n')
    print(f'Saved {rec_fn}: {bc["snr"]:.4f} {bc["la"]} {bc["sm"]}')

    # Montage PNG (slice = middle of tuning slab). We avoid calling
    # matplotlib.use() here — that would mutate the user's global backend
    # (notably breaking '%matplotlib inline' in Jupyter). Instead we render
    # off-screen via the Agg canvas without touching the active backend.
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        mid_slice = nTuneSlices // 2
        cols = min(4, nCfg)
        rows = (nCfg + cols - 1) // cols
        fig = Figure(figsize=(cols * 4, rows * 4))
        FigureCanvasAgg(fig)
        axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
        for ci in range(nCfg):
            sl = cube[ci, mid_slice]
            vmin = float(np.percentile(sl, 2))
            vmax = float(np.percentile(sl, 98))
            axes[ci].imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
            c = cleanup_configs[ci]
            title = (f'#{ci}  '
                     + (f'snr={c["snr"]:.1f} la={c["la"]} sm={c["sm"]}'
                        if c['snr'] > 0 else 'BASELINE')
                     + f'\nring={rm[ci]:.2e}')
            if ci == best_idx:
                title = 'BEST: ' + title
                for sp in axes[ci].spines.values():
                    sp.set_edgecolor('red'); sp.set_linewidth(3)
            axes[ci].set_title(title, fontsize=9)
            axes[ci].axis('off')
        for ax in axes[nCfg:]:
            ax.axis('off')
        fig.tight_layout()
        png = os.path.join(workingdir, 'cleanup_tuning_montage.png')
        fig.savefig(png, dpi=140, bbox_inches='tight')
        print(f'Saved {png}')
    except ImportError:
        print('matplotlib not available — skipped montage PNG')

    if doCleanup:
        for p in [outfn, par_fn, thetas_fn, infn, slices_fn,
                  os.path.join(workingdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'),
                  os.path.join(workingdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt'),
                  outfnstr + '_cleanup_configs.txt']:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    return {
        'configs': cleanup_configs,
        'recons': cube,
        'ring_metric': rm,
        'best_idx': best_idx,
        'best_config': cleanup_configs[best_idx],
        'tuning_slices': tuning_slices,
    }


def run_tomo_from_sinos(sinograms, workingdir, thetas,
                        shifts=0.0, filterNr=2, doLog=0, extraPad=0,
                        autoCentering=1, numCPUs=1, doCleanup=1,
                        ringRemoval=0, doStripeRemoval=0, stripeSnr=3.0,
                        stripeLaSize=61, stripeSmSize=21,
                        useGPU=False, fftwBridge=False):
    """Reconstruct from pre-formed sinograms (areSinos=1 mode).

    Parameters
    ----------
    sinograms : ndarray
        Shape ``(nThetas, detXdim)`` for a single slice, or
        ``(nSlices, nThetas, detXdim)`` for multiple slices.
        Data is converted to float32 internally.
    workingdir : str
        Directory for temporary files and output.
    thetas : 1D array
        Rotation angles in degrees.
    shifts : float or [start, end, interval]
        Shift values for rotation-axis centering.
    filterNr : int
        0=none, 1=Shepp-Logan, 2=Hann (default), 3=Hamming, 4=Ramp.
    doLog : int
        0 (default for sinogram input) to use intensities directly,
        1 to apply -log.
    extraPad : int
        0=half padding, 1=one-half padding.
    autoCentering : int
        1 to auto-center rotation axis.
    numCPUs : int
        Number of parallel threads.
    doCleanup : int
        1 to remove temporary files after reconstruction.
    ringRemoval : float
        Ring-removal coefficient (0 to disable).
    doStripeRemoval : int
        1 to enable Vo et al. 2018 stripe removal, 0 to disable.
    stripeSnr : float
        SNR threshold for stripe detection (default 3.0).
    stripeLaSize : int
        Median filter window for large stripes (default 61, must be odd).
    stripeSmSize : int
        Median filter window for small stripes (default 21, must be odd).
    useGPU : bool
        If True, use GPU-accelerated reconstruction (MIDAS_TOMO_GPU binary).
    fftwBridge : bool
        If True (and useGPU=True), use CPU FFTW for FFTs to ensure
        byte-identical output to the CPU path (slower).

    Returns
    -------
    recon : ndarray, shape (nrShifts, nSlices, xDimNew, xDimNew)
        Reconstructed slices.  xDimNew is the next power of 2 >= detXdim
        (doubled if extraPad=1).
    """
    start_time = time.time()
    os.makedirs(workingdir, exist_ok=True)

    # Normalize to 3D
    sinograms = np.asarray(sinograms, dtype=np.float32)
    if sinograms.ndim == 2:
        sinograms = sinograms[np.newaxis, :, :]  # (1, nThetas, detXdim)

    nSlices, nThetas, detXdim = sinograms.shape

    # --- Pad odd slices to even (MIDAS_TOMO requirement) ---
    originalNSlices = nSlices
    if nSlices % 2 != 0:
        sinograms = np.concatenate([sinograms, sinograms[-1:, :, :]], axis=0)
        nSlices = sinograms.shape[0]

    # Write sinogram binary (flat float32, each slice is nThetas × detXdim)
    infn = os.path.join(workingdir, 'input_sino.bin')
    sinograms.tofile(infn)

    outfnstr = os.path.join(workingdir, 'output')
    xDimNew = _next_power_of_2(detXdim)
    if extraPad == 1:
        xDimNew *= 2

    # Write thetas
    thetasFN = os.path.join(workingdir, 'midastomo_thetas.txt')
    _write_thetas(thetas, thetasFN)

    # Write config
    shift_str, nrShifts = _parse_shift_arg(shifts)
    configFN = os.path.join(workingdir, 'midastomo.par')
    with open(configFN, 'w') as f:
        f.write('saveReconSeparate 0\n')
        f.write(f'dataFileName {infn}\n')
        f.write(f'reconFileName {outfnstr}\n')
        f.write('areSinos 1\n')
        f.write(f'detXdim {detXdim}\n')
        f.write(f'detYdim {nSlices}\n')
        f.write(f'thetaFileName {thetasFN}\n')
        f.write(f'filter {filterNr}\n')
        f.write(f'shiftValues {shift_str}\n')
        f.write(f'ringRemovalCoefficient {ringRemoval}\n')
        f.write(f'doLog {doLog}\n')
        f.write('slicesToProcess -1\n')
        f.write(f'ExtraPad {extraPad}\n')
        f.write(f'AutoCentering {autoCentering}\n')
        if doStripeRemoval:
            f.write(f'doStripeRemoval 1\n')
            f.write(f'stripeSnr {stripeSnr}\n')
            f.write(f'stripeLaSize {stripeLaSize}\n')
            f.write(f'stripeSmSize {stripeSmSize}\n')

    print(f'Time elapsed in preprocessing: {time.time() - start_time:.3f}s.')

    # Run MIDAS_TOMO
    tomo_exe = _find_tomo_exe(useGPU=useGPU)
    cmd = [tomo_exe, configFN, str(numCPUs)]
    if useGPU:
        cmd.append('--gpu')
        if fftwBridge:
            cmd.append('--fftw-bridge')
    subprocess.run(cmd, check=True)

    # Read result
    start_time = time.time()
    recon, outfn = _read_recon(outfnstr, nrShifts, nSlices, xDimNew)

    # Truncate back to original slice count
    recon = recon[:, :originalNSlices, :, :]

    if doCleanup:
        for fn in [outfn, configFN, thetasFN, infn,
                   os.path.join(workingdir, f'fftwf_wisdom_1d_{2 * xDimNew}.txt'),
                   os.path.join(workingdir, f'fftwf_wisdom_2d_{2 * xDimNew}.txt')]:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    print(f'Time elapsed in postprocessing: {time.time() - start_time:.3f}s.')
    return recon
