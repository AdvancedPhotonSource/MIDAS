"""
process_hdf.py — Tomographic reconstruction from an APS-style HDF5 file.

Default behaviour is unchanged from earlier versions:

    python process_hdf.py -dataFN data.h5 -nCPUs 20

reads ``/exchange/data, /exchange/dark, /exchange/bright`` and the
``analysis_parameters/{CropXL,CropXR,CropZL,CropZR,shift}`` metadata, writes a
raw binary alongside the HDF5 file (if not already present), generates
``mt_par.txt``, and calls ``MIDAS_TOMO``.

New options
-----------
``--tuneCleanup [grid.txt]``
    Run a Vo stripe-removal parameter sweep on a thin slab before the full
    reconstruction. Auto-picks the best config and uses it in the full run.
    With no argument, uses the built-in detector-aware grid; otherwise
    reads ``grid.txt`` (one ``snr la sm`` per line, ``#`` comments allowed).

``--cleanup SNR LA SM``
    Skip tuning and force a single cleanup config.

``--noCleanup``
    Explicitly disable stripe removal.

``--shifts START END STEP``
    Override the single HDF5 ``shift`` with a sweep range.

``--tuningSlices N``
    Number of slices to use during cleanup tuning (default 4).
"""

import argparse
import os
import subprocess
import sys

import cv2
import h5py
import numpy as np


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


def _find_tomo_exe():
    try:
        utils_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'utils')
        if utils_dir not in sys.path:
            sys.path.append(utils_dir)
        import midas_config
        return os.path.join(midas_config.MIDAS_TOMO_BIN_DIR, 'MIDAS_TOMO')
    except ImportError:
        return os.path.expanduser('~/opt/MIDAS/TOMO/bin/MIDAS_TOMO')


def _parse_args():
    p = MyParser(description='Tomo recon driver for APS HDF5 input.',
                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-dataFN', type=str, required=True, help='HDF5 file path.')
    p.add_argument('-nCPUs', type=int, required=True,
                   help='Number of OMP threads passed to MIDAS_TOMO.')
    p.add_argument('--shifts', type=float, nargs=3,
                   metavar=('START', 'END', 'STEP'), default=None,
                   help='Sweep rotation-axis shift instead of using the '
                        'single HDF5 shift.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--tuneCleanup', nargs='?', const='__default__', default=None,
                   metavar='GRID.TXT',
                   help='Auto-tune Vo stripe-removal parameters on a thin '
                        'slab before the full recon. With no value, uses '
                        'the built-in detector-aware grid.')
    g.add_argument('--cleanup', type=float, nargs=3,
                   metavar=('SNR', 'LA', 'SM'), default=None,
                   help='Skip tuning; use this fixed (snr, la, sm) config.')
    g.add_argument('--noCleanup', action='store_true',
                   help='Explicitly disable stripe removal.')
    p.add_argument('--tuningSlices', type=int, default=4,
                   help='Number of mid-stack slices used during tuning.')
    return p.parse_known_args()[0]


def _generate_raw(dataFN, hf, dxL, dxR, dzL, dzR, rot, M):
    """Generate ``<dataFN>.raw`` with dark + 2 whites + uint16 projections
    if it does not already exist. Returns (dark_cropped, dark_shape)."""
    dark = hf['exchange/dark'][dzL:-dzR, dxL:-dxR].astype(np.float32)
    bright = hf['exchange/bright'][:, dzL:-dzR, dxL:-dxR].astype(np.float32)
    raw_path = f'{dataFN}.raw'
    if os.path.exists(raw_path):
        print(f'{raw_path} exists — skipping raw regeneration.')
        return dark, bright
    print(f'Generating {raw_path} ...')
    data = hf['exchange/data']
    nFrames = data.shape[0]
    nX, nY = dark.shape
    data_cropped = data[:, dzL:-dzR, dxL:-dxR].astype(np.uint16)
    if rot != 0:
        data_cropped = np.copy(data_cropped)
        dark = np.copy(dark)
        bright = np.copy(bright)
        for frameNr in range(nFrames):
            data_cropped[frameNr] = cv2.warpAffine(
                data_cropped[frameNr], M, (nY, nX))
        dark = cv2.warpAffine(dark, M, (nY, nX))
        bright[0] = cv2.warpAffine(bright[0], M, (nY, nX))
        bright[1] = cv2.warpAffine(bright[1], M, (nY, nX))
    with open(raw_path, 'wb') as f:
        dark.tofile(f)
        bright.tofile(f)
        data_cropped.tofile(f)
    return dark, bright


def _read_tuning_slab(hf, dzL, dzR, dxL, dxR, rot, M, n_tune):
    """Read dark + whites + an even number of mid-stack slices for tuning."""
    dark = hf['exchange/dark'][dzL:-dzR, dxL:-dxR].astype(np.float32)
    bright = hf['exchange/bright'][:, dzL:-dzR, dxL:-dxR].astype(np.float32)
    data = hf['exchange/data']
    nFrames = data.shape[0]
    cropped_yDim = dark.shape[0]
    mid = cropped_yDim // 2
    half = max(1, n_tune // 2)
    z0 = max(0, mid - half)
    z1 = min(cropped_yDim, z0 + (2 * half))
    if (z1 - z0) % 2 != 0:
        z1 -= 1
    # Read only the slab we need across all frames
    slab = data[:, dzL + z0:dzL + z1, dxL:-dxR].astype(np.uint16)
    dark_slab = dark[z0:z1, :]
    bright_slab = bright[:, z0:z1, :]
    if rot != 0:
        nX, nY = dark_slab.shape
        slab = np.copy(slab)
        for frameNr in range(nFrames):
            slab[frameNr] = cv2.warpAffine(
                slab[frameNr], M, (nY, nX))
        dark_slab = cv2.warpAffine(dark_slab, M, (nY, nX))
        bright_slab[0] = cv2.warpAffine(bright_slab[0], M, (nY, nX))
        bright_slab[1] = cv2.warpAffine(bright_slab[1], M, (nY, nX))
    return dark_slab, bright_slab, slab


def main():
    args = _parse_args()
    dataFN = args.dataFN
    nCPUs = args.nCPUs

    hf = h5py.File(dataFN, 'r')

    rot = 0
    dxL = int(hf['analysis/process/analysis_parameters/CropXL'][0])
    dxR = int(hf['analysis/process/analysis_parameters/CropXR'][0])
    dzL = int(hf['analysis/process/analysis_parameters/CropZL'][0])
    dzR = int(hf['analysis/process/analysis_parameters/CropZR'][0])
    shift = float(hf['analysis/process/analysis_parameters/shift'][0])
    if 'analysis/process/analysis_parameters/RotationAngle' in hf:
        rot = float(hf['analysis/process/analysis_parameters/RotationAngle'][0])

    # Provisional sizes (need dark to know cropped shape)
    dark = hf['exchange/dark'][dzL:-dzR, dxL:-dxR].astype(np.float32)
    nX, nY = dark.shape
    M = cv2.getRotationMatrix2D((nX / 2, nY / 2), rot, scale=1)

    _generate_raw(dataFN, hf, dxL, dxR, dzL, dzR, rot, M)

    nFrames = hf['exchange/data'].shape[0]
    st_ome = float(hf['measurement/process/scan_parameters/start'][0])
    step_ome = float(hf['measurement/process/scan_parameters/step'][0])
    stop_ome = st_ome + step_ome * (nFrames - 1)
    angles = np.linspace(st_ome, stop_ome, num=nFrames)
    np.savetxt('mt_angles.txt', angles.T, fmt='%.6f')

    # ── Decide cleanup config ─────────────────────────────────
    cleanup_lines = []  # extra lines appended to mt_par.txt
    if args.noCleanup:
        print('Cleanup: stripe removal disabled (--noCleanup)')
    elif args.cleanup is not None:
        snr, la, sm = float(args.cleanup[0]), int(args.cleanup[1]), int(args.cleanup[2])
        print(f'Cleanup: fixed config snr={snr} la={la} sm={sm}')
        cleanup_lines = [
            'doStripeRemoval 1\n',
            f'stripeSnr {snr}\n',
            f'stripeLaSize {la}\n',
            f'stripeSmSize {sm}\n',
        ]
    elif args.tuneCleanup is not None:
        # Run the tuning sweep on a thin slab to pick a config
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from midas_tomo_python import run_tomo_cleanup_sweep
        tune_dir = f'{dataFN}_cleanup_tuning'
        os.makedirs(tune_dir, exist_ok=True)
        print(f'Cleanup: tuning into {tune_dir}/ ...')
        dark_slab, bright_slab, slab = _read_tuning_slab(
            hf, dzL, dzR, dxL, dxR, rot, M, args.tuningSlices)
        grid_arg = (None if args.tuneCleanup == '__default__'
                    else args.tuneCleanup)
        # Use the HDF5 shift as the tuning shift (cleanup is largely
        # independent of small shift errors).
        result = run_tomo_cleanup_sweep(
            slab, dark_slab, bright_slab, tune_dir, angles,
            cleanup_configs=grid_arg, shift=shift,
            tuning_slices=list(range(slab.shape[1])),
            numCPUs=nCPUs)
        bc = result['best_config']
        print(f'Cleanup recommended: snr={bc["snr"]:.4f} la={bc["la"]} sm={bc["sm"]}')
        if bc['snr'] > 0:
            cleanup_lines = [
                'doStripeRemoval 1\n',
                f'stripeSnr {bc["snr"]:.4f}\n',
                f'stripeLaSize {bc["la"]}\n',
                f'stripeSmSize {bc["sm"]}\n',
            ]
        else:
            print('Cleanup: tuning chose baseline (no stripe removal).')
    else:
        # Legacy default: no stripe removal (preserves prior behavior)
        pass

    # ── Decide shift range ────────────────────────────────────
    if args.shifts is not None:
        shift_lo, shift_hi, shift_step = args.shifts
    else:
        shift_lo = shift_hi = shift
        shift_step = 1.0

    # ── Write parameter file ──────────────────────────────────
    par_lines = [
        'areSinos 0\n',
        'saveReconSeparate 0\n',
        f'dataFileName {dataFN}.raw\n',
        f'reconFileName recon_{dataFN}\n',
        f'detXdim {dark.shape[1]}\n',
        f'detYdim {dark.shape[0]}\n',
        'thetaFileName mt_angles.txt\n',
        'filter 2\n',
        f'shiftValues {shift_lo} {shift_hi} {shift_step}\n',
        'ringRemovalCoefficient 1.0\n',
        'slicesToProcess -1\n',
    ]
    par_lines.extend(cleanup_lines)
    with open('mt_par.txt', 'w') as f:
        f.writelines(par_lines)

    tomo_exe = _find_tomo_exe()
    subprocess.run([tomo_exe, 'mt_par.txt', str(nCPUs)], check=True,
                   cwd=os.getcwd())


if __name__ == '__main__':
    main()
