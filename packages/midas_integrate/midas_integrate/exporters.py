"""Clean CSV export for integrated zarr output (issue #23).

The output of ``IntegratorZarrOMP`` (or the new ``midas-integrate`` pipeline,
once it writes zarr) lives under a small set of well-known dataset names:

- ``REtaMap``          shape (5, nR, nEta)   — R, 2θ, η, binArea, Q
- ``IntegrationResult/FrameNr_<i>``  shape (nR, nEta)  — per-frame cake
- ``OmegaSumFrame/LastFrameNumber_<i>``  shape (nR, nEta) — chunked sum cake
- ``SumFrames``        shape (nR, nEta)      — all-frame summed cake
- ``Omegas``           shape (nFrames,)
- ``InstrumentParameters/<scalar>``           per-run instrument scalars

The exporter pulls only those — ignores ``measurement/`` and ``instrument/``
metadata trees on purpose, because dumping them recursively (as the user's
script in #23 did) produces a pile of one-row CSVs nobody opens twice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import zarr


def _frame_indices(group: zarr.hierarchy.Group, prefix: str) -> List[int]:
    """Return sorted frame indices for keys like ``FrameNr_0``, ``FrameNr_1`` …"""
    out = []
    for k in group.keys():
        if k.startswith(prefix):
            try:
                out.append(int(k[len(prefix):]))
            except ValueError:
                continue
    out.sort()
    return out


def _parse_frame_spec(spec: str, available: Sequence[int]) -> List[int]:
    """Parse a frame selector like ``"all"``, ``"0-99"``, ``"0,5,10"``."""
    if not spec or spec.lower() == 'all':
        return list(available)
    avail = set(available)
    chosen: List[int] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            lo, hi = token.split('-', 1)
            for i in range(int(lo), int(hi) + 1):
                if i in avail:
                    chosen.append(i)
        else:
            i = int(token)
            if i in avail:
                chosen.append(i)
    return chosen


def _retmap_columns(zf: zarr.hierarchy.Group) -> Optional[np.ndarray]:
    """Return REtaMap as a (nR*nEta, 5) array, or None if absent."""
    if 'REtaMap' not in zf:
        return None
    arr = zf['REtaMap'][...]
    if arr.ndim != 3 or arr.shape[0] != 5:
        return None
    n_r, n_eta = arr.shape[1], arr.shape[2]
    flat = arr.reshape(5, n_r * n_eta).T
    return flat


def _r_centers(zf: zarr.hierarchy.Group) -> Optional[np.ndarray]:
    """Mean R per radial bin (R is axis 0 of REtaMap; same value across eta)."""
    if 'REtaMap' not in zf:
        return None
    return np.nanmean(zf['REtaMap'][0, :, :], axis=1)


def _twotheta_centers(zf: zarr.hierarchy.Group) -> Optional[np.ndarray]:
    if 'REtaMap' not in zf:
        return None
    return np.nanmean(zf['REtaMap'][1, :, :], axis=1)


def _q_centers(zf: zarr.hierarchy.Group) -> Optional[np.ndarray]:
    if 'REtaMap' not in zf:
        return None
    return np.nanmean(zf['REtaMap'][4, :, :], axis=1)


def _eta_centers(zf: zarr.hierarchy.Group) -> Optional[np.ndarray]:
    if 'REtaMap' not in zf:
        return None
    return np.nanmean(zf['REtaMap'][2, :, :], axis=0)


def _frame_lineout(cake: np.ndarray) -> np.ndarray:
    """Eta-averaged 1D lineout, matching the C ``IntegratorZarrOMP`` formula
    (``sum/count`` over non-NaN eta bins per R)."""
    return np.nanmean(cake, axis=1)


def export(
    zarr_path: Path,
    out_dir: Path,
    *,
    frames: str = 'all',
    include_cake: bool = False,
    include_summed: bool = True,
    include_metadata: bool = True,
    include_retamap: bool = True,
) -> List[Path]:
    """Export the lineout / cake datasets in ``zarr_path`` as CSV files.

    Returns the list of files written. Skips silently when a requested
    dataset isn't in the zarr (e.g. a run with no peak fitting).
    """
    zarr_path = Path(zarr_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = zarr_path.name
    for sfx in ('.zip', '.zarr'):
        if stem.lower().endswith(sfx):
            stem = stem[: -len(sfx)]
    written: List[Path] = []

    zf = zarr.open(str(zarr_path), mode='r')

    omegas = zf['Omegas'][...] if 'Omegas' in zf else None

    frame_idx: List[int] = []
    if 'IntegrationResult' in zf:
        frame_idx = _frame_indices(zf['IntegrationResult'], 'FrameNr_')
    selected = _parse_frame_spec(frames, frame_idx)

    if include_metadata:
        meta_path = out_dir / f'{stem}_metadata.csv'
        rows = ['frame_idx,omega_deg']
        for i in selected:
            ome = float(omegas[i]) if (omegas is not None and i < len(omegas)) else float('nan')
            rows.append(f'{i},{ome:.8f}')
        if 'InstrumentParameters' in zf:
            ip = zf['InstrumentParameters']
            scalar_lines = ['# Instrument parameters (one value per run):']
            for k in sorted(ip.keys()):
                try:
                    v = ip[k][...]
                    val = v.item() if v.size == 1 else ' '.join(f'{x}' for x in v.flatten())
                    scalar_lines.append(f'# {k} = {val}')
                except Exception:
                    pass
            meta_path.write_text('\n'.join(scalar_lines + [''] + rows) + '\n')
        else:
            meta_path.write_text('\n'.join(rows) + '\n')
        written.append(meta_path)

    if include_retamap and 'REtaMap' in zf:
        flat = _retmap_columns(zf)
        if flat is not None:
            retmap_path = out_dir / f'{stem}_REtaMap.csv'
            header = 'R_pixels,twoTheta_deg,eta_deg,binArea_pixels,Q_invA'
            np.savetxt(retmap_path, flat, delimiter=',', header=header,
                       comments='', fmt='%.8g')
            written.append(retmap_path)

    if selected and 'IntegrationResult' in zf:
        ir = zf['IntegrationResult']
        r_vals = _r_centers(zf)
        tth_vals = _twotheta_centers(zf)
        q_vals = _q_centers(zf)
        first_cake = ir[f'FrameNr_{selected[0]}'][...]
        n_r = first_cake.shape[0]

        if r_vals is None or len(r_vals) != n_r:
            r_vals = np.arange(n_r, dtype=np.float64)
            tth_vals = np.full(n_r, np.nan)
            q_vals = np.full(n_r, np.nan)

        cols = [r_vals, tth_vals, q_vals]
        col_names = ['R_pixels', 'twoTheta_deg', 'Q_invA']
        for i in selected:
            cake = ir[f'FrameNr_{i}'][...]
            cols.append(_frame_lineout(cake))
            col_names.append(f'frame_{i:06d}')

        lineouts_path = out_dir / f'{stem}_lineouts.csv'
        stacked = np.column_stack(cols)
        np.savetxt(lineouts_path, stacked, delimiter=',',
                   header=','.join(col_names), comments='', fmt='%.8g')
        written.append(lineouts_path)

        if include_cake:
            eta_vals = _eta_centers(zf)
            n_eta = first_cake.shape[1]
            if eta_vals is None or len(eta_vals) != n_eta:
                eta_vals = np.arange(n_eta, dtype=np.float64)
            cake_dir = out_dir / f'{stem}_cakes'
            cake_dir.mkdir(exist_ok=True)
            eta_header = 'R_pixels,' + ','.join(f'eta_{e:+.4f}' for e in eta_vals)
            for i in selected:
                cake = ir[f'FrameNr_{i}'][...]
                stacked = np.column_stack([r_vals, cake])
                p = cake_dir / f'cake_frame_{i:06d}.csv'
                np.savetxt(p, stacked, delimiter=',', header=eta_header,
                           comments='', fmt='%.8g')
                written.append(p)

    if include_summed and 'SumFrames' in zf:
        sum_cake = zf['SumFrames'][...]
        r_vals = _r_centers(zf)
        if r_vals is None or len(r_vals) != sum_cake.shape[0]:
            r_vals = np.arange(sum_cake.shape[0], dtype=np.float64)
        sum_path = out_dir / f'{stem}_sum_lineout.csv'
        stacked = np.column_stack([r_vals, _frame_lineout(sum_cake)])
        np.savetxt(sum_path, stacked, delimiter=',',
                   header='R_pixels,sum_intensity', comments='', fmt='%.8g')
        written.append(sum_path)

    return written
