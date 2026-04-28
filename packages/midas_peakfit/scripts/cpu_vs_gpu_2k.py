"""Run the 2,000-grain FF-HEDM benchmark twice (CPU peakfit / GPU peakfit) and
compare the recovered grains against the simulation ground truth.

Usage:
    python cpu_vs_gpu_2k.py --nCPUs 16 --nGrains 2000 --seed 42
        [--midas-home /home/beams/S1IDUSER/opt/MIDAS]
        [--out-root /scratch/s1iduser/cpu_vs_gpu_2k]

Each pipeline runs through ff_MIDAS.py, the only difference being the new
``-peakFitGPU`` flag (0 = OMP C tool, 1 = peakfit_torch). Forward simulation
and the input zip are produced once and reused for both backends.

For each backend we report:
  * pipeline wall-clock
  * matched grain count
  * position-error distribution (Δx,Δy,Δz vs ground truth in µm)
  * orientation-error distribution (misorientation in degrees)
  * lattice-parameter / linear-strain distribution (a,b,c only — angles ~0)

The two backend reports are then printed side-by-side so any divergence in
fitted grains is immediately visible.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Grain parsing & metrics
# --------------------------------------------------------------------------

def parse_grainsim_csv(path):
    """Ground-truth grain dump from generate_grains.py.

    Columns: GrainID O11..O33 X Y Z a b c alpha beta gamma DiffPos DiffOme
             DiffAngle GrainRadius Confidence eFab(9) eKen(9) RMSErrorStrain
             PhaseNr Eul0 Eul1 Eul2
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            v = line.split()
            if len(v) < 19:
                continue
            try:
                rows.append({
                    'id': int(v[0]),
                    'orient': np.array([float(x) for x in v[1:10]]).reshape(3, 3),
                    'pos': np.array([float(x) for x in v[10:13]]),
                    'lattice': np.array([float(x) for x in v[13:19]]),
                })
            except ValueError:
                continue
    return rows


def parse_grains_csv(path):
    """Parse a fitted Grains.csv from ff_MIDAS.py / FitPosOrStrains.

    The header line starts with ``%GrainID`` and lists the columns. We grab
    O11..O33, X/Y/Z, a/b/c/alpha/beta/gamma indices from that header so we
    are not pinned to a specific column ordering.
    """
    if not Path(path).exists():
        return []
    with open(path) as f:
        lines = f.readlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith('%GrainID') or ln.lstrip().startswith('%ID'):
            header_idx = i
            break
    if header_idx is None:
        # Fall back to fixed columns matching GrainsSim.csv
        return parse_grainsim_csv(path)

    cols = lines[header_idx].lstrip('%').split()
    name_to_col = {c: i for i, c in enumerate(cols)}

    def col(name, default=None):
        if name in name_to_col:
            return name_to_col[name]
        return default

    om_cols = [col(f'O{i}{j}') for i in (1, 2, 3) for j in (1, 2, 3)]
    pos_cols = [col('X'), col('Y'), col('Z')]
    lat_cols = [col('a'), col('b'), col('c'),
                col('alpha'), col('beta'), col('gamma')]
    if any(c is None for c in om_cols + pos_cols + lat_cols):
        # Header didn't contain every field we needed
        return parse_grainsim_csv(path)

    grains = []
    for ln in lines[header_idx + 1:]:
        ln = ln.strip()
        if not ln or ln.startswith('%'):
            continue
        v = ln.split()
        if len(v) < max(om_cols + pos_cols + lat_cols) + 1:
            continue
        try:
            grains.append({
                'id': int(float(v[0])),
                'orient': np.array([float(v[c]) for c in om_cols]).reshape(3, 3),
                'pos': np.array([float(v[c]) for c in pos_cols]),
                'lattice': np.array([float(v[c]) for c in lat_cols]),
            })
        except (ValueError, IndexError):
            continue
    return grains


def misorientation_deg(R1, R2):
    """Crystal-symmetry-free misorientation angle (degrees)."""
    R = R1 @ R2.T
    tr = np.clip(np.trace(R), -1.0, 3.0)
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def match_to_ground_truth(gt_grains, fit_grains, pos_tol_um=200.0):
    """Greedy match by minimum distance, single-use of each fit grain.

    Returns a list of (gt, fit) pairs, plus the unmatched GT count.
    """
    pairs = []
    used = set()
    gt_pos = np.stack([g['pos'] for g in gt_grains])
    fit_pos = np.stack([g['pos'] for g in fit_grains]) if fit_grains else None

    for i, g in enumerate(gt_grains):
        if fit_pos is None:
            continue
        d = np.linalg.norm(fit_pos - gt_pos[i], axis=1)
        order = np.argsort(d)
        for j in order:
            if j in used:
                continue
            if d[j] > pos_tol_um:
                break
            # Confirm orientation also agrees within ~5 deg, else skip
            if misorientation_deg(g['orient'], fit_grains[j]['orient']) > 5.0:
                continue
            pairs.append((g, fit_grains[j]))
            used.add(int(j))
            break
    n_unmatched = len(gt_grains) - len(pairs)
    return pairs, n_unmatched


def summarize(label, gt_grains, fit_grains, runtime_s):
    pairs, n_unmatched = match_to_ground_truth(gt_grains, fit_grains)
    n_match = len(pairs)
    n_total_gt = len(gt_grains)
    n_fit = len(fit_grains)

    if n_match == 0:
        return {
            'label': label, 'runtime_s': runtime_s, 'n_total_gt': n_total_gt,
            'n_fit': n_fit, 'n_match': 0,
        }

    pos_err = np.array([np.linalg.norm(g['pos'] - f['pos']) for g, f in pairs])
    ori_err = np.array([misorientation_deg(g['orient'], f['orient'])
                        for g, f in pairs])
    # Linear strain along each lattice axis: (a_fit - a_gt) / a_gt
    strain_err = np.array([
        (f['lattice'][:3] - g['lattice'][:3]) / g['lattice'][:3]
        for g, f in pairs
    ])  # shape (N, 3)
    strain_err_microstrain = strain_err * 1.0e6  # convert to µε
    return {
        'label': label,
        'runtime_s': runtime_s,
        'n_total_gt': n_total_gt,
        'n_fit': n_fit,
        'n_match': n_match,
        'n_unmatched': n_unmatched,
        'pos_med_um': float(np.median(pos_err)),
        'pos_p95_um': float(np.percentile(pos_err, 95)),
        'pos_max_um': float(pos_err.max()),
        'ori_med_deg': float(np.median(ori_err)),
        'ori_p95_deg': float(np.percentile(ori_err, 95)),
        'ori_max_deg': float(ori_err.max()),
        'strain_med_ue': float(np.median(np.abs(strain_err_microstrain))),
        'strain_p95_ue': float(np.percentile(np.abs(strain_err_microstrain), 95)),
        'strain_max_ue': float(np.abs(strain_err_microstrain).max()),
    }


# --------------------------------------------------------------------------
# Pipeline driver
# --------------------------------------------------------------------------

def run_pipeline(midas_home: Path, work_dir: Path, n_cpus: int, n_grains: int,
                 seed: int, backend: str) -> float:
    """Run test_ff_hedm.py once for a specific backend, return wall-clock.

    The test script handles forward simulation, zarr generation, and the
    actual ff_MIDAS.py invocation. We disable cleanup so we can read the
    fitted Grains.csv afterwards.
    """
    test_script = midas_home / 'tests' / 'test_ff_hedm.py'
    cmd = [
        sys.executable, str(test_script),
        '-nCPUs', str(n_cpus),
        '--nGrains', str(n_grains),
        '--seed', str(seed),
        '--backend', backend,
        '--no-cleanup',
        '--skip-preflight',
    ]
    print(f"\n>>> Running backend={backend}: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(midas_home / 'tests')).returncode
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f'pipeline failed for backend={backend} (rc={rc})')
    print(f">>> backend={backend} finished in {elapsed:.1f} s")
    return elapsed


def stash_results(work_dir: Path, dest: Path, backend: str):
    """Move the fitted Grains.csv (and a couple of small companion files)
    to the per-backend directory before the next run clobbers LayerNr_1.
    """
    src = work_dir / 'LayerNr_1'
    if not src.exists():
        raise RuntimeError(f'expected {src} after pipeline run; not found')
    dest = dest / backend
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in ('Grains.csv', 'Output', 'midas_log'):
        s = src / name
        if s.exists():
            if s.is_dir():
                shutil.copytree(s, dest / name)
            else:
                shutil.copy2(s, dest / name)
    return dest


def render_table(rows):
    """Side-by-side metrics print."""
    keys = [
        ('runtime_s',     'pipeline wall-clock', 's',   '{:8.1f}'),
        ('n_fit',         'fitted grain count',  '',    '{:8d}'),
        ('n_match',       'matched to GT',       '',    '{:8d}'),
        ('n_unmatched',   'GT unmatched',        '',    '{:8d}'),
        ('pos_med_um',    'position median',     'µm',  '{:8.3f}'),
        ('pos_p95_um',    'position p95',        'µm',  '{:8.3f}'),
        ('pos_max_um',    'position max',        'µm',  '{:8.3f}'),
        ('ori_med_deg',   'misorient median',    '°',   '{:8.4f}'),
        ('ori_p95_deg',   'misorient p95',       '°',   '{:8.4f}'),
        ('ori_max_deg',   'misorient max',       '°',   '{:8.4f}'),
        ('strain_med_ue', 'strain |ε| median',   'µε',  '{:8.1f}'),
        ('strain_p95_ue', 'strain |ε| p95',      'µε',  '{:8.1f}'),
        ('strain_max_ue', 'strain |ε| max',      'µε',  '{:8.1f}'),
    ]

    headers = ['metric'] + [r['label'] for r in rows]
    width_metric = max(len(k[1] + ' [' + k[2] + ']') for k in keys) + 2
    width_val = 14
    print()
    print(' ' * width_metric + ''.join(h.center(width_val) for h in headers[1:]))
    print('-' * (width_metric + width_val * len(rows)))
    for key, label, unit, fmt in keys:
        suffix = f' [{unit}]' if unit else ''
        line = (label + suffix).ljust(width_metric)
        for r in rows:
            v = r.get(key)
            if v is None:
                cell = 'n/a'
            else:
                cell = fmt.format(v)
            line += cell.center(width_val)
        print(line)
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--midas-home', default=os.environ.get(
        'MIDAS_HOME', '/home/beams/S1IDUSER/opt/MIDAS'))
    p.add_argument('--nCPUs', type=int, default=16)
    p.add_argument('--nGrains', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-root', default=None,
                   help='Where to stash per-backend Grains.csv copies. '
                        'Defaults to <midas_home>/FF_HEDM/Example/_cpu_vs_gpu_out')
    args = p.parse_args()

    midas_home = Path(args.midas_home).resolve()
    work_dir = midas_home / 'FF_HEDM' / 'Example'
    out_root = Path(args.out_root) if args.out_root else (
        work_dir / '_cpu_vs_gpu_out')
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for backend in ('c', 'torch'):
        elapsed = run_pipeline(midas_home, work_dir, args.nCPUs, args.nGrains,
                               args.seed, backend)
        dest = stash_results(work_dir, out_root, backend)
        gt = parse_grainsim_csv(work_dir / 'GrainsSim.csv')
        fit = parse_grains_csv(dest / 'Grains.csv')
        print(f"[{backend}] ground-truth grains: {len(gt)}, "
              f"fitted grains: {len(fit)}")
        rows.append(summarize('CPU (OMP C)' if backend == 'c'
                              else 'GPU (peakfit_torch)', gt, fit, elapsed))

    print('\n' + '=' * 78)
    print(f"  CPU vs GPU peak-fitting comparison "
          f"(nGrains={args.nGrains}, seed={args.seed}, "
          f"nCPUs={args.nCPUs})")
    print('=' * 78)
    render_table(rows)

    # Inter-backend delta: same GT, same matching, what's the per-grain
    # spread between the two pipelines?
    gt = parse_grainsim_csv(work_dir / 'GrainsSim.csv')
    fit_c = parse_grains_csv(out_root / 'c' / 'Grains.csv')
    fit_t = parse_grains_csv(out_root / 'torch' / 'Grains.csv')
    pairs_c, _ = match_to_ground_truth(gt, fit_c)
    pairs_t, _ = match_to_ground_truth(gt, fit_t)
    # Match by GT id
    by_id_c = {pair[0]['id']: pair[1] for pair in pairs_c}
    by_id_t = {pair[0]['id']: pair[1] for pair in pairs_t}
    common = sorted(set(by_id_c) & set(by_id_t))
    if common:
        d_pos = np.array([np.linalg.norm(by_id_c[i]['pos'] - by_id_t[i]['pos'])
                          for i in common])
        d_ori = np.array([misorientation_deg(by_id_c[i]['orient'],
                                             by_id_t[i]['orient'])
                          for i in common])
        d_str = np.array([
            np.abs((by_id_c[i]['lattice'][:3] - by_id_t[i]['lattice'][:3])
                   / by_id_t[i]['lattice'][:3]).max() * 1e6
            for i in common
        ])
        print(f"Inter-backend delta over {len(common)} commonly-matched grains:")
        print(f"  position diff   (µm) : "
              f"med={np.median(d_pos):.3f}  p95={np.percentile(d_pos, 95):.3f}  "
              f"max={d_pos.max():.3f}")
        print(f"  misorientation (deg) : "
              f"med={np.median(d_ori):.4f}  p95={np.percentile(d_ori, 95):.4f}  "
              f"max={d_ori.max():.4f}")
        print(f"  strain |Δ|     (µε) : "
              f"med={np.median(d_str):.1f}  p95={np.percentile(d_str, 95):.1f}  "
              f"max={d_str.max():.1f}")
    else:
        print('No common matched grains between CPU and GPU runs.')

    print('\nPer-backend grains stashed under:', out_root)


if __name__ == '__main__':
    main()
