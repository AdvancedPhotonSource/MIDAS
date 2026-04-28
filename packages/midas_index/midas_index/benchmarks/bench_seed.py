"""Per-seed throughput benchmark for `process_seed`.

Drives the full forward + match pipeline on a synthetic 5-grain dataset
(generated in-memory via `IndexerForwardAdapter`) and reports seeds/sec.

Useful for validating perf changes (e.g. the (y0,z0) cartesian-batch
vectorization in pipeline.py).

Usage:
    python -m midas_index.benchmarks.bench_seed --n-grains 5 --n-iter 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

from midas_index import IndexerParams
from midas_index.compute.forward_adapter import IndexerForwardAdapter
from midas_index.io import build_bin_index
from midas_index.pipeline import IndexerContext, process_seed


def _toy_params() -> IndexerParams:
    p = IndexerParams()
    p.Distance = 1_000_000.0
    p.Wavelength = 0.172979
    p.Rsample = 250.0
    p.Hbeam = 200.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)
    p.StepsizePos = 100.0
    p.StepsizeOrient = 0.5
    p.MarginOme = 0.5
    p.MarginRad = 500.0
    p.MarginRadial = 500.0
    p.MarginEta = 500.0
    p.EtaBinSize = 0.1
    p.OmeBinSize = 0.1
    p.ExcludePoleAngle = 6.0
    p.MinMatchesToAcceptFrac = 0.1
    p.RingNumbers = [1, 2, 3, 4]
    p.RingRadii = {
        1: 73582.31550724161,
        2: 85023.04143552633,
        3: 120567.4304605822,
        4: 141666.90427076322,
    }
    p.OmegaRanges = [(-180.0, 180.0)]
    p.BoxSizes = [(-2_000_000.0, 2_000_000.0, -2_000_000.0, 2_000_000.0)]
    p.UseFriedelPairs = 0
    p.OutputFolder = "."
    return p


def _toy_hkls():
    a = 4.08
    one_over_a = 1.0 / a
    wl = 0.172979
    hkls = []
    int_rows = []
    real_rows = []
    for h, k, l, ring in [
        (1, -1, -1, 1), (1, 1, 1, 1), (-1, 1, 1, 1), (1, -1, 1, 1),
        (-1, -1, 1, 1), (-1, 1, -1, 1), (1, 1, -1, 1), (-1, -1, -1, 1),
        (2, 0, 0, 2), (-2, 0, 0, 2), (0, 2, 0, 2), (0, -2, 0, 2),
        (0, 0, 2, 2), (0, 0, -2, 2),
    ]:
        g = (h * one_over_a, k * one_over_a, l * one_over_a)
        d = 1.0 / math.sqrt(sum(x * x for x in g))
        sin_th = wl / (2.0 * d)
        if not 0 < sin_th <= 1.0:
            continue
        th_rad = math.asin(sin_th)
        radius = 73582.31550724161 if ring == 1 else 85023.04143552633
        real_rows.append((g[0], g[1], g[2], float(ring), d, th_rad, radius))
        int_rows.append((h, k, l, ring))
    return (
        np.asarray(real_rows, dtype=np.float64),
        np.asarray(int_rows, dtype=np.int64),
    )


def _build_dataset(n_grains: int, seed: int):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n_grains, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n_grains, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return torch.as_tensor(R, dtype=torch.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grains", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Iterations of process_seed to time")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = _toy_params()
    hkls_real_np, hkls_int_np = _toy_hkls()
    R = _build_dataset(args.n_grains, args.seed)

    # Forward-sim obs via the adapter
    adapter = IndexerForwardAdapter(
        params=params,
        hkls_real=torch.as_tensor(hkls_real_np, dtype=torch.float64),
        hkls_int=torch.as_tensor(hkls_int_np, dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    pos = torch.zeros(R.shape[0], 3, dtype=torch.float64)
    theor, valid = adapter.simulate(R, pos)

    rows = []
    spot_id = 1
    for g in range(theor.shape[0]):
        for k in range(theor.shape[1]):
            if not bool(valid[g, k]):
                continue
            y = float(theor[g, k, 10])
            z = float(theor[g, k, 11])
            rows.append([
                y, z, float(theor[g, k, 6]), math.sqrt(y * y + z * z),
                float(spot_id), float(theor[g, k, 9]),
                float(theor[g, k, 7]), float(theor[g, k, 8]) * 2.0,
                float(theor[g, k, 13]),
            ])
            spot_id += 1
    obs = np.asarray(rows, dtype=np.float64)
    bin_data, bin_ndata = build_bin_index(
        obs, eta_bin_size=0.1, ome_bin_size=0.1, n_rings=4,
        margin_eta=params.MarginEta, margin_ome=params.MarginOme,
        stepsize_orient=params.StepsizeOrient,
        ring_radii=params.RingRadii,
    )

    ctx = IndexerContext(
        params=params, hkls_real=hkls_real_np, hkls_int=hkls_int_np,
        obs=obs, bin_data=bin_data, bin_ndata=bin_ndata,
        device=torch.device("cpu"), dtype=torch.float64,
    )

    spot_ids = obs[:, 4].astype(int)
    print(f"benchmark: {args.n_grains} grains -> {len(obs)} obs spots")
    print(f"           {len(spot_ids)} candidate seed spots")

    # Warm up
    process_seed(int(spot_ids[0]), ctx)

    # Time process_seed across a sample
    sample = spot_ids[: min(args.n_grains, len(spot_ids))]
    timings = []
    for _ in range(args.n_iter):
        t0 = time.perf_counter()
        for sid in sample:
            process_seed(int(sid), ctx)
        timings.append(time.perf_counter() - t0)
    n_seeds = len(sample)
    best = min(timings)
    print(f"\n  best of {args.n_iter}: {best:.3f}s for {n_seeds} seeds")
    print(f"  -> {best / n_seeds * 1000:.1f} ms/seed, {n_seeds / best:.1f} seeds/sec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
