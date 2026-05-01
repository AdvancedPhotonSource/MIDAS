"""Focused micro-benchmark: cost of Poisson variance propagation.

Compares ``integrate`` (one SpMV) vs. ``integrate_with_variance``
(two SpMVs against the same image vector) back-to-back on identical
inputs. Reports the per-frame overhead so reviewers can see exactly
what the variance cost is on the same hardware that produced the
intensity-only numbers in the paper.

Usage on alleppey:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python tests/benchmarks/bench_variance_overhead.py \\
            --device cuda --detectors pilatus3_2m eiger2_16m
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from midas_integrate.detector_mapper import build_map
from midas_integrate.kernels import (
    build_csr,
    integrate,
    integrate_with_variance,
)

# Pull detector configs from the existing benchmark module to keep them in lockstep.
sys.path.insert(0, str(_THIS_DIR))
from run_benchmarks import DETECTORS, make_params  # noqa: E402


def time_run(
    name: str,
    *,
    device: str,
    dtype: str,
    n_warmup: int,
    n_iter: int,
) -> dict:
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    p = make_params(name, RBinSize=1.0, EtaBinSize=5.0)
    res = build_map(p, verbose=False, use_numba=True)
    pixmap = res.as_pixel_map()

    geom = build_csr(
        pixmap,
        n_r=p.n_r_bins, n_eta=p.n_eta_bins,
        n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
        device=device, dtype=torch_dtype,
        bc_y=p.BC_y, bc_z=p.BC_z,
        build_modes=("floor",),
        compute_variance=True,  # always build the squared CSR so both timed loops use the same geom
    )

    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 40_000, size=(p.NrPixelsZ, p.NrPixelsY),
                       dtype=np.uint16)
    img_t = torch.from_numpy(np.ascontiguousarray(img)).to(
        device=device, dtype=torch_dtype
    )
    flat = img_t.reshape(-1)

    sp = geom.csr_floor
    sp_sq = geom.csr_floor_sq
    area = geom.area_per_bin

    def one_intensity():
        raw = torch.matmul(sp, flat.unsqueeze(1)).squeeze(1)
        return raw / torch.clamp(area, min=1e-9)

    def one_intensity_plus_variance():
        raw = torch.matmul(sp, flat.unsqueeze(1)).squeeze(1)
        raw_v = torch.matmul(sp_sq, flat.unsqueeze(1)).squeeze(1)
        denom = torch.clamp(area, min=1e-9)
        return raw / denom, raw_v / (denom * denom)

    def time_loop(fn, n):
        ts = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter()
            fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ts[i] = (time.perf_counter() - t0) * 1000.0
        return ts

    # warmup both
    for _ in range(n_warmup):
        one_intensity()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        one_intensity_plus_variance()
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    # interleaved timing: time A, then B, then A, then B, ... so any system
    # transient affects both equally
    n_block = max(2, n_iter // 4)
    ts_int = np.empty(n_iter, dtype=np.float64)
    ts_var = np.empty(n_iter, dtype=np.float64)
    pos_i = 0
    pos_v = 0
    while pos_i < n_iter or pos_v < n_iter:
        b = min(n_block, n_iter - pos_i)
        if b > 0:
            tt = time_loop(one_intensity, b)
            ts_int[pos_i:pos_i + b] = tt
            pos_i += b
        b = min(n_block, n_iter - pos_v)
        if b > 0:
            tt = time_loop(one_intensity_plus_variance, b)
            ts_var[pos_v:pos_v + b] = tt
            pos_v += b

    out = {
        "detector": name,
        "device": device,
        "dtype": dtype,
        "mpx": p.NrPixelsY * p.NrPixelsZ / 1e6,
        "median_intensity_ms": float(np.median(ts_int)),
        "median_with_variance_ms": float(np.median(ts_var)),
        "min_intensity_ms": float(ts_int.min()),
        "min_with_variance_ms": float(ts_var.min()),
        "fps_intensity": 1000.0 / float(np.median(ts_int)),
        "fps_with_variance": 1000.0 / float(np.median(ts_var)),
        "n_warmup": n_warmup,
        "n_iter": n_iter,
    }
    out["overhead_pct"] = 100.0 * (out["median_with_variance_ms"]
                                   / out["median_intensity_ms"] - 1.0)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectors", nargs="+",
                        default=["pilatus3_2m", "eiger2_16m"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float64"])
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--output", type=Path,
                        default=Path(f"variance_overhead_{socket.gethostname()}.json"))
    args = parser.parse_args(argv)

    print(f"variance-overhead micro-benchmark @ {socket.gethostname()} "
          f"device={args.device} dtype={args.dtype}")
    print(f"  warmup={args.n_warmup}  iter={args.n_iter}")
    print()
    rows = []
    for name in args.detectors:
        if name not in DETECTORS:
            print(f"  skipped unknown detector {name!r}")
            continue
        print(f"  → {name}")
        r = time_run(
            name, device=args.device, dtype=args.dtype,
            n_warmup=args.n_warmup, n_iter=args.n_iter,
        )
        print(f"     intensity:        {r['median_intensity_ms']:7.4f} ms"
              f"  ({r['fps_intensity']:7.0f} fps)")
        print(f"     intensity+sigma:  {r['median_with_variance_ms']:7.4f} ms"
              f"  ({r['fps_with_variance']:7.0f} fps)"
              f"   overhead: {r['overhead_pct']:+6.1f}%")
        rows.append(r)

    with open(args.output, "w") as f:
        json.dump({"host": socket.gethostname(), "rows": rows}, f, indent=2)
    print()
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
