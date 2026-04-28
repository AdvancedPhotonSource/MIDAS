"""End-user performance benchmark for midas-integrate.

Runs the same three stages reported in the paper (detector-mapper build,
sparse-CSR build, per-frame integrate + 1D profile) on one or more
detector geometries embedded in this script. Synthetic uniform-noise
images are used — per-frame compute is deterministic in the sparse-CSR
weights and independent of pixel intensity, so noise images exercise
the same code paths as real diffraction frames.

Quick start
-----------
    # Default: PILATUS3 2M only, CPU, float32, mode=floor
    python tests/benchmarks/run_benchmarks.py

    # All 8 detectors from the paper, CPU
    python tests/benchmarks/run_benchmarks.py --all

    # GPU sweep (requires CUDA-capable PyTorch)
    python tests/benchmarks/run_benchmarks.py --all --device cuda

    # Compare against pyFAI on the same machine (requires `pip install pyFAI`)
    python tests/benchmarks/run_benchmarks.py --all --pyfai

    # Save results JSON
    python tests/benchmarks/run_benchmarks.py --all --output my_results.json

Output
------
JSON written to ``--output`` (default ``benchmark_results_<host>.json``)
with one row per (detector, device, dtype, mode) combination, plus a
summary table printed to stdout.

The default benchmark takes ~30 s on a recent laptop (numba JIT warmup
+ 100 iterations + summary). A full ``--all`` GPU sweep takes ~3 min.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Local imports — work whether the package is pip-installed or run from a
# source checkout (sys.path injection covers the latter).
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from midas_integrate import __version__
from midas_integrate.params import IntegrationParams
from midas_integrate.detector_mapper import build_map
from midas_integrate.kernels import build_csr, integrate, profile_1d


# ─────────────────────────────────────────────────────────────────────────────
# Detector configurations (matches paper/test_runs/bench_*/ps_bench.txt)
# ─────────────────────────────────────────────────────────────────────────────
DETECTORS: dict[str, dict] = {
    "eiger2_500k": dict(
        NrPixelsY=1028, NrPixelsZ=512, px=75.0,
        BC_y=514.0, BC_z=256.0, RhoD=674.0,
        RMin=10.0, RMax=574.2,
    ),
    "pilatus3_1m": dict(
        NrPixelsY=981, NrPixelsZ=1043, px=172.0,
        BC_y=490.5, BC_z=521.5, RhoD=815.0,
        RMin=10.0, RMax=715.9,
    ),
    "pilatus3_2m": dict(
        NrPixelsY=1475, NrPixelsZ=1679, px=172.0,
        BC_y=737.5, BC_z=839.5, RhoD=1217.0,
        RMin=10.0, RMax=1117.4,
    ),
    "eiger2_4m": dict(
        NrPixelsY=2068, NrPixelsZ=2162, px=75.0,
        BC_y=1034.0, BC_z=1081.0, RhoD=1595.0,
        RMin=10.0, RMax=1495.9,
    ),
    "pilatus3_6m": dict(
        NrPixelsY=2463, NrPixelsZ=2527, px=172.0,
        BC_y=1231.5, BC_z=1263.5, RhoD=1864.0,
        RMin=10.0, RMax=1764.4,
    ),
    "varex_2923": dict(
        NrPixelsY=2880, NrPixelsZ=2880, px=150.0,
        BC_y=1440.0, BC_z=1440.0, RhoD=2136.0,
        RMin=10.0, RMax=2036.5,
    ),
    "eiger2_9m": dict(
        NrPixelsY=3110, NrPixelsZ=3269, px=75.0,
        BC_y=1555.0, BC_z=1634.5, RhoD=2356.0,
        RMin=10.0, RMax=2256.0,
    ),
    "eiger2_16m": dict(
        NrPixelsY=4148, NrPixelsZ=4362, px=75.0,
        BC_y=2074.0, BC_z=2181.0, RhoD=3109.0,
        RMin=10.0, RMax=3009.7,
    ),
}


def make_params(name: str, *,
                RBinSize: float = 1.0, EtaBinSize: float = 5.0,
                Lsd_um: float = 500_000.0) -> IntegrationParams:
    """Build an IntegrationParams matching the paper's bench geometry."""
    if name not in DETECTORS:
        raise KeyError(f"Unknown detector {name!r}; choose one of "
                       f"{sorted(DETECTORS)}")
    cfg = DETECTORS[name]
    return IntegrationParams(
        NrPixelsY=cfg["NrPixelsY"], NrPixelsZ=cfg["NrPixelsZ"],
        pxY=cfg["px"], pxZ=cfg["px"],
        Lsd=Lsd_um,
        BC_y=cfg["BC_y"], BC_z=cfg["BC_z"],
        RhoD=cfg["RhoD"],
        RMin=cfg["RMin"], RMax=cfg["RMax"],
        RBinSize=RBinSize,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBinSize,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hardware probe
# ─────────────────────────────────────────────────────────────────────────────
def probe_host() -> dict:
    """Collect a snapshot of CPU/GPU/library info for the result file."""
    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "midas_integrate_version": __version__,
        "n_threads_env": int(os.environ.get("OMP_NUM_THREADS",
                                            os.cpu_count() or 1)),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_names"] = [torch.cuda.get_device_name(i)
                                         for i in range(torch.cuda.device_count())]
        else:
            info["cuda_device_count"] = 0
    except ImportError:
        info["torch_version"] = "(not installed)"
    return info


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def time_one_run(
    detector: str,
    *,
    device: str,
    dtype: str,
    mode: str,
    n_warmup: int,
    n_iter: int,
    image_dtype: str,
    rbin: float,
    eta_bin: float,
    verbose: bool,
) -> dict:
    """Run the full pipeline once for a given (detector, device, dtype, mode).

    Returns a result dict with timing for build_map, build_csr, and
    integrate+profile_1d.
    """
    import torch
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64

    p = make_params(detector, RBinSize=rbin, EtaBinSize=eta_bin)
    if verbose:
        print(f"  → {detector:<14}  {device:<5}  {dtype:<7}  mode={mode:<8}"
              f"  ({p.NrPixelsY*p.NrPixelsZ/1e6:.2f} Mpx, "
              f"{p.n_r_bins}×{p.n_eta_bins} bins)")

    # 1) detector mapper  (numba; first call includes JIT compile cost)
    t0 = time.perf_counter()
    res = build_map(p, verbose=False, use_numba=True)
    map_ms = (time.perf_counter() - t0) * 1000.0
    pixmap = res.as_pixel_map()

    # 2) CSR build
    build_modes = (mode,) if mode == "floor" else ("bilinear", "gradient")
    t0 = time.perf_counter()
    geom = build_csr(
        pixmap,
        n_r=p.n_r_bins, n_eta=p.n_eta_bins,
        n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
        device=device, dtype=torch_dtype,
        bc_y=p.BC_y, bc_z=p.BC_z,
        build_modes=build_modes,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    csr_ms = (time.perf_counter() - t0) * 1000.0

    # 3) Synthetic image — uniform noise, identical compute path to real frames
    rng = np.random.default_rng(seed=0)
    np_dtype = {"uint16": np.uint16, "uint32": np.uint32,
                "float32": np.float32}[image_dtype]
    if np_dtype == np.float32:
        img = rng.random((p.NrPixelsZ, p.NrPixelsY), dtype=np.float32) * 1000.0
    else:
        img = rng.integers(0, 40_000, size=(p.NrPixelsZ, p.NrPixelsY),
                           dtype=np_dtype)
    img_t = torch.from_numpy(np.ascontiguousarray(img)).to(
        device=device, dtype=torch_dtype
    )

    # 4) Per-frame integrate + 1D profile
    for _ in range(n_warmup):
        int2d = integrate(img_t, geom, mode=mode, normalize=True)
        prof = profile_1d(int2d, geom, mode="area_weighted")
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    ts = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        t0 = time.perf_counter()
        int2d = integrate(img_t, geom, mode=mode, normalize=True)
        prof = profile_1d(int2d, geom, mode="area_weighted")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts[i] = (time.perf_counter() - t0) * 1000.0

    median_ms = float(np.median(ts))
    nnz = int(geom.csr_floor._nnz()) if mode == "floor" \
          else int(geom.csr_bilinear._nnz())
    return {
        "detector": detector, "device": device, "dtype": dtype, "mode": mode,
        "n_pixels_y": p.NrPixelsY, "n_pixels_z": p.NrPixelsZ,
        "mpx": p.NrPixelsY * p.NrPixelsZ / 1e6,
        "n_r": p.n_r_bins, "n_eta": p.n_eta_bins,
        "n_entries": nnz,
        "build_map_ms": map_ms,
        "build_csr_ms": csr_ms,
        "median_ms": median_ms,
        "p95_ms": _percentile(ts, 95),
        "min_ms": float(ts.min()),
        "max_ms": float(ts.max()),
        "fps": 1000.0 / median_ms,
        "n_warmup": n_warmup,
        "n_iter": n_iter,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional pyFAI comparison
# ─────────────────────────────────────────────────────────────────────────────
def time_pyfai(
    detector: str,
    *,
    method: tuple,
    n_warmup: int,
    n_iter: int,
    rbin: float,
    eta_bin: float,
    Lsd_um: float,
    image_dtype: str,
    label: str,
    verbose: bool,
) -> Optional[dict]:
    """Time pyFAI (CPU CSR/Cython by default) on the same geometry.

    Returns None and prints a hint if pyFAI is not importable.
    """
    try:
        import pyFAI
    except ImportError:
        if verbose:
            print(f"    pyFAI not installed — `pip install pyFAI` to enable")
        return None

    cfg = DETECTORS[detector]
    NY = cfg["NrPixelsY"]; NZ = cfg["NrPixelsZ"]
    px_um = cfg["px"]
    BCY = cfg["BC_y"]; BCZ = cfg["BC_z"]

    p = make_params(detector, RBinSize=rbin, EtaBinSize=eta_bin, Lsd_um=Lsd_um)
    n_r = p.n_r_bins
    R_MIN_MM = p.RMin * px_um * 1e-3
    R_MAX_MM = (p.RMin + n_r * rbin) * px_um * 1e-3

    # Build a PONI in-memory
    poni_text = (
        f"poni_version: 2\n"
        f"Detector: Detector\n"
        f'Detector_config: {{"pixel1": {px_um*1e-6}, "pixel2": {px_um*1e-6}, '
        f'"max_shape": [{NZ}, {NY}]}}\n'
        f"Distance: {Lsd_um*1e-6}\n"
        f"Poni1: {BCZ * px_um * 1e-6}\n"
        f"Poni2: {BCY * px_um * 1e-6}\n"
        f"Rot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1.0e-10\n"
    )
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".poni", mode="w", delete=False) as f:
        f.write(poni_text)
        poni_path = f.name

    try:
        ai = pyFAI.load(poni_path)
        rng = np.random.default_rng(seed=0)
        np_dtype = {"uint16": np.uint16, "uint32": np.uint32,
                    "float32": np.float32}[image_dtype]
        if np_dtype == np.float32:
            img = rng.random((NZ, NY), dtype=np.float32) * 1000.0
        else:
            img = rng.integers(0, 40_000, size=(NZ, NY), dtype=np_dtype)

        for _ in range(n_warmup):
            ai.integrate1d(img, npt=n_r, unit="r_mm",
                           radial_range=(R_MIN_MM, R_MAX_MM),
                           method=method,
                           correctSolidAngle=False,
                           polarization_factor=None)
        ts = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            t0 = time.perf_counter()
            ai.integrate1d(img, npt=n_r, unit="r_mm",
                           radial_range=(R_MIN_MM, R_MAX_MM),
                           method=method,
                           correctSolidAngle=False,
                           polarization_factor=None)
            ts[i] = (time.perf_counter() - t0) * 1000.0
        median_ms = float(np.median(ts))
        return {
            "detector": detector,
            "tool": "pyFAI", "tool_version": pyFAI.version,
            "method": str(method), "label": label,
            "median_ms": median_ms,
            "p95_ms": _percentile(ts, 95),
            "min_ms": float(ts.min()),
            "max_ms": float(ts.max()),
            "fps": 1000.0 / median_ms,
            "n_warmup": n_warmup, "n_iter": n_iter,
        }
    finally:
        os.unlink(poni_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_benchmarks.py",
        description="Run midas-integrate performance benchmarks on the local machine.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--detectors", nargs="+", default=["pilatus3_2m"],
        help=("Detector configurations to benchmark. "
              f"Choices: {', '.join(sorted(DETECTORS))}.\n"
              "Default: pilatus3_2m only."),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Benchmark all 8 detector configurations from the paper.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device: cpu | cuda | cuda:0 | mps. Default: cpu.\n"
             "If 'cuda' is requested but unavailable, falls back to cpu.",
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64"],
        help="Sparse matrix dtype. Default: float32.",
    )
    parser.add_argument(
        "--mode", default="floor", choices=["floor", "bilinear", "gradient"],
        help="Per-frame integration mode. Default: floor.",
    )
    parser.add_argument(
        "--image-dtype", default="uint16",
        choices=["uint16", "uint32", "float32"],
        help="Synthetic image dtype. Default: uint16.",
    )
    parser.add_argument(
        "--rbin", type=float, default=1.0,
        help="Radial bin size in pixels. Default: 1.0.",
    )
    parser.add_argument(
        "--eta-bin", type=float, default=5.0,
        help="Azimuthal bin size in degrees. Default: 5.0.",
    )
    parser.add_argument(
        "--n-warmup", type=int, default=10,
        help="Number of warmup iterations (excluded from timing). Default: 10.",
    )
    parser.add_argument(
        "--n-iter", type=int, default=100,
        help="Number of timed iterations. Default: 100.",
    )
    parser.add_argument(
        "--pyfai", action="store_true",
        help="Also benchmark pyFAI on the same geometries (CPU CSR/Cython only).\n"
             "Requires `pip install pyFAI`. Skipped silently if unavailable.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path. Default: benchmark_results_<host>.json in cwd.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-run progress; print only the final summary.",
    )
    args = parser.parse_args(argv)

    detectors = list(DETECTORS) if args.all else args.detectors
    bad = [d for d in detectors if d not in DETECTORS]
    if bad:
        print(f"ERROR: unknown detectors {bad}. "
              f"Choices: {sorted(DETECTORS)}", file=sys.stderr)
        return 2

    device = args.device
    if device.startswith("cuda") and not cuda_available():
        print(f"WARN: --device {device} requested but CUDA is not available; "
              "falling back to cpu.")
        device = "cpu"

    info = probe_host()
    if not args.quiet:
        print(f"midas-integrate {info['midas_integrate_version']} "
              f"@ {info['host']}")
        print(f"  platform : {info['platform']}")
        print(f"  python   : {info['python']}    "
              f"torch: {info.get('torch_version', '?')}")
        print(f"  cpus     : {info['cpu_count']}    "
              f"torch_threads: {info.get('torch_threads', '?')}")
        if info.get("cuda_device_count", 0) > 0:
            print(f"  cuda     : {info['cuda_device_names']}")
        print()
        print(f"Sweep: {len(detectors)} detectors × device={device} × "
              f"dtype={args.dtype} × mode={args.mode}  "
              f"({args.n_warmup} warmup + {args.n_iter} iter)")
        print()

    results = []
    for det in detectors:
        verbose = not args.quiet
        try:
            r = time_one_run(
                det,
                device=device, dtype=args.dtype, mode=args.mode,
                n_warmup=args.n_warmup, n_iter=args.n_iter,
                image_dtype=args.image_dtype,
                rbin=args.rbin, eta_bin=args.eta_bin,
                verbose=verbose,
            )
            results.append({"tool": "midas-integrate", **r})
            if verbose:
                print(f"      build_map: {r['build_map_ms']:>8.0f} ms   "
                      f"build_csr: {r['build_csr_ms']:>8.0f} ms   "
                      f"per-frame: {r['median_ms']:>7.3f} ms  "
                      f"({r['fps']:>6.0f} fps)")
        except Exception as exc:
            print(f"  ERROR {det}: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append({
                "tool": "midas-integrate", "detector": det,
                "device": device, "dtype": args.dtype, "mode": args.mode,
                "error": f"{type(exc).__name__}: {exc}",
            })

    # Optional pyFAI comparison
    pyfai_results = []
    if args.pyfai:
        if not args.quiet:
            print()
            print(f"Optional pyFAI comparison (CSR/Cython CPU, full-polygon split):")
        for det in detectors:
            r = time_pyfai(
                det,
                method=("full", "csr", "cython"),
                n_warmup=args.n_warmup, n_iter=args.n_iter,
                rbin=args.rbin, eta_bin=args.eta_bin, Lsd_um=500_000.0,
                image_dtype=args.image_dtype,
                label="pyFAI csr_full Cython CPU",
                verbose=not args.quiet,
            )
            if r is not None:
                pyfai_results.append(r)
                if not args.quiet:
                    print(f"  {det:<14} pyFAI: {r['median_ms']:>7.3f} ms  "
                          f"({r['fps']:>6.0f} fps)")

    # Summary
    if not args.quiet:
        print()
        print("=" * 78)
        print("Summary (median per-frame, fps)")
        print("=" * 78)
        print(f"{'detector':<14} {'mpx':>5}  {'midas_fps':>11}  "
              f"{'midas_ms':>9}  {'csr_build_ms':>13}  {'mapper_ms':>10}")
        for r in results:
            if "error" in r:
                print(f"{r['detector']:<14}        ERROR: {r['error']}")
                continue
            print(f"{r['detector']:<14} {r['mpx']:>5.2f}  "
                  f"{r['fps']:>11.0f}  {r['median_ms']:>9.3f}  "
                  f"{r['build_csr_ms']:>13.0f}  {r['build_map_ms']:>10.0f}")
        if pyfai_results:
            print()
            print(f"{'detector':<14} {'pyFAI_fps':>10}  {'pyFAI_ms':>9}  "
                  f"{'speedup_vs_pyFAI':>17}")
            midas_by_det = {r["detector"]: r for r in results if "fps" in r}
            for r in pyfai_results:
                m = midas_by_det.get(r["detector"])
                speed = (m["fps"] / r["fps"]) if (m and "fps" in m) else 0.0
                print(f"{r['detector']:<14} {r['fps']:>10.0f}  "
                      f"{r['median_ms']:>9.3f}  {speed:>14.2f}x")

    # Write JSON
    out_path = args.output
    if out_path is None:
        out_path = Path(f"benchmark_results_{info['host'].split('.')[0]}.json")
    out = {
        "host_info": info,
        "config": {
            "device": device, "dtype": args.dtype, "mode": args.mode,
            "image_dtype": args.image_dtype,
            "rbin": args.rbin, "eta_bin": args.eta_bin,
            "n_warmup": args.n_warmup, "n_iter": args.n_iter,
            "detectors": detectors,
        },
        "midas_integrate": results,
        "pyfai": pyfai_results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    if not args.quiet:
        print()
        print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
