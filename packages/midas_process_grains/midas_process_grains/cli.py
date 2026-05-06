"""CLI: ``midas-process-grains`` (and ``python -m midas_process_grains``).

Mirrors the C ``ProcessGrains`` invocation pattern (single positional arg:
the parameter file path) with optional flags to override mode, device,
dtype, and a couple of merge knobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midas-process-grains",
        description=(
            "Pure-Python FF-HEDM grain-determination + strain pipeline "
            "(drop-in for ProcessGrains)."
        ),
    )
    p.add_argument(
        "param_file",
        type=Path,
        help="Path to paramstest.txt (the same file IndexerOMP/FitPosOrStrains "
             "consumed for this run).",
    )
    p.add_argument(
        "num_procs", type=int, nargs="?", default=1,
        help="CPU thread count (used only on cpu device). Default 1.",
    )
    p.add_argument(
        "--mode", choices=("legacy", "paper_claim", "spot_aware", "c_parity"),
        default="spot_aware",
        help="Pipeline mode. Use 'c_parity' for a bit-level replica of the "
             "C ProcessGrains pipeline (writes Grains.csv, GrainIDsKey.csv, "
             "SpotMatrix.csv in C's exact format).",
    )
    p.add_argument(
        "--min-nr-spots", type=int, default=None,
        help="MinNrSpots threshold (Stage 1 cluster-size cutoff). C ProcessGrains "
             "default is 1; the original peakfit_hard run used 3.",
    )
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    p.add_argument("--dtype", choices=("float32", "float64"), default=None)
    p.add_argument("--misori-tol", type=float, default=None,
                   help="Override the Phase 1 misorientation tolerance (degrees).")
    p.add_argument(
        "--strain-method",
        choices=(
            "kenesei", "kenesei_unbounded", "fable_beaudoin", "both",
            # backwards-compat aliases (resolved in params.validated())
            "lstsq", "lattice",
        ),
        default=None,
        help="Per-grain strain solver. Default: kenesei (bounded ±0.01, "
             "matches C reference). Use fable_beaudoin for the lattice-"
             "parameter route, or both to emit each.",
    )
    p.add_argument("--material", default=None,
                   help="Material name for stiffness lookup (e.g. Cu, Ni, Fe).")
    p.add_argument("--stiffness-file", type=Path, default=None,
                   help="Path to a 6×6 stiffness matrix (CSV/TXT/NPY).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write outputs. Default: param-file directory.")
    p.add_argument("--no-h5", action="store_true",
                   help="Skip writing data_consolidated.h5.")
    p.add_argument("--no-diagnostics-h5", action="store_true",
                   help="Skip writing processgrains_diagnostics.h5.")
    p.add_argument("--max-seeds", type=int, default=None,
                   help="Process only the first N alive seeds (smoke / dev).")
    p.add_argument("--version", action="version",
                   version=f"midas-process-grains {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns process exit code."""
    args = _build_parser().parse_args(argv)

    from .device import apply_cpu_threads, resolve_device, resolve_dtype
    from .pipeline import ProcessGrains

    # ── c_parity mode: dispatch to the C-replica pipeline and return ────────
    if args.mode == "c_parity":
        from .compute.c_parity_run import run_c_parity_pipeline_from_disk
        run_dir = args.param_file.parent
        out_dir = args.out_dir if args.out_dir is not None else run_dir
        device = resolve_device(args.device)
        # torch device strings: "cpu" / "cuda" / "cuda:0" / "mps"
        device_str = str(device) if not hasattr(device, "type") else (
            device.type if device.index is None else f"{device.type}:{device.index}"
        )
        apply_cpu_threads(args.num_procs, device)
        run_c_parity_pipeline_from_disk(
            run_dir=run_dir,
            out_dir=out_dir,
            min_nr_spots=(args.min_nr_spots
                          if args.min_nr_spots is not None else 1),
            device=device_str,
        )
        return 0

    pg = ProcessGrains.from_param_file(
        args.param_file,
        device=args.device,
        dtype=args.dtype,
    )
    apply_cpu_threads(args.num_procs, pg.device)

    # CLI overrides on top of paramstest.
    if args.misori_tol is not None:
        pg.params.MisoriTol = float(args.misori_tol)
    if args.strain_method is not None:
        pg.params.StrainMethod = args.strain_method
    if args.material is not None:
        pg.params.MaterialName = args.material
    if args.stiffness_file is not None:
        pg.params.StiffnessFile = str(args.stiffness_file)
    pg.params = pg.params.validated()

    if args.max_seeds is not None:
        pg.params.raw["__max_seeds__"] = [str(args.max_seeds)]

    result = pg.run(mode=args.mode)
    out_dir = args.out_dir if args.out_dir is not None else pg.run_dir
    result.write(
        out_dir,
        h5=not args.no_h5,
        diagnostics_h5=not args.no_diagnostics_h5,
    )
    print(
        f"midas-process-grains {__version__}: "
        f"{result.n_grains} grains written to {out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
