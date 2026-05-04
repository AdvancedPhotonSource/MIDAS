"""``midas-nf-preprocess hex-grid`` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .grid import HexGrid
from .params import HexGridParams


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "parameter_file", help="MIDAS parameter file with GridSize, Rsample, etc."
    )
    parser.add_argument(
        "--device", default=None, help="cpu | cuda | mps (default: auto-detect)."
    )
    parser.add_argument(
        "--dtype", default=None, help="fp32 | fp64 (default: fp64 on cpu)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output grid.txt path (default: <output_directory>/<grid_filename>).",
    )


def run(args: argparse.Namespace) -> int:
    params = HexGridParams.from_paramfile(args.parameter_file)
    import torch

    dtype = (
        torch.float32
        if args.dtype and args.dtype.lower() in ("fp32", "float32")
        else torch.float64
    )
    grid = HexGrid.from_params(params, device=args.device, dtype=dtype)
    out_path = args.output or str(
        Path(params.output_directory) / params.grid_filename
    )
    grid.write(out_path)
    print(f"Wrote {out_path} ({grid.n_points} grid points)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-nf-preprocess hex-grid",
        description="Generate a hexagonal voxel grid (port of MakeHexGrid).",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
