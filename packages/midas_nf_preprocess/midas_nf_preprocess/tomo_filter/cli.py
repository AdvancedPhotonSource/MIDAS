"""``midas-nf-preprocess tomo-filter`` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..hex_grid.io import read_grid_txt, write_grid_txt
from .filter import filter_grid_by_bbox, filter_grid_by_tomo, load_square_tomo


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "grid_in", help="Input grid.txt (typically from `midas-nf-preprocess hex-grid`)."
    )
    parser.add_argument(
        "grid_out", help="Output filtered grid.txt path."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--tomo",
        help="Path to a square uint8 binary tomography mask.",
    )
    mode.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="Rectangular bounding box mask in micrometers.",
    )
    parser.add_argument(
        "--px-tomo",
        type=float,
        default=None,
        help="Pixel size of --tomo image (um/pixel). Required with --tomo.",
    )


def run(args: argparse.Namespace) -> int:
    grid = read_grid_txt(args.grid_in)
    if args.tomo is not None:
        if args.px_tomo is None:
            raise SystemExit("--px-tomo is required when --tomo is given")
        tomo = load_square_tomo(args.tomo)
        filtered, mask = filter_grid_by_tomo(grid, tomo, args.px_tomo)
    else:
        filtered, mask = filter_grid_by_bbox(grid, args.bbox)
    write_grid_txt(filtered, args.grid_out)
    print(
        f"Wrote {args.grid_out} ({filtered.shape[0]}/{grid.shape[0]} grid points kept)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="midas-nf-preprocess tomo-filter",
        description="Filter a hex grid by a tomography mask or bounding box.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
