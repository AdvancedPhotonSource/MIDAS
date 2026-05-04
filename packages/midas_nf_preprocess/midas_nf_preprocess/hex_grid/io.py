"""grid.txt I/O.

File format (matches ``MakeHexGrid.c`` L209-L213):

    <N>
    dx0 dy0 x0 y0 edge_half0
    dx1 dy1 x1 y1 edge_half1
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch


def write_grid_txt(points: torch.Tensor, path: Union[str, Path]) -> None:
    """Write a hex grid to a MIDAS grid.txt file."""
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError(
            f"Expected (N, 5) tensor, got shape {tuple(points.shape)}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = points.detach().cpu().numpy()
    with open(path, "w") as f:
        f.write(f"{arr.shape[0]}\n")
        # Match the C "%f %f %f %f %f\n" format (default 6-digit precision).
        for row in arr:
            f.write(
                f"{row[0]:f} {row[1]:f} {row[2]:f} {row[3]:f} {row[4]:f}\n"
            )


def read_grid_txt(path: Union[str, Path]) -> torch.Tensor:
    """Read a MIDAS grid.txt file into a (N, 5) double tensor."""
    path = Path(path)
    with open(path, "r") as f:
        first = f.readline().strip()
        try:
            n = int(first)
        except ValueError:
            raise ValueError(
                f"{path}: expected first line to be an integer count, got '{first}'"
            )
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}: expected 5 columns per row, got {len(parts)}: {line!r}"
                )
            rows.append([float(p) for p in parts[:5]])
    if len(rows) != n:
        raise ValueError(
            f"{path}: header says {n} rows, found {len(rows)}"
        )
    return torch.from_numpy(np.asarray(rows, dtype=np.float64))
