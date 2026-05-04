"""Vectorized hex-grid generation.

Direct vectorized port of ``MakeHexGrid.c`` HexGrid (L23-L67) and the
allocation loop at L191-L196. The C version is two nested for-loops over
``i in [-NrHex, NrHex] \\ {0}`` and ``j in [0, NrRowElements)``; we replicate
the same arithmetic with index tensors and ``cat`` row blocks together.

All math is double-precision by default but the function accepts a ``dtype``
override. Output is a torch.Tensor of shape ``(N, 5)`` with columns
``(dx, dy, x, y, edge_half)`` matching the C grid.txt order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from ..device import resolve_device, resolve_dtype


def n_grid_points(grid_size: float, r_sample: float) -> int:
    """Closed form for the number of hex-grid voxels.

    The C code (L191-L193) sums ``2 * (2*NrHex - i) + 1`` for
    ``i in 1..NrHex`` (each row has that many triangles, doubled because both
    halves of the grid contribute):

        sum_{i=1}^{NrHex} (4*NrHex - 2i + 1)
            = NrHex * (4*NrHex + 1) - 2 * NrHex*(NrHex+1)/2
            = NrHex * (4*NrHex + 1) - NrHex*(NrHex+1)
            = NrHex * (3*NrHex)
            = 3 * NrHex^2

    Doubled (top + bottom of the grid, since i=0 is skipped):

        N = 2 * 3 * NrHex^2 = 6 * NrHex^2

    Wait -- the C accumulator (NrGridElements at L191-L193) already has the
    leading factor of 2 baked in, so the loop computes::

        NrGridElements = sum_{i=1}^{NrHex} 2 * (2 * (2*NrHex - i) + 1)
                       = 2 * NrHex * (3*NrHex)
                       = 6 * NrHex^2.
    """
    a_large = (2.0 * r_sample) / math.sqrt(3.0)
    nr_hex = int(math.ceil(a_large / grid_size))
    return 6 * nr_hex * nr_hex


def make_hex_grid(
    grid_size: float,
    r_sample: float,
    edge_length: Optional[float] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.Tensor:
    """Vectorized hex grid covering a disk of radius ``r_sample``.

    Parameters
    ----------
    grid_size  : in-plane spacing between adjacent voxel centers (um).
    r_sample   : sample radius the grid must cover (um).
    edge_length : equilateral-triangle edge length (um). If None, equals ``grid_size``
        (matches the C default at L158-L159).
    device, dtype : standard torch construction kwargs.

    Returns
    -------
    Tensor of shape ``(N, 5)``: columns ``(dx, dy, x, y, edge_half)`` where:

      - ``dx, dy``   : per-row "sub-triangle" offsets used by simulateNF
                       (see C L31-L32 ``xt1, xt2``)
      - ``x, y``     : voxel center coordinates (um)
      - ``edge_half`` : ``edge_length / 2`` (constant; mirrors C L58)

    The row order matches the C output exactly (C row index ``i`` from
    ``-NrHex`` to ``NrHex`` excluding zero, then within-row ``j``).
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be > 0, got {grid_size}")
    if r_sample <= 0:
        raise ValueError(f"r_sample must be > 0, got {r_sample}")
    if edge_length is None:
        edge_length = grid_size
    if edge_length <= 0:
        raise ValueError(f"edge_length must be > 0, got {edge_length}")

    device = resolve_device(device)
    dtype = resolve_dtype(device, dtype)

    # C L187-L190
    a_large = (2.0 * r_sample) / math.sqrt(3.0)
    ht_triangle = math.sqrt(3.0) * grid_size / 2.0
    nr_hex = int(math.ceil(a_large / grid_size))
    a_last = grid_size * nr_hex

    # C L29-L32: per-row sub-triangle offsets.
    ysmall = ht_triangle / 3.0
    ybig = ht_triangle * (2.0 / 3.0)
    xt1 = edge_length * math.sqrt(3.0) / 6.0
    xt2 = edge_length * math.sqrt(3.0) * 2.0 / 6.0

    rows: list[torch.Tensor] = []

    for i in range(-nr_hex, nr_hex + 1):
        if i == 0:
            continue
        # C L37-L41: starting parity for this row.
        # (i < 0) -> ynext starts at ybig; (i > 0) -> ynext starts at ysmall.
        first_is_big = i < 0
        # C L42
        nr_row_elements = (2 * ((2 * nr_hex) - abs(i))) + 1
        # C L43-L47
        ythis = ht_triangle * i
        xstart = -a_last + (abs(i) * grid_size * 0.5)
        if i < 0:
            xstart += grid_size * 0.5

        # C L49-L65: alternating (xt2, xt1) vs (xt1, xt2) per j.
        # We construct the alternation as boolean masks over j and select.
        j = torch.arange(nr_row_elements, device=device, dtype=dtype)
        # ynext starts at ybig if first_is_big else ysmall, and flips each step.
        # j=0 -> first_is_big. j=1 -> !first_is_big. Etc.
        is_big = ((j.long() % 2) == 0) ^ (not first_is_big)

        dx = torch.where(
            is_big,
            torch.full_like(j, xt2),
            torch.full_like(j, xt1),
        )
        dy = torch.where(
            is_big,
            torch.full_like(j, xt1),
            torch.full_like(j, xt2),
        )
        # x = xstart + (GridSize * j) / 2     (C L56)
        x = xstart + (grid_size * j) / 2.0
        # y = ythis - (ynext * i / |i|)       (C L57)
        # ynext is ybig where is_big else ysmall; sign(i) flips for the
        # negative-i half of the grid.
        ynext = torch.where(
            is_big,
            torch.full_like(j, ybig),
            torch.full_like(j, ysmall),
        )
        sign_i = 1.0 if i > 0 else -1.0
        y = ythis - (ynext * sign_i)
        edge_half = torch.full_like(j, edge_length / 2.0)

        rows.append(torch.stack([dx, dy, x, y, edge_half], dim=1))

    if not rows:
        return torch.empty((0, 5), device=device, dtype=dtype)
    return torch.cat(rows, dim=0)


@dataclass
class HexGrid:
    """Convenience wrapper around the (N, 5) grid tensor."""

    points: torch.Tensor  # (N, 5): dx, dy, x, y, edge_half

    @classmethod
    def make(
        cls,
        grid_size: float,
        r_sample: float,
        edge_length: Optional[float] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> "HexGrid":
        return cls(
            make_hex_grid(
                grid_size,
                r_sample,
                edge_length=edge_length,
                device=device,
                dtype=dtype,
            )
        )

    @classmethod
    def from_params(cls, params, **kwargs) -> "HexGrid":
        return cls.make(
            params.grid_size,
            params.r_sample,
            edge_length=params.edge_length,
            **kwargs,
        )

    def __len__(self) -> int:
        return self.points.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def x(self) -> torch.Tensor:
        return self.points[:, 2]

    @property
    def y(self) -> torch.Tensor:
        return self.points[:, 3]

    @property
    def dx(self) -> torch.Tensor:
        return self.points[:, 0]

    @property
    def dy(self) -> torch.Tensor:
        return self.points[:, 1]

    @property
    def edge_half(self) -> torch.Tensor:
        return self.points[:, 4]

    def write(self, path: Union[str, Path]) -> None:
        from .io import write_grid_txt

        write_grid_txt(self.points, path)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "HexGrid":
        from .io import read_grid_txt

        return cls(read_grid_txt(path))

    def filter(self, mask: torch.Tensor) -> "HexGrid":
        """Return a new HexGrid keeping only points where ``mask`` is True."""
        return HexGrid(self.points[mask])
