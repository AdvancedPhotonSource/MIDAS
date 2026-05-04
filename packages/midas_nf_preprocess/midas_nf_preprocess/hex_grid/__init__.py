"""hex_grid: voxel grid generation for NF-HEDM (port of MakeHexGrid.c).

The grid is a hexagonal lattice of equilateral-triangle voxels covering a disk
of radius ``Rsample`` centered at the origin. Each voxel carries:

  - ``(x_um, y_um)``     : center coordinates in micrometers
  - ``(dx_um, dy_um)``   : per-vertex offsets used by the forward simulator
  - ``edge_length_um``   : triangle edge length (typically equal to GridSize)
  - ``ud``               : up/down flag (+1 for upward, -1 for downward triangle)

Public API:

  - ``HexGridParams``      : parameter dataclass
  - ``make_hex_grid(...)`` : compute the grid as a torch.Tensor
  - ``HexGrid``            : convenience wrapper (tensor + IO helpers)
  - ``write_grid_txt(...)``, ``read_grid_txt(...)`` : MIDAS grid.txt I/O
"""

from .params import HexGridParams
from .grid import HexGrid, make_hex_grid, n_grid_points
from .io import read_grid_txt, write_grid_txt

__all__ = [
    "HexGridParams",
    "HexGrid",
    "make_hex_grid",
    "n_grid_points",
    "read_grid_txt",
    "write_grid_txt",
]
