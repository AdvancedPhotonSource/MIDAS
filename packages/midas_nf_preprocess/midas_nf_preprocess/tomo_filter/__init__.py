"""tomo_filter: mask a hex grid by a tomography image.

Port of ``filterGridfromTomo.c``. For each grid point at ``(x_um, y_um)``,
look up the corresponding pixel in a 2D tomography mask; keep the point if
the pixel is non-zero.

Coordinate convention (from filterGridfromTomo.c L39-L43):

    xPos = int(x_um / pxTomo) + nrPxTomo // 2
    yPos = int(y_um / pxTomo) + nrPxTomo // 2
    keep = (0 <= xPos < nrPxTomo
            and 0 <= yPos < nrPxTomo
            and tomo[nrPxTomo - yPos, xPos] != 0)

The Y axis is flipped (``nrPxTomo - yPos``) so that grid +y points up in the
image, matching the C code.

Beyond the C semantics, the Python module also offers in-memory filtering
on torch tensors (no file I/O).
"""

from .filter import (
    bbox_mask,
    filter_grid_by_tomo,
    filter_grid_by_bbox,
    load_square_tomo,
    sample_tomo,
)

__all__ = [
    "bbox_mask",
    "filter_grid_by_tomo",
    "filter_grid_by_bbox",
    "load_square_tomo",
    "sample_tomo",
]
