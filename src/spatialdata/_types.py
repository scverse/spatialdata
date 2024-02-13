from __future__ import annotations

from typing import Union

import numpy as np
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

__all__ = ["ArrayLike", "DTypeLike", "Raster_T"]

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

Raster_T = Union[SpatialImage, MultiscaleSpatialImage]
