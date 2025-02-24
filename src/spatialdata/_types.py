from typing import Any

import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str
