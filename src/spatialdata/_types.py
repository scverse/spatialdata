from typing import Any, TypeAlias

import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

Raster_T: TypeAlias = DataArray | DataTree
ColorLike = tuple[float, ...] | str
