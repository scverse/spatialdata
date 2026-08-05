from __future__ import annotations

from typing import Any

import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

type Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str
