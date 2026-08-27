from __future__ import annotations

from typing import Any

import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = int | float

ListOrNDArrayFloating = list[Number] | ArrayLike


type Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str
