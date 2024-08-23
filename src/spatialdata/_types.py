from __future__ import annotations

from typing import Union

import numpy as np
from datatree import DataTree
from xarray import DataArray

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float64]
except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc]

Raster_T = Union[DataArray, DataTree]
ColorLike = Union[tuple[float, ...], str]
