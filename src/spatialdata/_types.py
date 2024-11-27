import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T"]

try:
    from numpy.typing import DTypeLike, NDArray

    ArrayLike = NDArray[np.float64]
    IntArrayLike = NDArray[np.int64]  # or any np.integer

except (ImportError, TypeError):
    ArrayLike = np.ndarray  # type: ignore[misc]
    IntArrayLike = np.ndarray  # type: ignore[misc]
    DTypeLike = np.dtype  # type: ignore[misc, assignment]

Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str
