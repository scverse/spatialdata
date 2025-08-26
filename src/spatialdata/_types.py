from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import zarr.storage
from upath import UPath
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "Raster_T", "StoreLike"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str

StoreLike: TypeAlias = str | Path | UPath | zarr.storage.StoreLike | zarr.Group
