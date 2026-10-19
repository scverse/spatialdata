from __future__ import annotations

from typing import Any

import numpy as np
from xarray import DataArray, DataTree

__all__ = ["ArrayLike", "ColorLike", "DTypeLike", "JSONValue", "Raster_T"]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = int | float

ListOrNDArrayFloating = list[Number] | ArrayLike


type Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str

# A value that survives a round-trip through JSON, which is the invariant that `SpatialData.attrs` must satisfy: the
# attrs are persisted with `zarr.Group.attrs.put()`, which rejects anything that is not JSON-serializable (e.g. numpy
# arrays, sets, DataFrames). Note that JSON has no tuples and only string keys, so a tuple is read back as a list and
# a non-string key as a string.
type JSONValue = dict[str, JSONValue] | list[JSONValue] | str | int | float | bool | None
