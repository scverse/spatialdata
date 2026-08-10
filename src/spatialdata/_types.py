from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from xarray import DataArray, DataTree

__all__ = [
    "ArrayLike",
    "ColorLike",
    "DTypeLike",
    "JSONValue",
    "Raster_T",
    "ELEMENT_TYPE",
    "ELEMENT_TYPE_RASTER",
    "ELEMENT_TYPE_VECTOR",
    "GROUP_NAME",
]

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


class ELEMENT_TYPE(StrEnum):
    IMAGE = "image"
    LABELS = "labels"
    SHAPES = "shapes"
    POINTS = "points"
    TABLES = "tables"


class GROUP_NAME(StrEnum):
    IMAGES = "images"
    LABELS = "labels"
    SHAPES = "shapes"
    POINTS = "points"
    TABLES = "tables"


ELEMENT_TYPE_RASTER = Literal[ELEMENT_TYPE.IMAGE, ELEMENT_TYPE.LABELS]
ELEMENT_TYPE_VECTOR = Literal[ELEMENT_TYPE.POINTS, ELEMENT_TYPE.SHAPES]
