from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from xarray import DataArray, DataTree

__all__ = [
    "ArrayLike",
    "ColorLike",
    "DTypeLike",
    "Raster_T",
    "ELEMENT_TYPE",
    "ELEMENT_TYPE_RASTER",
    "ELEMENT_TYPE_VECTOR",
    "GROUP_NAME",
]

from numpy.typing import DTypeLike, NDArray

ArrayLike = NDArray[np.floating[Any]]
IntArrayLike = NDArray[np.integer[Any]]

type Raster_T = DataArray | DataTree
ColorLike = tuple[float, ...] | str


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
