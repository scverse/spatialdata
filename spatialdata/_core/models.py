"""This file contains models and schema for SpatialData"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from pandera import Field, SchemaModel
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries
from xarray_dataclasses import AsDataArray, Data
from xarray_schema import DataArraySchema

Labels_t = np.int_
Image_t = np.float_
Table_X_t = np.float_

X, X_t = "x", Literal["x"]
Y, Y_t = "y", Literal["y"]
C, C_t = "c", Literal["c"]
Z, Z_t = "z", Literal["z"]

Labels2D_s = DataArraySchema(dtype=Labels_t, dims=(Y, X))
Image2D_s = DataArraySchema(dtype=Image_t, dims=(C, Y, X))

Labels3D_s = DataArraySchema(dtype=Labels_t, dims=(Z, Y, X))
Image3D_s = DataArraySchema(dtype=Image_t, dims=(C, Z, Y, X))


@dataclass
class Labels2D(AsDataArray):
    """2D Label as DataArray."""

    data: Data[Tuple[Y_t, X_t], Labels_t]


@dataclass
class Image2D(AsDataArray):
    """2D Image as DataArray."""

    data: Data[Tuple[C_t, Y_t, X_t], Image_t]


@dataclass
class Labels3D(AsDataArray):
    """3D Label as DataArray."""

    data: Data[Tuple[Z_t, Y_t, X_t], Labels_t]


@dataclass
class Image3D(AsDataArray):
    """3D Image as DataArray."""

    data: Data[Tuple[C_t, Z_t, Y_t, X_t], Labels_t]


class CirclesSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)
    radius: Series[int] = Field(coerce=True)


class SquareSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)
    sides: Optional[Series[int]] = Field(coerce=True)


class PolygonSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)


# Points (AnnData)?
# Tables (AnnData)?
