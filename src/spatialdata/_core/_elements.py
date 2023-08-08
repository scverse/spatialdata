"""SpatialData elements."""
from __future__ import annotations

from collections import UserDict
from typing import Union
from warnings import warn

from dask.dataframe.core import DataFrame
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame

from spatialdata._types import Raster_T
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    get_axes_names,
    get_model,
)


class Elements(UserDict[str, Union[Raster_T, GeoDataFrame, DaskDataFrame]]):
    def __setitem__(self, key: str, value: Raster_T | GeoDataFrame | DataFrame) -> None:
        if key in self:
            warn(f"Key `{key}` already exists.", UserWarning, stacklevel=2)
        super().__setitem__(key, value)


class Images(Elements):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        schema = get_model(value)
        if schema not in (Image2DModel, Image3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ndim = len(get_axes_names(value))
        if ndim == 3:
            Image2DModel().validate(value)
            super().__setitem__(key, value)
        elif ndim == 4:
            Image3DModel().validate(value)
            super().__setitem__(key, value)
        else:
            NotImplementedError("TODO: implement for ndim > 4.")


class Labels(Elements):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        schema = get_model(value)
        if schema not in (Labels2DModel, Labels3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ndim = len(get_axes_names(value))
        if ndim == 2:
            Labels2DModel().validate(value)
            super().__setitem__(key, value)
        elif ndim == 3:
            Labels3DModel().validate(value)
            super().__setitem__(key, value)
        else:
            NotImplementedError("TODO: implement for ndim > 3.")


class Shapes(Elements):
    def __setitem__(self, key: str, value: GeoDataFrame) -> None:
        schema = get_model(value)
        if schema != ShapesModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ShapesModel().validate(value)
        super().__setitem__(key, value)


class Points(Elements):
    def __setitem__(self, key: str, value: DaskDataFrame) -> None:
        schema = get_model(value)
        if schema != PointsModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        PointsModel().validate(value)
        super().__setitem__(key, value)
