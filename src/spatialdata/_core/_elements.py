"""SpatialData elements."""
from __future__ import annotations

from collections import UserDict

from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._types import Raster_T
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    get_model,
)


class Images(UserDict[str, Raster_T]):
    def __setitem__(self, key: str, value: SpatialImage | MultiscaleSpatialImage) -> None:
        schema = get_model(value)
        if schema in (Image2DModel, Image3DModel):
            if key in self:
                raise KeyError(f"Key `{key}` already exists.")
            super().__setitem__(key, value)


class Labels(UserDict[str, Raster_T]):
    def __setitem__(self, key: str, value: SpatialImage | MultiscaleSpatialImage) -> None:
        schema = get_model(value)
        if schema in (Labels2DModel, Labels3DModel):
            if key in self:
                raise KeyError(f"Key {key} already exists.")
            super().__setitem__(key, value)


class Shapes(UserDict[str, GeoDataFrame]):
    def __setitem__(self, key: str, value: GeoDataFrame) -> None:
        schema = get_model(value)
        if schema == ShapesModel:
            if key in self:
                raise KeyError(f"Key {key} already exists.")
            super().__setitem__(key, value)


class Points(UserDict[str, DaskDataFrame]):
    def __setitem__(self, key: str, value: DaskDataFrame) -> None:
        schema = get_model(value)
        if schema == PointsModel:
            if key in self:
                raise KeyError(f"Key {key} already exists.")
            super().__setitem__(key, value)
