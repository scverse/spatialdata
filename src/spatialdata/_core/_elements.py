"""SpatialData elements."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any
from warnings import warn

from dask.dataframe.core import DataFrame as DaskDataFrame
from datatree import DataTree
from geopandas import GeoDataFrame

from spatialdata._types import Raster_T
from spatialdata._utils import multiscale_spatial_image_from_data_tree
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


class Elements(OrderedDict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self._shared_keys: set[str] = set()

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._shared_keys:
            warn(f"Key `{key}` already exists.", UserWarning, stacklevel=2)
        super().__setitem__(key, value)
        self._shared_keys.add(key)


class Images(Elements):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        if isinstance(value, (DataTree)):
            value = multiscale_spatial_image_from_data_tree(value)
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
        if isinstance(value, (DataTree)):
            value = multiscale_spatial_image_from_data_tree(value)
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
