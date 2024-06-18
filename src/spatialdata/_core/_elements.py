"""SpatialData elements."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Iterable
from typing import Any
from warnings import warn

from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame

from spatialdata._types import Raster_T
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
    get_axes_names,
    get_model,
)


class Elements(UserDict[str, Any]):
    def __init__(self, shared_keys: set[str | None]) -> None:
        self._shared_keys = shared_keys
        super().__init__()

    @staticmethod
    def _check_valid_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, not {type(name).__name__}.")
        if len(name) == 0:
            raise ValueError("Name cannot be an empty string.")
        if not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("Name must contain only alphanumeric characters, underscores, and hyphens.")

    @staticmethod
    def _check_key(key: str, element_keys: Iterable[str], shared_keys: set[str | None]) -> None:
        Elements._check_valid_name(key)
        if key in element_keys:
            warn(f"Key `{key}` already exists. Overwriting it in-memory.", UserWarning, stacklevel=2)
        else:
            if key in shared_keys:
                raise KeyError(f"Key `{key}` already exists.")

    def __setitem__(self, key: str, value: Any) -> None:
        self._shared_keys.add(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._shared_keys.remove(key)
        super().__delitem__(key)


class Images(Elements):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
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
        self._check_key(key, self.keys(), self._shared_keys)
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
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != ShapesModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ShapesModel().validate(value)
        super().__setitem__(key, value)


class Points(Elements):
    def __setitem__(self, key: str, value: DaskDataFrame) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != PointsModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        PointsModel().validate(value)
        super().__setitem__(key, value)


class Tables(Elements):
    def __setitem__(self, key: str, value: AnnData) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != TableModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        TableModel().validate(value)
        super().__setitem__(key, value)
