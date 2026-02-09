"""SpatialData elements."""

from collections import UserDict
from collections.abc import Iterable, KeysView, ValuesView
from typing import TypeVar

from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata._core.validation import check_key_is_case_insensitively_unique, check_valid_name
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

T = TypeVar("T")


class Elements(UserDict[str, T]):
    def __init__(self, shared_keys: set[str | None]) -> None:
        self._shared_keys = shared_keys
        super().__init__()

    def _add_shared_key(self, key: str) -> None:
        self._shared_keys.add(key)

    def _remove_shared_key(self, key: str) -> None:
        self._shared_keys.remove(key)

    @staticmethod
    def _check_key(key: str, element_keys: Iterable[str], shared_keys: set[str | None]) -> None:
        check_valid_name(key)
        if key not in element_keys:
            try:
                check_key_is_case_insensitively_unique(key, shared_keys)
            except ValueError as e:
                # Validation raises ValueError, but inappropriate mapping key must raise KeyError.
                raise KeyError(*e.args) from e

    def __setitem__(self, key: str, value: T) -> None:
        self._add_shared_key(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._remove_shared_key(key)
        super().__delitem__(key)

    def keys(self) -> KeysView[str]:
        """Return the keys of the Elements."""
        return self.data.keys()

    def values(self) -> ValuesView[T]:
        """Return the values of the Elements."""
        return self.data.values()


class Images(Elements[DataArray | DataTree]):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema not in (Image2DModel, Image3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ndim = len(get_axes_names(value))
        if ndim == 3:
            Image2DModel.validate(value)
            super().__setitem__(key, value)
        elif ndim == 4:
            Image3DModel.validate(value)
            super().__setitem__(key, value)
        else:
            NotImplementedError("TODO: implement for ndim > 4.")


class Labels(Elements[DataArray | DataTree]):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema not in (Labels2DModel, Labels3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ndim = len(get_axes_names(value))
        if ndim == 2:
            Labels2DModel.validate(value)
            super().__setitem__(key, value)
        elif ndim == 3:
            Labels3DModel.validate(value)
            super().__setitem__(key, value)
        else:
            NotImplementedError("TODO: implement for ndim > 3.")


class Shapes(Elements[GeoDataFrame]):
    def __setitem__(self, key: str, value: GeoDataFrame) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != ShapesModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        ShapesModel.validate(value)
        super().__setitem__(key, value)


class Points(Elements[DaskDataFrame]):
    def __setitem__(self, key: str, value: DaskDataFrame) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != PointsModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        PointsModel.validate(value)
        super().__setitem__(key, value)


class Tables(Elements[AnnData]):
    def __setitem__(self, key: str, value: AnnData) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value)
        if schema != TableModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        TableModel.validate(value)
        super().__setitem__(key, value)
