"""SpatialData elements."""

from __future__ import annotations

import contextlib
from collections import UserDict
from collections.abc import Iterable, Iterator, KeysView, ValuesView
from contextvars import ContextVar
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
    get_model,
)

_skip_element_validation: ContextVar[bool] = ContextVar("_skip_element_validation", default=False)


@contextlib.contextmanager
def skip_element_validation() -> Iterator[None]:
    """
    Context manager to skip schema validation when inserting elements into SpatialData containers.

    Use this only when inserting elements that are already known to be valid (e.g. elements
    taken directly from an existing SpatialData object).  Skipping validation is unsafe for
    externally-sourced data.
    """
    token = _skip_element_validation.set(True)
    try:
        yield
    finally:
        _skip_element_validation.reset(token)


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
        # note that each __setitem__ in the subclasses calls get_model(), which performs data validation
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
        schema = get_model(value, validate=not _skip_element_validation.get())
        if schema not in (Image2DModel, Image3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        super().__setitem__(key, value)


class Labels(Elements[DataArray | DataTree]):
    def __setitem__(self, key: str, value: Raster_T) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value, validate=not _skip_element_validation.get())
        if schema not in (Labels2DModel, Labels3DModel):
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        super().__setitem__(key, value)


class Shapes(Elements[GeoDataFrame]):
    def __setitem__(self, key: str, value: GeoDataFrame) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value, validate=not _skip_element_validation.get())
        if schema != ShapesModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        super().__setitem__(key, value)


class Points(Elements[DaskDataFrame]):
    def __setitem__(self, key: str, value: DaskDataFrame) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value, validate=not _skip_element_validation.get())
        if schema != PointsModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        super().__setitem__(key, value)


class Tables(Elements[AnnData]):
    def __setitem__(self, key: str, value: AnnData) -> None:
        self._check_key(key, self.keys(), self._shared_keys)
        schema = get_model(value, validate=not _skip_element_validation.get())
        if schema != TableModel:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")
        super().__setitem__(key, value)
