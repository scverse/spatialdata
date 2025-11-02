from collections.abc import Iterator, MutableMapping
from typing import Any, Literal

from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import Series as DaskSeries
from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)


@register_dataframe_accessor("attrs")
@register_series_accessor("attrs")
class AttrsAccessor(MutableMapping[str, str | dict[str, Any]]):
    """Accessor that stores a dict of arbitrary metadata on Dask objects."""

    def __init__(self, dask_obj: DaskDataFrame | DaskSeries):
        self._obj = dask_obj
        if not hasattr(dask_obj, "_attrs"):
            dask_obj._attrs = {}

    def __getitem__(self, key: str) -> Any:
        return self._obj._attrs[key]

    def __setitem__(self, key: str, value: str | dict[str, Any]) -> None:
        self._obj._attrs[key] = value

    def __delitem__(self, key: str) -> None:
        del self._obj._attrs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._obj._attrs)

    def __len__(self) -> int:
        return len(self._obj._attrs)

    def __repr__(self) -> str:
        return repr(self._obj._attrs)

    def __str__(self) -> str:
        return str(self._obj._attrs)

    def copy(self) -> Any:
        return self._obj._attrs.copy()

    @property
    def data(self) -> Any:
        """Access the raw internal attrs dict."""
        return self._obj._attrs


def wrap_method_with_attrs(method_name: str, dask_class: type[DaskDataFrame] | type[DaskSeries]) -> None:
    """Wrap a Dask DataFrame method to preserve _attrs.

    Copies _attrs from self before calling method, then assigns to result.
    Safe for lazy operations like set_index, assign, map_partitions.
    """
    original_method = getattr(dask_class, method_name)

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.attrs, AttrsAccessor):
            raise RuntimeError(
                "Invalid .attrs: expected an accessor (`AttrsAccessor`), "
                f"got {type(self.attrs).__name__}. A common cause is assigning a dict, e.g. "
                "my_dd_object.attrs = {...}. Do not assign to 'attrs'; use "
                "my_dd_object.attrs.update(...) instead."
            )

        old_attrs = self.attrs.copy()
        result = original_method(self, *args, **kwargs)
        # the pandas Index do not have attrs, but dd.Index, since they are a subclass of dd.Series, do have attrs
        # thanks to our accessor. Here we ensure that we do not assign attrs to pd.Index objects.
        if hasattr(result, "attrs"):
            result.attrs.update(old_attrs)

        return result

    setattr(dask_class, method_name, wrapper)


def wrap_indexer_with_attrs(
    indexer_name: Literal["loc", "iloc"], dask_class: type[DaskDataFrame] | type[DaskSeries]
) -> None:
    """Patch dd.DataFrame or dd.Series loc or iloc to preserve _attrs.

    Reason for having this separate from methods is because both loc and iloc are a property that return an indexer.
    Therefore, they have to be wrapped differently from methods in order to preserve attrs.
    """
    original_property = getattr(dask_class, indexer_name)  # this is a property

    def indexer_with_attrs(self: DaskDataFrame | DaskSeries) -> Any:
        parent_obj = self
        indexer = original_property.fget(parent_obj)

        class IndexerWrapper:
            def __init__(self, parent_indexer: Any, parent_obj: DaskDataFrame | DaskSeries) -> None:
                self._parent_indexer = parent_indexer
                self._parent_obj = parent_obj

            def __getitem__(self, key: str) -> Any:
                result = self._parent_indexer[key]
                if hasattr(self._parent_obj, "attrs"):
                    result._attrs = self._parent_obj.attrs.copy()
                return result

            def __setitem__(self, key: str, value: Any) -> DaskDataFrame | DaskSeries:
                # preserve attrs even if user assigns via .loc
                self._parent_indexer[key] = value
                return self._parent_obj

            def __repr__(self) -> str:
                return repr(self._parent_indexer)

        return IndexerWrapper(indexer, parent_obj)

    setattr(dask_class, indexer_name, property(indexer_with_attrs))


for method_name in [
    "__getitem__",
    "compute",
    "copy",
    "drop",
    "map_partitions",
    "set_index",
]:
    wrap_method_with_attrs(method_name=method_name, dask_class=DaskDataFrame)

for method_name in [
    "__getitem__",
    "compute",
    "copy",
    "map_partitions",
]:
    wrap_method_with_attrs(method_name=method_name, dask_class=DaskSeries)

wrap_indexer_with_attrs(indexer_name="loc", dask_class=DaskDataFrame)
wrap_indexer_with_attrs(indexer_name="iloc", dask_class=DaskDataFrame)
wrap_indexer_with_attrs(indexer_name="loc", dask_class=DaskSeries)
# DaskSeries do not have iloc
