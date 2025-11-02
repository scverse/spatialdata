from collections.abc import Iterator, MutableMapping
from typing import Any, Literal, cast

import dask.dataframe as dd
from dask.dataframe.extensions import (
    register_dataframe_accessor,
    register_series_accessor,
)


class _AttrsBase(MutableMapping[str, str | dict[str, str]]):
    """Base accessor that stores arbitrary metadata on Dask objects."""

    def __init__(self, dask_obj: dd.DataFrame | dd.Series):
        self._obj = dask_obj
        if not hasattr(dask_obj, "_attrs"):
            dask_obj._attrs = {}

    def __getitem__(self, key: str) -> Any:
        return self._obj._attrs[key]

    def __setitem__(self, key: str, value: str | dict[str, str]) -> None:
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


@register_dataframe_accessor("attrs")
class DfAttrsAccessor(_AttrsBase):
    """Dict-like .attrs accessor for Dask DataFrames."""

    pass


@register_series_accessor("attrs")
class SeriesAttrsAccessor(_AttrsBase):
    """Dict-like .attrs accessor for Dask Series."""

    pass


def wrap_with_attrs(method_name: str) -> None:
    """Wrap a Dask DataFrame method to preserve _attrs.

    Copies _attrs from self before calling method, then assigns to result.
    Safe for lazy operations like set_index, assign, map_partitions.
    """
    original_method = getattr(dd.DataFrame, method_name)

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(self.attrs, DfAttrsAccessor | SeriesAttrsAccessor):
            raise RuntimeError(
                "Invalid .attrs: expected an accessor (DfAttrsAccessor or SeriesAttrsAccessor), "
                f"got {type(self.attrs).__name__}. A common cause is assigning a dict, e.g. "
                "my_dd_object.attrs = {...}. Do not assign to 'attrs'; use "
                "my_dd_object.attrs.update(...) instead."
            )

        old_attrs = self.attrs.copy()
        result = original_method(self, *args, **kwargs)
        result.attrs.update(old_attrs)
        return result

    setattr(dd.DataFrame, method_name, wrapper)


def wrap_indexer_with_attrs(indexer_name: Literal["loc", "iloc"]) -> None:
    """Patch dd.DataFrame.loc or iloc to preserve _attrs.

    Reason for having this separate from methods is because both loc and iloc are a property that return an indexer.
    Therefore, they have to be wrapped differently from methods in order to preserve attrs.
    """
    original_property = getattr(dd.DataFrame, indexer_name)  # this is a property

    def indexer_with_attrs(self: dd.DataFrame) -> Any:
        df = self
        loc = original_property.fget(df)

        class IndexerWrapper:
            def __init__(self, parent_loc: Any, parent_df: dd.DataFrame) -> None:
                self._parent_loc = parent_loc
                self._parent_df = parent_df

            def __getitem__(self, key: str) -> Any:
                result = self._parent_loc[key]
                if hasattr(self._parent_df, "attrs"):
                    result._attrs = self._parent_df.attrs.copy()
                return result

            def __setitem__(self, key: str, value: Any) -> dd.DataFrame:
                # preserve attrs even if user assigns via .loc
                self._parent_loc[key] = value
                return self._parent_df

            def __repr__(self) -> str:
                return repr(self._parent_loc)

        return IndexerWrapper(loc, df)

    setattr(dd.DataFrame, indexer_name, property(indexer_with_attrs))


for method_name in [
    "set_index",
    "compute",
    "drop",
    "__getitem__",
    "copy",
    "map_partitions",
]:
    wrap_with_attrs(method_name)


for indexer_name in ["loc", "iloc"]:
    wrap_indexer_with_attrs(cast(Literal["loc", "iloc"], indexer_name))
