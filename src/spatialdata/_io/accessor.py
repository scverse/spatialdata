from collections.abc import MutableMapping
from typing import Any

import dask.dataframe as dd
from dask.dataframe.extensions import register_dataframe_accessor, register_series_accessor


class _AttrsBase(MutableMapping):
    """Base accessor that stores arbitrary metadata on Dask objects."""

    def __init__(self, dask_obj):
        self._obj = dask_obj
        if not hasattr(dask_obj, "_attrs"):
            dask_obj._attrs = {}

    def __getitem__(self, key):
        return self._obj._attrs[key]

    def __setitem__(self, key, value):
        self._obj._attrs[key] = value

    def __delitem__(self, key):
        del self._obj._attrs[key]

    def __iter__(self):
        return iter(self._obj._attrs)

    def __len__(self):
        return len(self._obj._attrs)

    def __repr__(self):
        return repr(self._obj._attrs)

    def __str__(self):
        return str(self._obj._attrs)

    def copy(self):
        return self._obj._attrs.copy()

    @property
    def data(self):
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


def wrap_with_attrs(method: Any):
    """Wrap a Dask DataFrame method to preserve _attrs.

    Copies _attrs from self before calling method, then assigns to result.
    Safe for lazy operations like set_index, assign, map_partitions.
    """

    def wrapper(self, *args, **kwargs):
        old_accessor = getattr(self, "attrs", {})
        if hasattr(old_accessor, "_obj") and hasattr(old_accessor._obj, "_attrs"):
            old_attrs = old_accessor._obj._attrs.copy()
        elif isinstance(old_accessor, dict):
            old_attrs = old_accessor.copy()
        else:
            old_attrs = {}
        result = method(self, *args, **kwargs)
        result.attrs = old_attrs
        return result

    return wrapper


def wrap_loc_with_attrs():
    """Patch dd.DataFrame.loc to preserve _attrs."""
    original_property = dd.DataFrame.loc  # this is a property

    @property
    def loc_with_attrs(self):
        df = self
        loc = original_property.fget(df)

        class LocWrapper:
            def __init__(self, parent_loc, parent_df):
                self._parent_loc = parent_loc
                self._parent_df = parent_df

            def __getitem__(self, key):
                result = self._parent_loc[key]
                if hasattr(self._parent_df, "_attrs"):
                    result.attrs = self._parent_df._attrs.copy()
                return result

            def __setitem__(self, key, value):
                # preserve attrs even if user assigns via .loc
                self._parent_loc[key] = value
                return self._parent_df

            def __repr__(self):
                return repr(self._parent_loc)

        return LocWrapper(loc, df)

    dd.DataFrame.loc = loc_with_attrs


def wrap_iloc_with_attrs():
    """Patch dd.DataFrame.iloc to preserve _attrs."""
    original_property = dd.DataFrame.iloc  # this is a property

    @property
    def iloc_with_attrs(self):
        df = self
        iloc = original_property.fget(df)

        class ILocWrapper:
            def __init__(self, parent_iloc, parent_df):
                self._parent_iloc = parent_iloc
                self._parent_df = parent_df

            def __getitem__(self, key):
                result = self._parent_iloc[key]
                if hasattr(self._parent_df, "_attrs"):
                    result.attrs = self._parent_df._attrs.copy()
                return result

            def __setitem__(self, key, value):
                self._parent_iloc[key] = value
                return self._parent_df

            def __repr__(self):
                return repr(self._parent_iloc)

        return ILocWrapper(iloc, df)

    dd.DataFrame.iloc = iloc_with_attrs


methods_to_wrap = [
    "set_index",
    "compute",
    "drop",
    "__getitem__",
    "copy",
    "cat",
    "map_partitions",
    # "merge",
    # "join",
    # "repartition",
]

for method_name in methods_to_wrap:
    if hasattr(dd.DataFrame, method_name):
        original_method = getattr(dd.DataFrame, method_name)
        setattr(dd.DataFrame, method_name, wrap_with_attrs(original_method))

wrap_loc_with_attrs()
wrap_iloc_with_attrs()
