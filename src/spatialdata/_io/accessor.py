from typing import Any

import dask.dataframe as dd
from dask.dataframe.extensions import register_dataframe_accessor, register_series_accessor


@register_dataframe_accessor("attrs")
class DfAttrsAccessor:
    def __init__(self, dask_obj):
        self._obj = dask_obj
        if not hasattr(dask_obj, "_attrs"):
            dask_obj._attrs = {}

    def __getitem__(self, key):
        return self._obj._attrs[key]

    def __setitem__(self, key, value):
        self._obj._attrs[key] = value

    def __iter__(self):
        return iter(self._obj._attrs)

    def __repr__(self):
        return repr(self._obj._attrs)


@register_series_accessor("attrs")
class SeriesAttrsAccessor:
    def __init__(self, dask_obj):
        self._obj = dask_obj
        if not hasattr(dask_obj, "_attrs"):
            dask_obj._attrs = {}

    def __getitem__(self, key):
        return self._obj._attrs[key]

    def __setitem__(self, key, value):
        self._obj._attrs[key] = value

    def __iter__(self):
        return iter(self._obj._attrs)

    def __repr__(self):
        return repr(self._obj._attrs)


def wrap_with_attrs(method: Any):
    """Wrap a Dask DataFrame method to preserve _attrs.

    Copies _attrs from self before calling method, then assigns to result.
    Safe for lazy operations like set_index, assign, map_partitions.
    """

    def wrapper(self, *args, **kwargs):
        old_attrs = getattr(self, "_attrs", {}).copy()
        result = method(self, *args, **kwargs)
        result.attrs = old_attrs
        return result

    return wrapper


methods_to_wrap = [
    "set_index",
    "compute",
    "drop",
    # "assign",
    # "map_partitions",
    # "merge",
    # "join",
    # "repartition",
]

for method_name in methods_to_wrap:
    if hasattr(dd.DataFrame, method_name):
        original_method = getattr(dd.DataFrame, method_name)
        setattr(dd.DataFrame, method_name, wrap_with_attrs(original_method))
