"""Compatibility layer for dask.array.to_zarr when callers pass array options via **kwargs.

ome_zarr.writer calls da.to_zarr(..., **options) with array options (compressor, dimension_names,
etc.). Dask deprecated **kwargs in favor of zarr_array_kwargs. This module patches da.to_zarr to
forward such kwargs into zarr_array_kwargs (excluding dask-internal keys like zarr_format that
zarr.Group.create_array() does not accept), avoiding the FutureWarning and keeping behavior correct.
"""

from __future__ import annotations

import dask.array as _da

_orig_to_zarr = _da.to_zarr

# Keys from ome_zarr/dask **kwargs that must not be passed to zarr.Group.create_array()
# dimension_separator: not accepted by all zarr versions in the create_array() path.
_DASK_INTERNAL_KEYS = frozenset({"zarr_format", "dimension_separator"})


def _to_zarr(
    arr,
    url,
    component=None,
    storage_options=None,
    region=None,
    compute=True,
    return_stored=False,
    zarr_array_kwargs=None,
    zarr_read_kwargs=None,
    **kwargs,
):
    """Forward deprecated **kwargs into zarr_array_kwargs, excluding _DASK_INTERNAL_KEYS."""
    if kwargs:
        zarr_array_kwargs = dict(zarr_array_kwargs) if zarr_array_kwargs else {}
        for k, v in kwargs.items():
            if k not in _DASK_INTERNAL_KEYS:
                zarr_array_kwargs[k] = v
        kwargs = {}
    return _orig_to_zarr(
        arr,
        url,
        component=component,
        storage_options=storage_options,
        region=region,
        compute=compute,
        return_stored=return_stored,
        zarr_array_kwargs=zarr_array_kwargs,
        zarr_read_kwargs=zarr_read_kwargs,
        **kwargs,
    )


_da.to_zarr = _to_zarr
