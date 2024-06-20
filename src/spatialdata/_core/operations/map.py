from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Callable

import dask.array as da
from dask.array.overlap import coerce_depth
from datatree import DataTree
from xarray import DataArray

from spatialdata.models._utils import get_axes_names, get_channels, get_raster_model_from_data_dims
from spatialdata.models.models import Labels2DModel, Labels3DModel, get_model
from spatialdata.transformations import get_transformation

__all__ = ["map_raster"]


def map_raster(
    data: DataArray | DataTree,
    func: Callable[[da.Array], da.Array],
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    blockwise: bool = True,
    depth: int | tuple[int, ...] | dict[int, int] | None = None,
    chunks: tuple[tuple[int, ...], ...] | None = None,
    c_coords: Iterable[int] | Iterable[str] | None = None,
    dims: tuple[str, ...] | None = None,
    transformations: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Apply a function to raster data.

    Parameters
    ----------
    data
        The data to process. It can be a `DataArray` or `DataTree`. If it's a `DataTree`,
        the function is applied to the first scale (full-resolution data).
    func
        The callable that is applied to the data.
    fn_kwargs
        Additional keyword arguments to pass to the function `func`.
    blockwise
        If `True`, distributed processing will be achieved with `dask.array.map_overlap`/`dask.array.map_blocks`,
        otherwise the function is applied to the full data. If `False`, `depth` and `chunks` are ignored.
    depth
        Specifies the overlap between chunks, i.e. the number of elements that each chunk
        should share with its neighbor chunks. If not `None`, distributed processing will be achieved with 
        `dask.array.map_overlap`, otherwise with `dask.array.map_blocks`.  Please see 
        :func:`dask.array.map_overlap` for more information on the accepted values.
    chunks
        Passed to `dask.array.map_overlap`/`dask.array.map_blocks` as `chunks`. Ignored if `blockwise` is `False`.
        Chunk shape of resulting blocks if the function does not preserve the data shape. If not provided, the resulting
        array is assumed to have the same chunk structure as the first input array.
        E.g. ( (3,), (100,100), (100,100) ).
    c_coords
        The channel coordinates for the output data. If not provided, the channel coordinates of the input data are
        used. It should be specified if the function changes the number of channels.
    dims
        The dimensions of the output data. If not provided, the dimensions of the input data are used. It must be
        specified if the function changes the data dimensions.
        E.g. ('c', 'y', 'x').
    transformations
        The transformations of the output data. If not provided, the transformations of the input data are copied to the
        output data. It should be specified if the function changes the data transformations.
    kwargs
        Additional keyword arguments to pass to `dask.array.map_overlap` or `dask.array.map_blocks`.

    Returns
    -------
    The processed data as a `DataArray`.
    """
    if isinstance(data, DataArray):
        arr = data.data
    elif isinstance(data, DataTree):
        arr = data["scale0"].values().__iter__().__next__().data
    else:
        raise ValueError("Only 'DataArray' and 'DataTree' are supported.")

    model = get_model(data)
    if model in (Labels2DModel, Labels3DModel) and c_coords is not None:
        raise ValueError("Channel coordinates can not be provided for labels data.")

    kwargs = kwargs.copy()
    kwargs["chunks"] = chunks

    if not blockwise:
        arr = func(arr, **fn_kwargs)
    else:
        if depth is not None:
            kwargs.setdefault("boundary", "reflect")

            if not isinstance(depth, int) and len(depth) != arr.ndim:
                raise ValueError(
                    f"Depth {depth} is provided for {len(depth)} dimensions. "
                    f"Please (only) provide depth for {arr.ndim} dimensions."
                )
            kwargs["depth"] = coerce_depth(arr.ndim, depth)
            map_func = da.map_overlap
        else:
            map_func = da.map_blocks

        arr = map_func(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)

    dims = dims if dims is not None else get_axes_names(data)
    if model not in (Labels2DModel, Labels3DModel):
        c_coords = c_coords if c_coords is not None else get_channels(data)
    else:
        c_coords = None
    if transformations is None:
        d = get_transformation(data, get_all=True)
        assert isinstance(d, dict)
        transformations = d

    model_kwargs = {
        "chunks": arr.chunksize,
        "c_coords": c_coords,
        "dims": dims,
        "transformations": transformations,
    }
    model = get_raster_model_from_data_dims(dims)
    return model.parse(arr, **model_kwargs)
