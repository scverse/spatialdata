from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

import dask.array as da
from dask.array.overlap import coerce_depth
from datatree import DataTree
from xarray import DataArray

from spatialdata.models._utils import get_axes_names, get_channels, get_raster_model_from_data_dims
from spatialdata.transformations import get_transformation

__all__ = ["map_raster"]


def map_raster(
    data: DataArray | DataTree,
    func: Callable[[da.Array], da.Array],
    func_kwargs: Mapping[str, Any] = MappingProxyType({}),
    blockwise: bool = True,
    depth: int | tuple[int, ...] | dict[int, int] | None = None,
    chunks: tuple[tuple[int, ...], ...] | None = None,
    c_coords: Iterable[int] | Iterable[str] | None = None,
    dims: tuple[str, ...] | None = None,
    transformations: dict[str, Any] | None = None,
    **kwargs: Any,
) -> DataArray:
    """
    Apply a callable to raster data.

    Applies a `func` callable to raster data. If `blockwise` is set to `True`,
    distributed processing will be achieved with:

        - :func:`dask.array.map_overlap` if `depth` is not `None`
        - :func:`dask.array.map_blocks`, if `depth` is `None`

    otherwise `func` is applied to the full data.

    Parameters
    ----------
    data
        The data to process. It can be a :class:`xarray.DataArray` or :class:`datatree.DataTree`.
        If it's a `DataTree`, the callable is applied to the first scale (`scale0`, the full-resolution data).
    func
        The callable that is applied to the data.
    func_kwargs
        Additional keyword arguments to pass to the callable `func`.
    blockwise
        If `True`, `func` will be distributed with :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`,
        otherwise `func` is applied to the full data. If `False`, `depth`, `chunks` and `kwargs` are ignored.
    depth
        Specifies the overlap between chunks, i.e. the number of elements that each chunk
        should share with its neighboring chunks. If not `None`, distributed processing will be achieved with
        :func:`dask.array.map_overlap`, otherwise with :func:`dask.array.map_blocks`.
    chunks
        Chunk shape of resulting blocks if the callable does not preserve the data shape.
        For example, if the input block has `shape: (3,100,100)` and the resulting block after the `map_raster`
        call has `shape: (1, 100,100)`, the argument `chunks` should be passed accordingly.
        Passed to :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`. Ignored if `blockwise` is `False`.
    c_coords
        The channel coordinates for the output data. If not provided, the channel coordinates of the input data are
        used. If the callable `func` is expected to change the number of channel coordinates,
        this argument should be provided, otherwise will default to `range(len(output_coords))`.
    dims
        The dimensions of the output data. If not provided, the dimensions of the input data are used. It must be
        specified if the callable changes the data dimensions, e.g. `('c', 'y', 'x') -> ('y', 'x')`.
    transformations
        The transformations of the output data. If not provided, the transformations of the input data are copied to the
        output data. It should be specified if the callable changes the data transformations.
    kwargs
        Additional keyword arguments to pass to :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`.
        Ignored if `blockwise` is set to `False`.

    Returns
    -------
    The processed data as a :class:`xarray.DataArray`.
    """
    if isinstance(data, DataArray):
        arr = data.data
    elif isinstance(data, DataTree):
        arr = data["scale0"].values().__iter__().__next__().data
    else:
        raise ValueError("Only 'DataArray' and 'DataTree' are supported.")

    dims = dims if dims is not None else get_axes_names(data)

    if "c" not in dims and c_coords is not None:
        raise ValueError(
            "Channel coordinates `c_coords` can not be provided if output data consists of labels "
            "('c' channel missing)."
        )

    kwargs = kwargs.copy()
    kwargs["chunks"] = chunks

    if not blockwise:
        arr = func(arr, **func_kwargs)
    else:
        if depth is not None:
            kwargs.setdefault("boundary", "reflect")

            if not isinstance(depth, int) and len(depth) != arr.ndim:
                raise ValueError(
                    f"Depth {depth} is provided for {len(depth)} dimensions. "
                    f"Please provide depth for {arr.ndim} dimensions."
                )
            kwargs["depth"] = coerce_depth(arr.ndim, depth)
            map_func = da.map_overlap
        else:
            map_func = da.map_blocks

        arr = map_func(func, arr, **func_kwargs, **kwargs)

    if arr.ndim != len(dims):
        raise ValueError(
            f"The number of dimensions of the output data ({arr.ndim}) "
            f"differs from the number of dimensions in 'dims' ({dims}). "
            "Please provide correct output dimension via the 'dims' parameter."
        )

    if "c" in dims:
        if c_coords is None:
            c_coords = range(arr.shape[0]) if arr.shape[0] != len(get_channels(data)) else get_channels(data)
    else:
        c_coords = None
    if transformations is None:
        d = get_transformation(data, get_all=True)
        if TYPE_CHECKING:
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
