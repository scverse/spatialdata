from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Callable

import dask.array as da
from dask.array import Array
from dask.array.overlap import coerce_depth
from multiscale_spatial_image import MultiscaleSpatialImage
from numpy.typing import NDArray
from spatial_image import SpatialImage

import spatialdata
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

__all__ = ["map_raster"]


def map_raster(
    data: SpatialImage | MultiscaleSpatialImage,
    func: Callable,
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    chunks: str | int | tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
    output_chunks: tuple[tuple[int, ...], ...] | None = None,
    depth: str | int | tuple[int, ...] | dict[int:int] | None = None,
    scale_factors: ScaleFactors_t | None = None,  # if specified will return multiscale
    c_coords: int | str | Iterable[int | str] | None = None,
    **kwargs,
) -> SpatialImage | MultiscaleSpatialImage:
    """
    Apply a function to raster data.

    Parameters
    ----------
    data
        The data to process. Can be a `SpatialImage` or `MultiscaleSpatialImage`.
    func
        The function to apply to the data.
    fn_kwargs
        Additional keyword arguments to pass to the function `func`.
    chunks
        If specified, data will be rechunked and processed via `dask.array.map_blocks` or `dask.array.map_overlap`.
        If `None`, `func` is applied to the data without use of `dask.array.map_blocks`/`dask.array.map_overlap`.
    output_chunks
        Chunk shape of resulting blocks if the function does not preserve
        shape. If not provided, the resulting array is assumed to have the same
        block structure as the first input array.
        Passed to `dask.array.map_overlap`/`dask.array.map_blocks` as `chunks`.
        Ignored when `chunks` is `None`.
        E.g. ( (3,), (256,) , (256,)  ).
    depth
        If not `None` and `chunks` is not `None`, will use `dask.array.map_overlap` for distributed processing.
        Specifies the number of elements that each block should share with its neighbors
    scale_factors
        If specified, the function returns a `MultiscaleSpatialImage`.
    c_coords
        Can be used to set the channel coordinates for the output data.
        If the number of channels is altered, `c_coords` should match the output dimension.
    kwargs
        Additional keyword arguments to pass to `dask.array.map_overlap` or `dask.array.map_blocks`.

    Returns
    -------
    The processed data. If `scale_factors` is provided, returns a `MultiscaleSpatialImage`, else `SpatialImage`.

    Notes
    -----
    The transformations of the input data are preserved and applied to the output data.
    """

    def _map_func(
        func: Callable[..., NDArray | Array],
        arr: NDArray | Array,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> Array:
        if chunks is None:
            # if dask array, we want to rechunk
            if isinstance(arr, Array):
                arr = arr.rechunk(arr.chunksize)
            arr = func(arr, **fn_kwargs)
            arr = da.asarray(arr)
            # func could have cause irregular chunking
            return arr.rechunk(arr.chunksize)
        if output_chunks is not None:
            kwargs["chunks"] = output_chunks
        arr = da.asarray(arr).rechunk(chunks)
        if depth is not None:
            kwargs.setdefault("boundary", "reflect")

            if not isinstance(depth, int) and len(depth) != arr.ndim:
                raise ValueError(
                    f"Depth ({depth}) is provided for {len(depth)} dimensions. "
                    f"Please (only) provide depth for {arr.ndim} dimensions."
                )

            kwargs["depth"] = coerce_depth(arr.ndim, depth)

            arr = da.map_overlap(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        else:
            arr = da.map_blocks(func, arr, **fn_kwargs, **kwargs, dtype=arr.dtype)
        # not sure if we want to rechunk here; it fixes irregular chunk sizes, necessary when wanting to save to zarr
        return arr.rechunk(arr.chunksize)

    # pass transformations as parameter to map_raster?
    # If transformations is not None, then we can use this transformation when parsing dask array
    # necessary if dimension is altered of spatialimage (via output_chunks parameter)
    transformations = get_transformation(data, get_all=True)

    if isinstance(data, SpatialImage):
        arr = data.data
    elif isinstance(data, MultiscaleSpatialImage):
        scale_0 = data.__iter__().__next__()
        name = data[scale_0].__iter__().__next__()
        data = data[scale_0][name]
        arr = data.data
    else:
        raise ValueError("Currently only supports 'SpatialImage' and 'MultiscaleSpatialImage'.")

    arr = _map_func(func=func, arr=arr, fn_kwargs=fn_kwargs)

    # should we add this line? if added, user needs to pass c_coords when nr of channels is altered,
    # but doing this, allows users to not pass c_coords, and still c_coords are preserveed.
    # probably remove, user can just copy coordinates from input image
    # if c_coords is None:
    #    c_coords = se.c.data

    if "z" in data.dims:
        data = spatialdata.models.Image3DModel.parse(
            arr,
            dims=data.dims,  # currently does not allow changing dims, we could allow passing dims to map_raster
            scale_factors=scale_factors,
            chunks=arr.chunksize,
            c_coords=c_coords,  # Note that if c_coords is not None, it should match the output channels.
            transformations=transformations,
        )

    else:
        data = spatialdata.models.Image2DModel.parse(
            arr,
            dims=data.dims,
            scale_factors=scale_factors,
            chunks=arr.chunksize,
            c_coords=c_coords,
            transformations=transformations,
        )

    return data
