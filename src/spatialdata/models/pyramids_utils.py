from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dask.array as da
from ome_zarr.dask_utils import resize
from xarray import DataArray, Dataset, DataTree

from spatialdata.models.chunks_utils import Chunks_t, normalize_chunks

ScaleFactors_t = Sequence[dict[str, int] | int]


def dask_arrays_to_datatree(
    arrays: Sequence[da.Array],
    dims: Sequence[str],
    channels: list[Any] | None = None,
) -> DataTree:
    """Build a multiscale DataTree from a sequence of dask arrays.

    Parameters
    ----------
    arrays
        Sequence of dask arrays, one per scale level (scale0, scale1, ...).
    dims
        Dimension names for the arrays (e.g. ``("c", "y", "x")``).
    channels
        Optional channel coordinate values. If provided, a ``"c"`` coordinate
        is added to each scale level.

    Returns
    -------
    DataTree with one child per scale level.
    """
    if "c" in dims and channels is None:
        raise ValueError("channels must be provided if the image has a channel dimension")
    coords = {"c": channels} if channels is not None else {}
    d = {}
    for i, arr in enumerate(arrays):
        d[f"scale{i}"] = Dataset(
            {
                "image": DataArray(
                    arr,
                    name="image",
                    dims=list(dims),
                    coords=coords,
                )
            }
        )
    return DataTree.from_dict(d)


def to_multiscale(
    image: DataArray,
    scale_factors: ScaleFactors_t,
    chunks: Chunks_t | None = None,
) -> DataTree:
    """Build a multiscale pyramid DataTree from a single-scale image.

    Iteratively downscales the image by the given scale factors using
    interpolation (order 1 for images with a channel dimension, order 0
    for labels) and assembles all levels into a DataTree.

    Makes uses of internal ome-zarr-py APIs for dask downscaling.

    ome-zarr-py will support 3D downscaling once https://github.com/ome/ome-zarr-py/pull/516 is merged, and this
    function could make use of it. Also the PR will introduce new downscaling methods such as "nearest". Nevertheless,
    this function supports different scaling factors per axis, a feature that could be also added to ome-zarr-py.

    TODO: once the PR above is merged, use the new APIs for 3D downscaling and additional downscaling methods
    TODO: once the PR above is merged, consider adding support for per-axis scale factors to ome-zarr-py so that this
     function can be simplified even further.

    Parameters
    ----------
    image
        Input image/labels as an xarray DataArray (e.g. with dims ``("c", "y", "x")``
        or ``("y", "x")``). Supports both 2D/3D images and 2D/3D labels.
    scale_factors
        Sequence of per-level scale factors. Each element is either an int
        (applied to all spatial axes) or a dict mapping dimension names to
        per-axis factors (e.g. ``{"y": 2, "x": 2}``).
    chunks
        Optional chunk specification passed to :meth:`dask.array.Array.rechunk`
        after building the pyramid.

    Returns
    -------
    DataTree
        Multiscale DataTree with children ``scale0``, ``scale1``, etc.
    """
    dims = [str(dim) for dim in image.dims]
    spatial_dims = [d for d in dims if d != "c"]
    order = 1 if "c" in dims else 0
    channels = None if "c" not in dims else image.coords["c"].values
    pyramid = [image.data]
    for sf in scale_factors:
        prev = pyramid[-1]
        # Compute per-axis scale factors: int applies to spatial axes only, dict to specific ones.
        sf_by_axis = dict.fromkeys(dims, 1)
        if isinstance(sf, int):
            sf_by_axis.update(dict.fromkeys(spatial_dims, sf))
        else:
            sf_by_axis.update(sf)
        # skip axes where the scale factor exceeds the axis size.
        for ax, factor in sf_by_axis.items():
            ax_size = prev.shape[dims.index(ax)]
            if factor > ax_size:
                sf_by_axis[ax] = 1
        output_shape = tuple(prev.shape[dims.index(ax)] // f for ax, f in sf_by_axis.items())
        resized = resize(
            image=prev.astype(float),
            output_shape=output_shape,
            order=order,
            mode="reflect",
            anti_aliasing=False,
        )
        pyramid.append(resized.astype(prev.dtype))
    if chunks is not None:
        chunks_dict = normalize_chunks(chunks, axes=dims)
        chunks_tuple = tuple(chunks_dict[d] for d in dims)
        pyramid = [arr.rechunk(chunks_tuple) for arr in pyramid]
    return dask_arrays_to_datatree(pyramid, dims=dims, channels=channels)
