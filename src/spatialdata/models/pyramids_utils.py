from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import dask.array as da
from ome_zarr.dask_utils import resize
from xarray import DataArray, Dataset, DataTree

Chunks_t: TypeAlias = int | tuple[int, ...] | tuple[tuple[int, ...], ...] | Mapping[Any, None | int | tuple[int, ...]]
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
        if isinstance(chunks, Mapping):
            chunks = {dims.index(k) if isinstance(k, str) else k: v for k, v in chunks.items()}
        pyramid = [arr.rechunk(chunks) for arr in pyramid]
    return dask_arrays_to_datatree(pyramid, dims=dims, channels=channels)
