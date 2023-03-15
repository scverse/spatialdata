from __future__ import annotations

from collections.abc import Generator
from typing import Union

import numpy as np
from dask import array as da
from datatree import DataTree
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.spatialdata_operations import (
    get_transformation,
    set_transformation,
)
from spatialdata._core.transformations import Sequence, Translation


def unpad_raster(raster: Union[SpatialImage, MultiscaleSpatialImage]) -> Union[SpatialImage, MultiscaleSpatialImage]:
    """
    Remove padding from a raster type that was eventually added by the rotation component of a transformation.

    Parameters
    ----------
    raster
        The raster to unpad. Contiguous zero values are considered padding.

    Returns
    -------
    The unpadded raster.
    """
    from spatialdata._core.core_utils import compute_coordinates, get_axis_names

    def _compute_paddings(data: DataArray, axis: str) -> tuple[int, int]:
        others = list(data.dims)
        others.remove(axis)
        # mypy (luca's pycharm config) can't see the isclose method of dask array
        s = da.isclose(data.sum(dim=others), 0)  # type: ignore[attr-defined]
        # TODO: rewrite this to use dask array; can't get it to work with it
        x = s.compute()
        non_zero = np.where(x == 0)[0]
        if len(non_zero) == 0:
            min_coordinate, max_coordinate = data.coords[axis].min().item(), data.coords[axis].max().item()
            if not min_coordinate != 0:
                raise ValueError(
                    f"Expected minimum coordinate for axis {axis} to be 0, but got {min_coordinate}. Please report this bug."
                )
            if max_coordinate != data.shape[data.dims.index(axis)] - 1:
                raise ValueError(
                    f"Expected maximum coordinate for axis {axis} to be {data.shape[data.dims.index(axis)] - 1}, but got {max_coordinate}. Please report this bug."
                )
            return 0, data.shape[data.dims.index(axis)]
        else:
            left_pad = non_zero[0]
            right_pad = non_zero[-1] + 1
        return left_pad, right_pad

    axes = get_axis_names(raster)
    translation_axes = []
    translation_values: list[float] = []
    unpadded = raster

    # TODO: this "if else" will be unnecessary once we remove the concept of intrinsic coordinate systems and we make the
    #  transformations and xarray coordinates more interoperable
    if isinstance(unpadded, SpatialImage):
        for ax in axes:
            if ax != "c":
                left_pad, right_pad = _compute_paddings(data=unpadded, axis=ax)
                unpadded = unpadded.isel({ax: slice(left_pad, right_pad)})
                translation_axes.append(ax)
                translation_values.append(left_pad)
    elif isinstance(unpadded, MultiscaleSpatialImage):
        for ax in axes:
            if ax != "c":
                # let's just operate on the highest resolution. This is not an efficient implementation but we can
                # always optimize later
                d = dict(unpadded["scale0"])
                assert len(d) == 1
                xdata = d.values().__iter__().__next__()

                left_pad, right_pad = _compute_paddings(data=xdata, axis=ax)
                unpadded = unpadded.sel({ax: slice(left_pad, right_pad)})
                translation_axes.append(ax)
                translation_values.append(left_pad)
        d = {}
        for k, v in unpadded.items():
            assert len(v.values()) == 1
            xdata = v.values().__iter__().__next__()
            if 0 not in xdata.shape:
                d[k] = xdata
        unpadded = MultiscaleSpatialImage.from_dict(d)
    else:
        raise TypeError(f"Unsupported type: {type(raster)}")

    translation = Translation(translation_values, axes=tuple(translation_axes))
    old_transformations = get_transformation(element=raster, get_all=True)
    assert isinstance(old_transformations, dict)
    for target_cs, old_transform in old_transformations.items():
        assert old_transform is not None
        sequence = Sequence([translation, old_transform])
        set_transformation(element=unpadded, transformation=sequence, to_coordinate_system=target_cs)
    unpadded = compute_coordinates(unpadded)
    return unpadded


# TODO: probably we want this method to live in multiscale_spatial_image
def multiscale_spatial_image_from_data_tree(data_tree: DataTree) -> MultiscaleSpatialImage:
    d = {}
    for k, dt in data_tree.items():
        v = dt.values()
        assert len(v) == 1
        xdata = v.__iter__().__next__()
        d[k] = xdata
    return MultiscaleSpatialImage.from_dict(d)


# TODO: this functions is similar to _iter_multiscale(), the latter is more powerful but not exposed to the user.
#  Use only one and expose it to the user in this file
def iterate_pyramid_levels(image: MultiscaleSpatialImage) -> Generator[DataArray, None, None]:
    """
    Iterate over the pyramid levels of a multiscale spatial image.

    Parameters
    ----------
    image
        The multiscale spatial image.

    Returns
    -------
    A generator that yields the pyramid levels.
    """
    for k in range(len(image)):
        scale_name = f"scale{k}"
        dt = image[scale_name]
        v = dt.values()
        assert len(v) == 1
        xdata = next(iter(v))
        yield xdata