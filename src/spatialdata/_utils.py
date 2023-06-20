from __future__ import annotations

import re
from collections.abc import Generator
from copy import deepcopy
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from dask import array as da
from datatree import DataTree
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._types import ArrayLike
from spatialdata.transformations import (
    Sequence,
    Translation,
    get_transformation,
    set_transformation,
)

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = Union[int, float]

if TYPE_CHECKING:
    pass


def _parse_list_into_array(array: list[Number] | ArrayLike) -> ArrayLike:
    if isinstance(array, list):
        array = np.array(array)
    if array.dtype != float:
        return array.astype(float)
    return array


def _atoi(text: str) -> int | str:
    return int(text) if text.isdigit() else text


# from https://stackoverflow.com/a/5967539/3343783
def _natural_keys(text: str) -> list[int | str]:
    """Sort keys in natural order.

    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments).
    """
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def _affine_matrix_multiplication(matrix: ArrayLike, data: ArrayLike) -> ArrayLike:
    assert len(data.shape) == 2
    assert matrix.shape[1] - 1 == data.shape[1]
    vector_part = matrix[:-1, :-1]
    offset_part = matrix[:-1, -1]
    result = data @ vector_part.T + offset_part
    assert result.shape[0] == data.shape[0]
    return result  # type: ignore[no-any-return]


def unpad_raster(raster: SpatialImage | MultiscaleSpatialImage) -> SpatialImage | MultiscaleSpatialImage:
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
    from spatialdata.models import get_axes_names
    from spatialdata.transformations._utils import compute_coordinates

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
                    f"Expected minimum coordinate for axis {axis} to be 0,"
                    f"but got {min_coordinate}. Please report this bug."
                )
            if max_coordinate != data.shape[data.dims.index(axis)] - 1:
                raise ValueError(
                    f"Expected maximum coordinate for axis {axis} to be"
                    f"{data.shape[data.dims.index(axis)] - 1},"
                    f"but got {max_coordinate}. Please report this bug."
                )
            return 0, data.shape[data.dims.index(axis)]

        left_pad = non_zero[0]
        right_pad = non_zero[-1] + 1
        return left_pad, right_pad

    axes = get_axes_names(raster)
    translation_axes = []
    translation_values: list[float] = []
    unpadded = raster

    # TODO: this "if else" will be unnecessary once we remove the
    #  concept of intrinsic coordinate systems and we make the
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


def _inplace_fix_subset_categorical_obs(subset_adata: AnnData, original_adata: AnnData) -> None:
    """
    Fix categorical obs columns of subset_adata to match the categories of original_adata.

    Parameters
    ----------
    subset_adata
        The subset AnnData object
    original_adata
        The original AnnData object

    Notes
    -----
    See discussion here: https://github.com/scverse/anndata/issues/997
    """
    obs = subset_adata.obs
    for column in obs.columns:
        is_categorical = pd.api.types.is_categorical_dtype(obs[column])
        if is_categorical:
            c = obs[column].cat.set_categories(original_adata.obs[column].cat.categories)
            obs[column] = c


def _deepcopy_geodataframe(gdf: GeoDataFrame) -> GeoDataFrame:
    """
    temporary fix for https://github.com/scverse/spatialdata/issues/286.

    Parameters
    ----------
    gdf
        The GeoDataFrame to deepcopy

    Returns
    -------
    A deepcopy of the GeoDataFrame
    """
    #
    new_gdf = deepcopy(gdf)
    new_attrs = deepcopy(gdf.attrs)
    new_gdf.attrs = new_attrs
    return new_gdf
