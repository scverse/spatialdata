from __future__ import annotations

# from https://stackoverflow.com/a/24860799/3343783
import filecmp
import os.path
import re
import tempfile
from collections.abc import Generator
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from datatree import DataTree
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata_ops import get_transformation, set_transformation
from spatialdata._core.transformations import Sequence, Translation

if TYPE_CHECKING:
    from spatialdata import SpatialData


class dircmp(filecmp.dircmp):  # type: ignore[type-arg]
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """

    def phase3(self) -> None:
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files, shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp


def _are_directories_identical(
    dir1: Any,
    dir2: Any,
    exclude_regexp: Optional[str] = None,
    _root_dir1: Optional[str] = None,
    _root_dir2: Optional[str] = None,
) -> bool:
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    if _root_dir1 is None:
        _root_dir1 = dir1
    if _root_dir2 is None:
        _root_dir2 = dir2
    if exclude_regexp is not None:
        if re.match(rf"{_root_dir1}/" + exclude_regexp, str(dir1)) or re.match(
            rf"{_root_dir2}/" + exclude_regexp, str(dir2)
        ):
            return True

    compared = dircmp(dir1, dir2)
    if compared.left_only or compared.right_only or compared.diff_files or compared.funny_files:
        return False
    for subdir in compared.common_dirs:
        if not _are_directories_identical(
            os.path.join(dir1, subdir),
            os.path.join(dir2, subdir),
            exclude_regexp=exclude_regexp,
            _root_dir1=_root_dir1,
            _root_dir2=_root_dir2,
        ):
            return False
    return True


def _compare_sdata_on_disk(a: SpatialData, b: SpatialData) -> bool:
    from spatialdata import SpatialData

    if not isinstance(a, SpatialData) or not isinstance(b, SpatialData):
        return False
    # TODO: if the sdata object is backed on disk, don't create a new zarr file
    with tempfile.TemporaryDirectory() as tmpdir:
        a.write(os.path.join(tmpdir, "a.zarr"))
        b.write(os.path.join(tmpdir, "b.zarr"))
        return _are_directories_identical(os.path.join(tmpdir, "a.zarr"), os.path.join(tmpdir, "b.zarr"))


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
    from spatialdata._core.core_utils import compute_coordinates, get_dims

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

    axes = get_dims(raster)
    translation_axes = []
    translation_values: list[float] = []
    unpadded = raster

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


def _get_backing_files_raster(raster: DataArray) -> list[str]:
    files = []
    for k, v in raster.data.dask.layers.items():
        if k.startswith("original-from-zarr-"):
            mapping = v.mapping[k]
            path = mapping.store.path
            files.append(path)
    return files


@singledispatch
def get_backing_files(element: Union[SpatialImage, MultiscaleSpatialImage, DaskDataFrame]) -> list[str]:
    raise TypeError(f"Unsupported type: {type(element)}")


@get_backing_files.register(SpatialImage)
def _(element: SpatialImage) -> list[str]:
    return _get_backing_files_raster(element)


@get_backing_files.register(MultiscaleSpatialImage)
def _(element: MultiscaleSpatialImage) -> list[str]:
    xdata0 = next(iter(iterate_pyramid_levels(element)))
    return _get_backing_files_raster(xdata0)


@get_backing_files.register(DaskDataFrame)
def _(element: DaskDataFrame) -> list[str]:
    files = []
    layers = element.dask.layers
    for k, v in layers.items():
        if k.startswith("read-parquet-"):
            t = v.creation_info["args"]
            assert isinstance(t, tuple)
            assert len(t) == 1
            parquet_file = t[0]
            files.append(parquet_file)
    return files


# TODO: probably we want this method to live in multiscale_spatial_image
def multiscale_spatial_image_from_data_tree(data_tree: DataTree) -> MultiscaleSpatialImage:
    d = {}
    for k, dt in data_tree.items():
        v = dt.values()
        assert len(v) == 1
        xdata = v.__iter__().__next__()
        d[k] = xdata
    return MultiscaleSpatialImage.from_dict(d)


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
