from __future__ import annotations

# from https://stackoverflow.com/a/24860799/3343783
import filecmp
import os.path
import re
import tempfile
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import numpy as np
from anndata import AnnData
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
    from spatialdata._core.models import get_schema

    def _unpad_axis(data: DataArray, axis: str) -> tuple[DataArray, float]:
        others = list(data.dims)
        others.remove(axis)
        # mypy (luca's pycharm config) can't see the isclose method of dask array
        s = da.isclose(data.sum(dim=others), 0)  # type: ignore[attr-defined]
        # TODO: rewrite this to use dask array; can't get it to work with it
        x = s.compute()
        non_zero = np.where(x == 0)[0]
        if len(non_zero) == 0:
            return data, 0
        else:
            left_pad = non_zero[0]
            right_pad = non_zero[-1] + 1
            unpadded = data.isel({axis: slice(left_pad, right_pad)})
            return unpadded, left_pad

    from spatialdata._core.core_utils import get_dims

    axes = get_dims(raster)
    if isinstance(raster, SpatialImage):
        unpadded = raster
        translation_axes = []
        translation_values: list[float] = []
        for ax in axes:
            if ax != "c":
                unpadded, left_pad = _unpad_axis(unpadded, axis=ax)
                translation_axes.append(ax)
                translation_values.append(left_pad)
        translation = Translation(translation_values, axes=tuple(translation_axes))
        old_transformations = get_transformation(element=raster, get_all=True)
        assert isinstance(old_transformations, dict)
        for target_cs, old_transform in old_transformations.items():
            assert old_transform is not None
            sequence = Sequence([translation, old_transform])
            set_transformation(element=unpadded, transformation=sequence, to_coordinate_system=target_cs)
        return unpadded
    elif isinstance(raster, MultiscaleSpatialImage):
        # let's just operate on the highest resolution. This is not an efficient implementation but we can always optimize later
        d = dict(raster["scale0"])
        assert len(d) == 1
        xdata = d.values().__iter__().__next__()
        unpadded = unpad_raster(SpatialImage(xdata))
        # TODO: here I am using some arbitrary scalingfactors, I think that we need an automatic initialization of multiscale. See discussion: https://github.com/scverse/spatialdata/issues/108
        # mypy thinks that the schema could be a ShapeModel, ... but it's not
        unpadded_multiscale = get_schema(raster).parse(unpadded, multiscale_factors=[2, 2])  # type: ignore[call-arg]
        return unpadded_multiscale
    else:
        raise TypeError(f"Unsupported type: {type(raster)}")


def get_table_mapping_metadata(table: AnnData) -> dict[str, Union[Optional[Union[str, list[str]]], Optional[str]]]:
    """
    Get the region, region_key and instance_key from the table metadata.

    Parameters
    ----------
    table
        The table to get the metadata from.

    Returns
    -------
    The `region`, `region_key`, and `instance_key` values.
    """
    from spatialdata._core.models import TableModel

    TableModel().validate(table)
    region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
    region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    instance_key = table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    d = {TableModel.REGION_KEY: region, TableModel.REGION_KEY_KEY: region_key, TableModel.INSTANCE_KEY: instance_key}
    return d
