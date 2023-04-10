from __future__ import annotations

import filecmp
import logging
import os.path
import re
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import zarr
from dask.dataframe.core import DataFrame as DaskDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.writer import _get_valid_axes
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._utils import iterate_pyramid_levels
from spatialdata.models._utils import (
    MappingToCoordinateSystem_t,
    ValidAxis_t,
    _validate_mapping_to_coordinate_system_type,
)
from spatialdata.transformations.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata.transformations.transformations import (
    BaseTransformation,
    _get_current_output_axes,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


# suppress logger debug from ome_zarr with context manager
@contextmanager
def ome_zarr_logger(level: Any) -> Generator[None, None, None]:
    logger = logging.getLogger("ome_zarr")
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)


def _get_transformations_from_ngff_dict(
    list_of_encoded_ngff_transformations: list[dict[str, Any]]
) -> MappingToCoordinateSystem_t:
    list_of_ngff_transformations = [NgffBaseTransformation.from_dict(d) for d in list_of_encoded_ngff_transformations]
    list_of_transformations = [BaseTransformation.from_ngff(t) for t in list_of_ngff_transformations]
    transformations = {}
    for ngff_t, t in zip(list_of_ngff_transformations, list_of_transformations):
        assert ngff_t.output_coordinate_system is not None
        transformations[ngff_t.output_coordinate_system.name] = t
    return transformations


def overwrite_coordinate_transformations_non_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformations: MappingToCoordinateSystem_t
) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    ngff_transformations = []
    for target_coordinate_system, t in transformations.items():
        output_axes = _get_current_output_axes(transformation=t, input_axes=tuple(axes))
        ngff_transformations.append(
            t.to_ngff(
                input_axes=tuple(axes),
                output_axes=tuple(output_axes),
                output_coordinate_system_name=target_coordinate_system,
            ).to_dict()
        )
    group.attrs["coordinateTransformations"] = ngff_transformations


def overwrite_coordinate_transformations_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformations: MappingToCoordinateSystem_t
) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    # prepare the transformations in the dict representation
    ngff_transformations = []
    for target_coordinate_system, t in transformations.items():
        output_axes = _get_current_output_axes(transformation=t, input_axes=tuple(axes))
        ngff_transformations.append(
            t.to_ngff(
                input_axes=tuple(axes),
                output_axes=tuple(output_axes),
                output_coordinate_system_name=target_coordinate_system,
            )
        )
    coordinate_transformations = [t.to_dict() for t in ngff_transformations]
    # replace the metadata storage
    multiscales = group.attrs["multiscales"]
    assert len(multiscales) == 1
    multiscale = multiscales[0]
    # the transformation present in multiscale["datasets"] are the ones for the multiscale, so and we leave them intact
    # we update multiscale["coordinateTransformations"] and multiscale["coordinateSystems"]
    # see the first post of https://github.com/scverse/spatialdata/issues/39 for an overview
    # fix the io to follow the NGFF specs, see https://github.com/scverse/spatialdata/issues/114
    multiscale["coordinateTransformations"] = coordinate_transformations
    # multiscale["coordinateSystems"] = [t.output_coordinate_system_name for t in ngff_transformations]
    group.attrs["multiscales"] = multiscales


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    fmt: Format,
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write metdata to a group."""
    axes = _get_valid_axes(axes=axes, fmt=fmt)

    group.attrs["encoding-type"] = group_type
    group.attrs["axes"] = axes
    # we write empty coordinateTransformations and then overwrite
    # them with overwrite_coordinate_transformations_non_raster()
    group.attrs["coordinateTransformations"] = []
    # group.attrs["coordinateTransformations"] = coordinate_transformations
    group.attrs["spatialdata_attrs"] = attrs


def _iter_multiscale(
    data: MultiscaleSpatialImage,
    attr: Optional[str],
) -> list[Any]:
    # TODO: put this check also in the validator for raster multiscales
    for i in data:
        variables = set(data[i].variables.keys())
        names: set[str] = variables.difference({"c", "z", "y", "x"})
        if len(names) != 1:
            raise ValueError(f"Invalid variable name: `{names}`.")
    name: str = next(iter(names))
    if attr is not None:
        return [getattr(data[i][name], attr) for i in data]
    return [data[i][name] for i in data]


class dircmp(filecmp.dircmp):  # type: ignore[type-arg]
    """
    Compare the content of dir1 and dir2.

    In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """

    # from https://stackoverflow.com/a/24860799/3343783
    def phase3(self) -> None:
        """
        Differences between common files.

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
    if exclude_regexp is not None and (
        re.match(rf"{_root_dir1}/" + exclude_regexp, str(dir1))
        or re.match(rf"{_root_dir2}/" + exclude_regexp, str(dir2))
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


def _get_backing_files_raster(raster: DataArray) -> list[str]:
    files = []
    for k, v in raster.data.dask.layers.items():
        if k.startswith("original-from-zarr-"):
            mapping = v.mapping[k]
            path = mapping.store.path
            files.append(os.path.realpath(path))
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
            files.append(os.path.realpath(parquet_file))
    return files


@singledispatch
def get_channels(data: Any) -> list[Any]:
    """Get channels from data.

    Parameters
    ----------
    data
        data to get channels from

    Returns
    -------
    List of channels
    """
    raise ValueError(f"Cannot get channels from {type(data)}")


@get_channels.register
def _(data: SpatialImage) -> list[Any]:
    return data.coords["c"].values.tolist()  # type: ignore[no-any-return]


@get_channels.register
def _(data: MultiscaleSpatialImage) -> list[Any]:
    name = list({list(data[i].data_vars.keys())[0] for i in data})[0]
    channels = {tuple(data[i][name].coords["c"].values) for i in data}
    if len(channels) > 1:
        raise ValueError("TODO")
    return list(next(iter(channels)))


def save_transformations(sdata: SpatialData) -> None:
    """
    Save all the transformations of a SpatialData object to disk.

    sdata
        The SpatialData object
    """
    from spatialdata.transformations import get_transformation, set_transformation

    for element in sdata._gen_elements_values():
        transformations = get_transformation(element, get_all=True)
        set_transformation(element, transformations, set_all=True, write_to_sdata=sdata)
