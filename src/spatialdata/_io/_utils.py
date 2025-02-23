from __future__ import annotations

import filecmp
import logging
import os.path
import re
import sys
import tempfile
import traceback
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any, Literal

import zarr
from anndata import AnnData
from dask.array import Array as DaskArray
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata._utils import get_pyramid_levels
from spatialdata.models._utils import (
    MappingToCoordinateSystem_t,
    SpatialElement,
    ValidAxis_t,
    _validate_mapping_to_coordinate_system_type,
)
from spatialdata.transformations.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata.transformations.transformations import (
    BaseTransformation,
    _get_current_output_axes,
)


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
    list_of_encoded_ngff_transformations: list[dict[str, Any]],
) -> MappingToCoordinateSystem_t:
    list_of_ngff_transformations = [NgffBaseTransformation.from_dict(d) for d in list_of_encoded_ngff_transformations]
    list_of_transformations = [BaseTransformation.from_ngff(t) for t in list_of_ngff_transformations]
    transformations = {}
    for ngff_t, t in zip(list_of_ngff_transformations, list_of_transformations, strict=True):
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


def overwrite_channel_names(group: zarr.Group, element: DataArray | DataTree) -> None:
    """Write channel metadata to a group."""
    if isinstance(element, DataArray):
        channel_names = element.coords["c"].data.tolist()
    else:
        channel_names = element["scale0"]["image"].coords["c"].data.tolist()

    channel_metadata = [{"label": name} for name in channel_names]
    omero_meta = group.attrs["omero"]
    omero_meta["channels"] = channel_metadata
    group.attrs["omero"] = omero_meta
    multiscales_meta = group.attrs["multiscales"]
    if len(multiscales_meta) != 1:
        raise ValueError(
            f"Multiscale metadata must be of length one but got length {len(multiscales_meta)}. Data mightbe corrupted."
        )
    multiscales_meta[0]["metadata"]["omero"]["channels"] = channel_metadata
    group.attrs["multiscales"] = multiscales_meta


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    axes: list[str],
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write metdata to a group."""
    axes = sorted(axes)

    group.attrs["encoding-type"] = group_type
    group.attrs["axes"] = axes
    # we write empty coordinateTransformations and then overwrite
    # them with overwrite_coordinate_transformations_non_raster()
    group.attrs["coordinateTransformations"] = []
    group.attrs["spatialdata_attrs"] = attrs


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
    exclude_regexp: str | None = None,
    _root_dir1: str | None = None,
    _root_dir2: str | None = None,
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
        re.match(rf"{re.escape(str(_root_dir1))}/" + exclude_regexp, str(dir1))
        or re.match(rf"{re.escape(str(_root_dir2))}/" + exclude_regexp, str(dir2))
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
    if not isinstance(a, SpatialData) or not isinstance(b, SpatialData):
        return False
    # TODO: if the sdata object is backed on disk, don't create a new zarr file
    with tempfile.TemporaryDirectory() as tmpdir:
        a.write(os.path.join(tmpdir, "a.zarr"))
        b.write(os.path.join(tmpdir, "b.zarr"))
        return _are_directories_identical(os.path.join(tmpdir, "a.zarr"), os.path.join(tmpdir, "b.zarr"))


@singledispatch
def get_dask_backing_files(element: SpatialData | SpatialElement | AnnData) -> list[str]:
    """
    Get the backing files that appear in the Dask computational graph of an element/any element of a SpatialData object.

    Parameters
    ----------
    element
        The element to get the backing files from.

    Returns
    -------
    List of backing files.

    Notes
    -----
    It is possible for lazy objects to be constructed from multiple files.
    """
    raise TypeError(f"Unsupported type: {type(element)}")


@get_dask_backing_files.register(SpatialData)
def _(element: SpatialData) -> list[str]:
    files: set[str] = set()
    for e in element._gen_spatial_element_values():
        if isinstance(e, DataArray | DataTree | DaskDataFrame):
            files = files.union(get_dask_backing_files(e))
    return list(files)


@get_dask_backing_files.register(DataArray)
def _(element: DataArray) -> list[str]:
    return _get_backing_files(element.data)


@get_dask_backing_files.register(DataTree)
def _(element: DataTree) -> list[str]:
    dask_data_scale0 = get_pyramid_levels(element, attr="data", n=0)
    return _get_backing_files(dask_data_scale0)


@get_dask_backing_files.register(DaskDataFrame)
def _(element: DaskDataFrame) -> list[str]:
    return _get_backing_files(element)


@get_dask_backing_files.register(AnnData)
@get_dask_backing_files.register(GeoDataFrame)
def _(element: AnnData | GeoDataFrame) -> list[str]:
    return []


def _get_backing_files(element: DaskArray | DaskDataFrame) -> list[str]:
    files: list[str] = []
    _search_for_backing_files_recursively(subgraph=element.dask, files=files)
    return files


def _search_for_backing_files_recursively(subgraph: Any, files: list[str]) -> None:
    # see the types allowed for the dask graph here: https://docs.dask.org/en/stable/spec.html

    # search recursively
    if isinstance(subgraph, Mapping):
        for k, v in subgraph.items():
            _search_for_backing_files_recursively(subgraph=k, files=files)
            _search_for_backing_files_recursively(subgraph=v, files=files)
    elif isinstance(subgraph, Sequence) and not isinstance(subgraph, str):
        for v in subgraph:
            _search_for_backing_files_recursively(subgraph=v, files=files)

    # cases where a backing file is found
    if isinstance(subgraph, Mapping):
        for k, v in subgraph.items():
            name = None
            if isinstance(k, Sequence) and not isinstance(k, str):
                name = k[0]
            elif isinstance(k, str):
                name = k
            if name is not None:
                if name.startswith("original-from-zarr"):
                    path = v.store.path
                    files.append(os.path.realpath(path))
                elif name.startswith("read-parquet") or name.startswith("read_parquet"):
                    if hasattr(v, "creation_info"):
                        # https://github.com/dask/dask/blob/ff2488aec44d641696e0b7aa41ed9e995c710705/dask/dataframe/io/parquet/core.py#L625
                        t = v.creation_info["args"]
                        if not isinstance(t, tuple) or len(t) != 1:
                            raise ValueError(
                                f"Unable to parse the parquet file from the dask subgraph {subgraph}. Please "
                                f"report this bug."
                            )
                        parquet_file = t[0]
                        files.append(os.path.realpath(parquet_file))
                    elif isinstance(v, tuple) and len(v) > 1 and isinstance(v[1], dict) and "piece" in v[1]:
                        # https://github.com/dask/dask/blob/ff2488aec44d641696e0b7aa41ed9e995c710705/dask/dataframe/io/parquet/core.py#L870
                        parquet_file, check0, check1 = v[1]["piece"]
                        if not parquet_file.endswith(".parquet") or check0 is not None or check1 is not None:
                            raise ValueError(
                                f"Unable to parse the parquet file from the dask subgraph {subgraph}. Please "
                                f"report this bug."
                            )
                        files.append(os.path.realpath(parquet_file))


def _backed_elements_contained_in_path(path: Path, object: SpatialData | SpatialElement | AnnData) -> list[bool]:
    """
    Return the list of boolean values indicating if backing files for an object are child directory of a path.

    Parameters
    ----------
    path
        The path to check if the backing files are contained in.
    object
        The object to check the backing files of.

    Returns
    -------
    List of boolean values for each of the backing files.

    Notes
    -----
    If an object does not have a Dask computational graph, it will return an empty list.
    It is possible for a single SpatialElement to contain multiple files in their Dask computational graph.
    """
    if not isinstance(path, Path):
        raise TypeError(f"Expected a Path object, got {type(path)}")
    return [_is_subfolder(parent=path, child=Path(fp)) for fp in get_dask_backing_files(object)]


def _is_subfolder(parent: Path, child: Path) -> bool:
    """
    Check if a path is a subfolder of another path.

    Parameters
    ----------
    parent
        The parent folder.
    child
        The child folder.

    Returns
    -------
    True if the child is a subfolder of the parent.
    """
    if isinstance(child, str):
        child = Path(child)
    if isinstance(parent, str):
        parent = Path(parent)
    if not isinstance(parent, Path) or not isinstance(child, Path):
        raise TypeError(f"Expected a Path object, got {type(parent)} and {type(child)}")
    return child.resolve().is_relative_to(parent.resolve())


def _is_element_self_contained(
    element: DataArray | DataTree | DaskDataFrame | GeoDataFrame | AnnData, element_path: Path
) -> bool:
    if isinstance(element, DaskDataFrame):
        pass
    return all(_backed_elements_contained_in_path(path=element_path, object=element))


def save_transformations(sdata: SpatialData) -> None:
    """
    Save all the transformations of a SpatialData object to disk.

    sdata
        The SpatialData object
    """
    warnings.warn(
        "This function is deprecated and should be replaced by `SpatialData.write_transformations()` or "
        "`SpatialData.write_metadata()`, which gives more control over which metadata to write. This function will call"
        " `SpatialData.write_transformations()`; please call this function directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    sdata.write_transformations()


class BadFileHandleMethod(Enum):
    ERROR = "error"
    WARN = "warn"


@contextmanager
def handle_read_errors(
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN],
    location: str,
    exc_types: tuple[type[Exception], ...],
) -> Generator[None, None, None]:
    """
    Handle read errors according to parameter `on_bad_files`.

    Parameters
    ----------
    on_bad_files
        Specifies what to do upon encountering an exception.
        Allowed values are :

        - 'error', let the exception be raised.
        - 'warn', convert the exception into a warning if it is one of the expected exception types.
    location
        String identifying the function call where the exception happened
    exc_types
        A tuple of expected exception classes that should be converted into warnings.

    Raises
    ------
    If `on_bad_files="error"`, all encountered exceptions are raised.
    If `on_bad_files="warn"`, any encountered exceptions not matching the `exc_types` are raised.
    """
    on_bad_files = BadFileHandleMethod(on_bad_files)  # str to enum
    if on_bad_files == BadFileHandleMethod.WARN:
        try:
            yield
        except exc_types as e:
            # Extract the original filename and line number from the exception and
            # create a warning from it.
            exc_traceback = sys.exc_info()[-1]
            last_frame, lineno = list(traceback.walk_tb(exc_traceback))[-1]
            filename = last_frame.f_code.co_filename
            # Include the location (element path) in the warning message.
            message = f"{location}: {e.__class__.__name__}: {e.args[0]}"
            warnings.warn_explicit(
                message=message,
                category=UserWarning,
                filename=filename,
                lineno=lineno,
            )
            # continue
    else:  # on_bad_files == BadFileHandleMethod.ERROR
        # Let it raise exceptions
        yield
