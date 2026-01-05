import filecmp
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
from dask._task_spec import Task
from dask.array import Array as DaskArray
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from upath import UPath
from upath.implementations.local import PosixUPath, WindowsUPath
from xarray import DataArray, DataTree
from zarr.storage import FsspecStore, LocalStore

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io.format import RasterFormatType, RasterFormatV01, RasterFormatV02, RasterFormatV03
from spatialdata._utils import get_pyramid_levels
from spatialdata.models._utils import (
    MappingToCoordinateSystem_t,
    SpatialElement,
    ValidAxis_t,
    _validate_mapping_to_coordinate_system_type,
)
from spatialdata.transformations.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata.transformations.transformations import BaseTransformation, _get_current_output_axes


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
    """Write coordinate transformations of non-raster element to disk.

    Parameters
    ----------
    group: zarr.Group
        The zarr group containing the non-raster element for which to write the transformations, e.g. the zarr group
        containing sdata['points'].
    axes: tuple[ValidAxis_t, ...]
        The list with axes names in the same order as the coordinates of the non-raster element.
    transformations: MappingToCoordinateSystem_t
        Mapping between names of the coordinate system and the transformations.
    """
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
    group: zarr.Group,
    axes: tuple[ValidAxis_t, ...],
    transformations: MappingToCoordinateSystem_t,
    raster_format: RasterFormatType,
) -> None:
    """Write transformations of raster elements to disk.

    This function supports both writing of transformations for raster elements stored using zarr v3 and v2.
    For the case of zarr v3, there is already a 'coordinateTransformations' from ome-zarr in the metadata of
    the group. However, we store our transformations in the first element of the 'multiscales' of the attributes
    in the group metadata. This is subject to change.
    In the case of zarr v2 the existing 'coordinateTransformations' from ome-zarr is overwritten.

    Parameters
    ----------
    group
        The zarr group containing the raster element for which to write the transformations, e.g. the zarr group
        containing sdata['image2d'].
    axes
        The list with axes names in the same order as the dimensions of the raster element.
    transformations
        Mapping between names of the coordinate system and the transformations.
    raster_format
        The raster format of the raster element used to determine where in the metadata the transformations should be
        written.
    """
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
    if group.metadata.zarr_format == 3 and len(multiscales := group.metadata.attributes["ome"]["multiscales"]) != 1:
        len_scales = len(multiscales)
        raise ValueError(f"The length of multiscales metadata should be 1, found the length to be {len_scales}")
    if group.metadata.zarr_format == 2:
        multiscales = group.attrs["multiscales"]
        if (len_scales := len(multiscales)) != 1:
            raise ValueError(f"The length of multiscales metadata should be 1, found length of {len_scales}")
    multiscale = multiscales[0]

    # Previously, there was CoordinateTransformations key present at the level of multiscale and datasets in multiscale.
    # This is not the case anymore so we are creating a new key here and keeping the one in datasets intact.
    multiscale["coordinateTransformations"] = coordinate_transformations
    if raster_format is not None:
        if isinstance(raster_format, RasterFormatV01 | RasterFormatV02):
            multiscale["version"] = raster_format.version
            group.attrs["multiscales"] = multiscales
        elif isinstance(raster_format, RasterFormatV03):
            ome = group.metadata.attributes["ome"]
            ome["version"] = raster_format.version
            ome["multiscales"] = multiscales
            group.attrs["ome"] = ome
        else:
            raise ValueError(f"Unsupported raster format: {type(raster_format)}")


def overwrite_channel_names(group: zarr.Group, element: DataArray | DataTree) -> None:
    """Write channel metadata to a group."""
    if isinstance(element, DataArray):
        channel_names = element.coords["c"].data.tolist()
    else:
        channel_names = element["scale0"]["image"].coords["c"].data.tolist()

    channel_metadata = [{"label": name} for name in channel_names]
    # This is required here as we do not use the load node API of ome-zarr
    omero_meta = group.attrs.get("omero", None) or group.attrs.get("ome", {}).get("omero")
    omero_meta["channels"] = channel_metadata
    if ome_meta := group.attrs.get("ome", None):
        ome_meta["omero"] = omero_meta
        group.attrs["ome"] = ome_meta
    else:
        group.attrs["omero"] = omero_meta


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


def _find_piece_dict(obj: dict[str, tuple[str | None]] | Task) -> dict[str, tuple[str | None | None]] | None:
    """Recursively search for dict containing the key 'piece' in Dask task specs containing the parquet file path."""
    if isinstance(obj, dict):
        if "piece" in obj:
            return obj
    elif hasattr(obj, "args"):  # Handles dask._task_spec.* objects like Task and List
        for v in obj.args:
            result = _find_piece_dict(v)
            if result is not None:
                return result
    return None


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
                    # LocalStore.store does not have an attribute path, but we keep it like this for backward compat.
                    path = getattr(v.store, "path", None) if getattr(v.store, "path", None) else v.store.root
                    files.append(str(UPath(path).resolve()))
                elif name.startswith("read-parquet") or name.startswith("read_parquet"):
                    # Here v is a read_parquet task with arguments and the only value is a dictionary.
                    if "piece" in v.args[0]:
                        # https://github.com/dask/dask/blob/ff2488aec44d641696e0b7aa41ed9e995c710705/dask/dataframe/io/parquet/core.py#L870
                        parquet_file, check0, check1 = v.args[0]["piece"]
                        if not parquet_file.endswith(".parquet") or check0 is not None or check1 is not None:
                            raise ValueError(
                                f"Unable to parse the parquet file from the dask subgraph {subgraph}. Please "
                                f"report this bug."
                            )
                        files.append(os.path.realpath(parquet_file))
                    else:
                        # This occurs when for example points and images are mixed, the main task still starts with
                        # read_parquet, but the execution happens through a subgraph which we iterate over to get the
                        # actual read_parquet task.
                        for task in v.args[0].values():
                            # Recursively go through tasks, this is required because differences between dask versions.
                            piece_dict = _find_piece_dict(task)
                            if isinstance(piece_dict, dict) and "piece" in piece_dict:
                                parquet_file, check0, check1 = piece_dict["piece"]  # type: ignore[misc]
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
    # TODO when running test_save_transformations it seems that for the same element this is called multiple times
    return all(_backed_elements_contained_in_path(path=element_path, object=element))


def _resolve_zarr_store(
    path: str | Path | UPath | zarr.storage.StoreLike | zarr.Group, **kwargs: Any
) -> zarr.storage.StoreLike:
    """
    Normalize different Zarr store inputs into a usable store instance.

    This function accepts various forms of input (e.g. filesystem paths,
    UPath objects, existing Zarr stores, or `zarr.Group`s) and resolves
    them into a `StoreLike` that can be passed to Zarr APIs. It handles
    local files, fsspec-backed stores, consolidated metadata stores, and
    groups with nested paths.

    Parameters
    ----------
    path
        The input representing a Zarr store or group. Can be a filesystem
        path, remote path, existing store, or Zarr group.
    **kwargs
        Additional keyword arguments forwarded to the underlying store
        constructor (e.g. `mode`, `storage_options`).

    Returns
    -------
    A normalized store instance suitable for use with Zarr.

    Raises
    ------
    TypeError
        If the input type is unsupported.
    ValueError
        If a `zarr.Group` has an unsupported store type.
    """
    # TODO: ensure kwargs like mode are enforced everywhere and passed correctly to the store
    if isinstance(path, str | Path):
        # if the input is str or Path, map it to UPath
        path = UPath(path)

    if isinstance(path, PosixUPath | WindowsUPath):
        # if the input is a local path, use LocalStore
        return LocalStore(path.path)

    if isinstance(path, zarr.Group):
        # if the input is a zarr.Group, wrap it with a store
        if isinstance(path.store, LocalStore):
            store_path = UPath(path.store.root) / path.path
            return LocalStore(store_path.path)
        if isinstance(path.store, FsspecStore):
            # if the store within the zarr.Group is an FSStore, return it
            # but extend the path of the store with that of the zarr.Group
            return FsspecStore(path.store.path + "/" + path.path, fs=path.store.fs, **kwargs)
        if isinstance(path.store, zarr.storage.ConsolidatedMetadataStore):
            # if the store is a ConsolidatedMetadataStore, just return the underlying FSSpec store
            return path.store.store
        raise ValueError(f"Unsupported store type or zarr.Group: {type(path.store)}")
    if isinstance(path, zarr.storage.StoreLike):
        # if the input already a store, wrap it in an FSStore
        return FsspecStore(path, **kwargs)
    if isinstance(path, UPath):
        # if input is a remote UPath, map it to an FSStore
        return FsspecStore(path.path, fs=path.fs, **kwargs)
    raise TypeError(f"Unsupported type: {type(path)}")


class BadFileHandleMethod(Enum):
    ERROR = "error"
    WARN = "warn"


@contextmanager
def handle_read_errors(
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN],
    location: str,
    exc_types: type[BaseException] | tuple[type[BaseException], ...],
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
