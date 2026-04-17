from __future__ import annotations

import filecmp
import json
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
from pathlib import Path, PurePosixPath
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


def join_fsspec_store_path(store_path: str, relative_path: str) -> str:
    """Append a relative zarr-group path to an FsspecStore root, yielding a fsspec key."""
    rel = relative_path.lstrip("/")
    return str(PurePosixPath(store_path) / rel) if rel else store_path


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


def _extract_parquet_paths_from_task(obj: Any) -> list[str]:
    """Recursively extract parquet file paths from a dask ``read_parquet`` task.

    Dask's task-graph shape changed between the version pinned before scverse/spatialdata
    PR #1006 (https://github.com/scverse/spatialdata/pull/1006 "unpinning dask", commit
    53b9438a https://github.com/scverse/spatialdata/commit/53b9438a328c5fc2a451d2c8afab439b945ba2b8)
    and the current one; we tolerate both.

    - Legacy shape: a dict ``{"piece": (parquet_file, None, None)}`` somewhere in the args
      (possibly wrapped in other dicts for mixed points+images element graphs). The trailing
      elements of the ``piece`` tuple encode row-group / filter constraints; we only support
      unfiltered reads (hence the validation on ``check0`` / ``check1``).
    - Current shape: a ``dask.dataframe.dask_expr.io.parquet.FragmentWrapper`` whose
      ``.fragment.path`` is the parquet file (from ``dask_expr.io.parquet.ReadParquetPyarrowFS``).
      The wrapper may live in Task ``kwargs["fragment_wrapper"]`` for simple reads, but in fused
      expressions (``readparquetpyarrowfs-fused-*``) it is nested inside lists and tuples
      inside a subgraph dict, so we walk every container uniformly rather than targeting named
      kwargs.

    ``FragmentWrapper`` is detected via the ``.fragment.path`` attribute chain instead of an
    isinstance check to avoid importing private dask_expr internals; the ``endswith(".parquet")``
    guard keeps false positives from random objects out of the result.
    """
    found: list[str] = []

    frag = getattr(obj, "fragment", None)
    if frag is not None:
        path = getattr(frag, "path", None)
        if isinstance(path, str) and path.endswith(".parquet"):
            found.append(path)

    if isinstance(obj, Mapping):
        # TODO(legacy-dask): the ``"piece"`` branch targets the pre-PR-#1006 dask graph shape
        # (``dask/dataframe/io/parquet/core.py`` produced ``{"piece": (file, rg, filters)}``). The
        # current dask pin (``dask>=2025.12.0``) no longer emits this shape at runtime; the branch
        # is kept only as a safety net for users forcing an older dask via pip. Remove once the
        # lower pin is bumped past the PR-#1006 cut-off and CI covers only the new shape.
        if "piece" in obj:
            piece = obj["piece"]
            if isinstance(piece, tuple) and len(piece) >= 1 and isinstance(piece[0], str):
                parquet_file = piece[0]
                check0 = piece[1] if len(piece) > 1 else None
                check1 = piece[2] if len(piece) > 2 else None
                if not parquet_file.endswith(".parquet") or check0 is not None or check1 is not None:
                    raise ValueError(
                        f"Unable to parse the parquet file from the dask task {obj!r}. Please report this bug."
                    )
                found.append(parquet_file)
        for v in obj.values():
            found.extend(_extract_parquet_paths_from_task(v))
        return found

    if isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(_extract_parquet_paths_from_task(item))
        return found

    # TODO(dask-task-api): the ``kwargs`` / ``args`` getattr probes here rely on the Task wrapper
    # object introduced alongside PR #1006. The attribute contract is not documented as public
    # (``dask.dataframe.dask_expr``), so we access it defensively via getattr and traverse every
    # container uniformly. If dask stabilises a public accessor (e.g. ``task.iter_leaves()`` or an
    # expr-level ``file_paths`` property) or if ``FragmentWrapper`` becomes importable from a
    # stable namespace, replace the attribute-chain walk with a typed call and drop the getattrs.
    kwargs = getattr(obj, "kwargs", None)
    if isinstance(kwargs, Mapping):
        for v in kwargs.values():
            found.extend(_extract_parquet_paths_from_task(v))

    args = getattr(obj, "args", None)
    if isinstance(args, (list, tuple)):
        for a in args:
            found.extend(_extract_parquet_paths_from_task(a))

    return found


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
                    # TODO(zarr-v3-store-path): the ``getattr(..., "path", None)`` fallback dates
                    # back to zarr v2, where ``DirectoryStore`` exposed ``.path`` and the v3
                    # ``LocalStore`` exposes ``.root`` instead. With the current pin
                    # (``zarr>=3.0.0``) the getattr branch is never taken for local backends -- it
                    # only covers exotic third-party stores that still mimic the v2 attribute.
                    # Once we are confident no such shim stores are in use, collapse this to just
                    # ``v.store.root`` and drop the getattr probe.
                    path = getattr(v.store, "path", None) if getattr(v.store, "path", None) else v.store.root
                    files.append(str(UPath(path).resolve()))
                elif "parquet" in name.lower():
                    # Matches every dask task-key that wraps a parquet read across versions:
                    #   - legacy ``read-parquet-<hash>`` / ``read_parquet-<hash>`` (pre scverse/
                    #     spatialdata PR #1006, https://github.com/scverse/spatialdata/pull/1006),
                    #   - current ``read_parquet-<hash>`` plus fused-expression forms such as
                    #     ``readparquetpyarrowfs-fused-values-<hash>`` produced by
                    #     ``dask_expr.io.parquet.ReadParquetPyarrowFS`` when a parquet column is
                    #     combined with other arrays (see ``test_self_contained``).
                    # Any false-positive key that matches but carries no parquet payload is filtered
                    # inside ``_extract_parquet_paths_from_task`` (paths must ``endswith(".parquet")``).
                    for parquet_file in _extract_parquet_paths_from_task(v):
                        files.append(os.path.realpath(parquet_file))


def _backed_elements_contained_in_path(
    path: Path | UPath, object: SpatialData | SpatialElement | AnnData
) -> list[bool]:
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

    For a remote ``path`` (:class:`upath.UPath`), this always returns an empty list: Dask backing paths
    are resolved as local filesystem paths, so they cannot be compared to object-store locations.
    :meth:`spatialdata.SpatialData.write` therefore skips the local "backing files in target" guard
    for remote targets; ``overwrite=True`` on a remote URL must be used only when overwriting is safe.
    """
    if isinstance(path, UPath):
        return []
    if not isinstance(path, Path):
        raise TypeError(f"Expected a Path or UPath object, got {type(path)}")
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
    element: DataArray | DataTree | DaskDataFrame | GeoDataFrame | AnnData,
    element_path: Path | UPath,
) -> bool:
    """Whether element Dask graphs only reference files under ``element_path`` (local) or N/A (remote)."""
    if isinstance(element_path, UPath):
        # Backing-file paths are local; cannot relate them to remote keys—assume OK for this heuristic.
        return True
    if isinstance(element, DaskDataFrame):
        pass
    # TODO when running test_save_transformations it seems that for the same element this is called multiple times
    return all(_backed_elements_contained_in_path(path=element_path, object=element))


def _ensure_async_fs(fs: Any) -> Any:
    """Return an async fsspec filesystem for use with zarr's FsspecStore.

    Zarr's FsspecStore expects an async filesystem. If the given fs is synchronous,
    it is converted using fsspec's public API (async instance or AsyncFileSystemWrapper)
    so that ZarrUserWarning is not raised.
    """
    if getattr(fs, "asynchronous", False):
        return fs
    import fsspec

    if getattr(fs, "async_impl", False):
        fs_dict = json.loads(fs.to_json())
        fs_dict["asynchronous"] = True
        return fsspec.AbstractFileSystem.from_json(json.dumps(fs_dict))
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    return AsyncFileSystemWrapper(fs, asynchronous=True)


def _resolve_zarr_store(
    path: str | Path | UPath | zarr.storage.StoreLike | zarr.Group,
    *,
    read_only: bool = False,
    **kwargs: Any,
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
    read_only
        If ``True``, constructed ``LocalStore`` / ``FsspecStore`` instances are built with
        ``read_only=True``. Stores that already exist (when ``path`` is a ``StoreLike`` or
        a ``zarr.Group`` whose wrapped store is not reconstructable) are returned as-is;
        the caller is responsible for opening them at the right mode.
    **kwargs
        Additional keyword arguments forwarded to the underlying store
        constructor.

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
    if isinstance(path, str | Path):
        path = UPath(path)

    if isinstance(path, PosixUPath | WindowsUPath):
        # if the input is a local path, use LocalStore
        return LocalStore(path.path, read_only=read_only)

    if isinstance(path, zarr.Group):
        # Re-wrap the group's store at the group's subpath. Note: zarr v3 no longer ships
        # ``ConsolidatedMetadataStore`` (v2 wrapped the backend in a store; v3 surfaces
        # consolidated metadata as a field on ``GroupMetadata`` instead), so we only need to
        # handle the two concrete backends below.
        if isinstance(path.store, LocalStore):
            store_path = UPath(path.store.root) / path.path
            return LocalStore(store_path.path, read_only=read_only)
        if isinstance(path.store, FsspecStore):
            return FsspecStore(
                fs=_ensure_async_fs(path.store.fs),
                path=join_fsspec_store_path(path.store.path, path.path),
                read_only=read_only,
                **kwargs,
            )
        raise ValueError(f"Unsupported store type or zarr.Group: {type(path.store)}")
    if isinstance(path, UPath):
        # Check before StoreLike to avoid UnionType isinstance.
        return FsspecStore(_ensure_async_fs(path.fs), path=path.path, read_only=read_only, **kwargs)
    if isinstance(path, zarr.storage.StoreLike):
        # Already a concrete store (LocalStore, FsspecStore, MemoryStore, ...). Do not pass it as ``fs=`` to
        # FsspecStore -- that only accepts an async fsspec filesystem and raises on stores (e.g. ``async_impl``).
        return path
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
