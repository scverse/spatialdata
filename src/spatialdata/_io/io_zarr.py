import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import zarr
import zarr.storage
from anndata import AnnData
from upath import UPath
from zarr.storage import FSStore

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import ome_zarr_logger, read_table_and_validate
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._logging import logger


def _open_zarr_store(store: Union[str, Path, zarr.Group, UPath]) -> tuple[zarr.Group, UPath]:
    """
    Open a zarr store (on-disk or remote) and return the zarr.Group object and the path to the store.

    Parameters
    ----------
    store
        Path to the zarr store (on-disk or remote) or a zarr.Group object.

    Returns
    -------
    A tuple of the zarr.Group object and the UPath to the store.
    """
    if isinstance(store, (str, Path)):
        f_store_path = UPath(store)
    elif isinstance(store, UPath):
        f_store_path = store
    if isinstance(store, zarr.Group):
        f_store_path = UPath(store._path)
        f = store
    else:
        fsstore = FSStore(url=f_store_path.path, fs=f_store_path.fs)
        f = zarr.open(fsstore, mode="r")
    return f, f_store_path


def _get_substore(upath: UPath, subpath: str) -> tuple[zarr.storage.BaseStore, UPath]:
    f, f_store_path = _open_zarr_store(upath / subpath)
    return f.store, f_store_path
    # if isinstance(store, (str, Path)):
    #     store = zarr.open(store, mode="r").store
    # if isinstance(store, zarr.Group):
    #     store = store.store
    # if isinstance(store, zarr.storage.DirectoryStore):
    #     # if local store, use local sepertor
    #     return os.path.join(store.path, path) if path else store.path
    # if isinstance(store, zarr.storage.FSStore):
    #     # reuse the same fs object, assume '/' as separator
    #     return zarr.storage.FSStore(url=store.path + "/" + path, fs=store.fs, mode="r")
    # if isinstance(store, zarr.storage.ConsolidatedMetadataStore):
    #     # reuse the same fs object, assume '/' as separator
    #     return store.store.path + path
    # # fallback to FSStore with standard fs, assume '/' as separator
    # return zarr.storage.FSStore(url=store.path + "/" + path, mode="r")


def read_zarr(store: Union[str, Path, zarr.Group, UPath], selection: Optional[tuple[str]] = None) -> SpatialData:
    """
    Read a SpatialData dataset from a zarr store (on-disk or remote).

    Parameters
    ----------
    store
        Path to the zarr store (on-disk or remote) or a zarr.Group object.

    selection
        List of elements to read from the zarr store (images, labels, points, shapes, table). If None, all elements are
        read.

    Returns
    -------
    A SpatialData object.
    """
    f, f_store_path = _open_zarr_store(store)

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    # TODO: remove table once deprecated.
    selector = {"images", "labels", "points", "shapes", "tables", "table"} if not selection else set(selection or [])
    logger.debug(f"Reading selection {selector}")

    # read multiscale images
    if "images" in selector and "images" in f:
        group = f["images"]
        count = 0
        for subgroup_name in group:
            if Path(subgroup_name).name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store, _ = _get_substore(f_store_path, f_elem.path)
            element = _read_multiscale(f_elem_store, raster_type="image")
            images[subgroup_name] = element
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if "labels" in selector and "labels" in f:
            group = f["labels"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                f_elem = group[subgroup_name]
                f_elem_store, _ = _get_substore(f_store_path, f_elem.path)
                labels[subgroup_name] = _read_multiscale(f_elem_store, raster_type="labels")
                count += 1
            logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in f:
        group = f["points"]
        count = 0
        for subgroup_name in group:
            f_elem = group[subgroup_name]
            if Path(subgroup_name).name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem_store, _ = _get_substore(f_store_path, f_elem.path)
            points[subgroup_name] = _read_points(f_elem_store)
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in f:
        group = f["shapes"]
        count = 0
        for subgroup_name in group:
            if Path(subgroup_name).name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store, _ = _get_substore(f_store_path, f_elem.path)
            shapes[subgroup_name] = _read_shapes(f_elem_store)
            count += 1
        logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in f:
        group = f["tables"]
        tables = read_table_and_validate(f_store_path, f, group, tables)

    if "table" in selector and "table" in f:
        warnings.warn(
            f"Table group found in zarr store at location {f_store_path}. Please update the zarr store"
            f"to use tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        subgroup_name = "table"
        group = f[subgroup_name]
        tables = read_table_and_validate(f_store_path, f, group, tables)

        logger.debug(f"Found {count} elements in {group}")

    sdata = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        tables=tables,
    )
    sdata.path = Path(store._path if isinstance(store, zarr.Group) else store)
    return sdata
