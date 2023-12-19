import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import zarr
from anndata import AnnData

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import ome_zarr_logger, read_table_and_validate
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._logging import logger


def _open_zarr_store(store: Union[str, Path, zarr.Group]) -> tuple[zarr.Group, str]:
    """
    Open a zarr store (on-disk or remote) and return the zarr.Group object and the path to the store.

    Parameters
    ----------
    store
        Path to the zarr store (on-disk or remote) or a zarr.Group object.

    Returns
    -------
    A tuple of the zarr.Group object and the path to the store.
    """
    f = store if isinstance(store, zarr.Group) else zarr.open(store, mode="r")
    # workaround: .zmetadata is being written as zmetadata (https://github.com/zarr-developers/zarr-python/issues/1121)
    if isinstance(store, (str, Path)) and str(store).startswith("http") and len(f) == 0:
        f = zarr.open_consolidated(store, mode="r", metadata_key="zmetadata")
    f_store_path = f.store.store.path if isinstance(f.store, zarr.storage.ConsolidatedMetadataStore) else f.store.path
    return f, f_store_path


def read_zarr(store: Union[str, Path, zarr.Group], selection: Optional[tuple[str]] = None) -> SpatialData:
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
            f_elem_store = os.path.join(f_store_path, f_elem.path)
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
                f_elem_store = os.path.join(f_store_path, f_elem.path)
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
            f_elem_store = os.path.join(f_store_path, f_elem.path)
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
            f_elem_store = os.path.join(f_store_path, f_elem.path)
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
    sdata._path = Path(store)
    return sdata
