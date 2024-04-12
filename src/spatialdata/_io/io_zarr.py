import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import zarr
import zarr.storage
from anndata import AnnData
from upath import UPath

from spatialdata._core._utils import _open_zarr_store
from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import ome_zarr_logger, read_table_and_validate
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._logging import logger


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
    if isinstance(store, (zarr.Group)):
        logger.debug("No support for converting zarr.Group to UPath. Using the store object as is.")
        f = store.store
        f_store_path = UPath(f.store.path if isinstance(f, zarr.storage.ConsolidatedMetadataStore) else f.path)
    else:
        f_store_path = UPath(store) if not isinstance(store, UPath) else store
        f = _open_zarr_store(f_store_path)
    root = zarr.group(f)

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    # TODO: remove table once deprecated.
    selector = {"images", "labels", "points", "shapes", "tables", "table"} if not selection else set(selection or [])
    logger.debug(f"Reading selection {selector}")

    # read multiscale images
    if "images" in selector and "images" in root:
        group = root["images"]
        count = 0
        for subgroup_name in group:
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store = _open_zarr_store(f_store_path / f_elem.path)
            element = _read_multiscale(f_elem_store, raster_type="image")
            images[subgroup_name] = element
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if "labels" in selector and "labels" in root:
            group = root["labels"]
            count = 0
            for subgroup_name in group:
                if subgroup_name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                f_elem = group[subgroup_name]
                f_elem_store = _open_zarr_store(f_store_path / f_elem.path)
                labels[subgroup_name] = _read_multiscale(f_elem_store, raster_type="labels")
                count += 1
            logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in root:
        group = root["points"]
        count = 0
        for subgroup_name in group:
            f_elem = group[subgroup_name]
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            points[subgroup_name] = _read_points(f_store_path / f_elem.path)
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in root:
        group = root["shapes"]
        count = 0
        for subgroup_name in group:
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store = _open_zarr_store(f_store_path / f_elem.path)
            shapes[subgroup_name] = _read_shapes(f_elem_store)
            count += 1
        logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in root:
        group = root["tables"]
        tables = read_table_and_validate(f_store_path, group, tables)

    if "table" in selector and "table" in root:
        warnings.warn(
            f"Table group found in zarr store at location {f_store_path}. Please update the zarr store"
            f"to use tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        subgroup_name = "table"
        group = f[subgroup_name]
        tables = read_table_and_validate(f_store_path, group, tables)
        logger.debug(f"Found {count} elements in {group}")

    sdata = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        tables=tables,
    )
    sdata.path = f_store_path
    return sdata
