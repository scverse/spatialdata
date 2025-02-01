import logging
import warnings

import zarr
import zarr.storage
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import StoreLike, _create_upath, _open_zarr_store, ome_zarr_logger
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger


def read_image_element(path: StoreLike) -> DataArray | DataTree:
    """Read a single image element from a store location.

    Parameters
    ----------
    path
        Path to the zarr store.

    Returns
    -------
    A DataArray or DataTree object.
    """
    store = _open_zarr_store(path)
    return _read_multiscale(store, raster_type="image")


def read_labels_element(path: StoreLike) -> DataArray | DataTree:
    """Read a single image element from a store location.

    Parameters
    ----------
    path
        Path to the zarr store.

    Returns
    -------
    A DataArray or DataTree object.
    """
    store = _open_zarr_store(path)
    return _read_multiscale(store, raster_type="labels")


def read_points_element() -> DaskDataFrame:
    pass


def read_shapes_element() -> GeoDataFrame:
    pass


def read_table_element() -> AnnData:
    pass


def read_zarr(store_like: StoreLike, selection: None | tuple[str] = None) -> SpatialData:
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
    store = _open_zarr_store(store_like)
    f = zarr.group(store)
    # TODO: remove table once deprecated.
    selector = {"images", "labels", "points", "shapes", "tables", "table"} if not selection else set(selection or [])

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    logger.debug(f"Reading selection {selector}")

    # read multiscale images
    if "images" in selector and "images" in f:
        group = f.images
        count = 0
        for subgroup_name in group:
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            element = read_image_element(f_elem)
            images[subgroup_name] = element
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if "labels" in selector and "labels" in f:
            group = f.labels
            count = 0
            for subgroup_name in group:
                if subgroup_name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                f_elem = group[subgroup_name]
                labels[subgroup_name] = read_labels_element(f_elem)
                count += 1
            logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in f:
        group = f["points"]
        count = 0
        for subgroup_name in group:
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            points[subgroup_name] = _read_points(f_elem)
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in f:
        group = f["shapes"]
        count = 0
        for subgroup_name in group:
            if subgroup_name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            shapes[subgroup_name] = _read_shapes(f_elem)
            count += 1
        logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in f:
        group = f["tables"]
        tables = _read_table(zarr_store_path=str(store_like), group=f, subgroup=group, tables=tables)

    if "table" in selector and "table" in f:
        warnings.warn(
            f"Table group found in zarr store at location {f}. Please update the zarr store to use tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        subgroup_name = "table"
        group = f[subgroup_name]
        tables = _read_table(zarr_store_path=store_like, group=f, subgroup=group, tables=tables)

    # read attrs metadata
    attrs = f.attrs.asdict()
    if "spatialdata_attrs" in attrs:
        # when refactoring the read_zarr function into reading componenets separately (and according to the version),
        # we can move the code below (.pop()) into attrs_from_dict()
        attrs.pop("spatialdata_attrs")
    else:
        attrs = None

    sdata = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        tables=tables,
        attrs=attrs,
    )
    # TODO: create a UPath object from any StoreLike object
    sdata.path = _create_upath(store_like)
    return sdata
