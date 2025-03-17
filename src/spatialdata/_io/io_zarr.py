from __future__ import annotations

import logging
import warnings
from json import JSONDecodeError
from typing import TYPE_CHECKING, Literal

import zarr
from anndata import AnnData
from pyarrow import ArrowInvalid
from zarr.errors import ArrayNotFoundError, MetadataError

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import (
    BadFileHandleMethod,
    StoreLike,
    _create_upath,
    _open_zarr_store,
    handle_read_errors,
    ome_zarr_logger,
)
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger

if TYPE_CHECKING:
    from dask.dataframe import DataFrame as DaskDataFrame
    from geopandas import GeoDataFrame
    from xarray import DataArray, DataTree


def is_hidden_zarr_entry(name: str) -> bool:
    """Skip hidden files like '.zgroup' or '.zmetadata'."""
    return name.rpartition("/")[2].startswith(".")


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


def read_points_element(path: StoreLike) -> DaskDataFrame:
    store = _open_zarr_store(path)
    return _read_points(store)


def read_shapes_element(path: StoreLike) -> GeoDataFrame:
    store = _open_zarr_store(path)
    return _read_shapes(store)


def read_tables_element(
    zarr_store_path: StoreLike,
    group: zarr.Group,
    subgroup: zarr.Group,
    tables: dict[str, AnnData],
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN] = BadFileHandleMethod.ERROR,
) -> dict[str, AnnData]:
    store = _open_zarr_store(zarr_store_path)
    return _read_table(
        store,
        group,
        subgroup,
        tables,
        on_bad_files,
    )


def read_zarr(
    store: StoreLike,
    selection: None | tuple[str] = None,
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN] = BadFileHandleMethod.ERROR,
) -> SpatialData:
    """
    Read a SpatialData dataset from a zarr store (on-disk or remote).

    Parameters
    ----------
    store
        Path to the zarr store (on-disk or remote) or a zarr.Group object.

    selection
        List of elements to read from the zarr store (images, labels, points, shapes, table). If None, all elements are
        read.

    on_bad_files
        Specifies what to do upon encountering a bad file, e.g. corrupted, invalid or missing files.
        Allowed values are :

        - 'error', raise an exception when a bad file is encountered. Reading aborts immediately
          with an error.
        - 'warn', raise a warning when a bad file is encountered and skip that file. A SpatialData
          object is returned containing only elements that could be read. Failures can only be
          determined from the warnings.

    Returns
    -------
    A SpatialData object.
    """
    _store = _open_zarr_store(store)
    f = zarr.group(_store)

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
        with handle_read_errors(
            on_bad_files,
            location="images",
            exc_types=(JSONDecodeError, MetadataError),
        ):
            group = f["images"]
            count = 0
            for subgroup_name in group:
                if is_hidden_zarr_entry(subgroup_name):
                    continue
                f_elem = group[subgroup_name]
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        JSONDecodeError,  # JSON parse error
                        ValueError,  # ome_zarr: Unable to read the NGFF file
                        KeyError,  # Missing JSON key
                        ArrayNotFoundError,  # Image chunks missing
                        TypeError,  # instead of ArrayNotFoundError, with dask>=2024.10.0 zarr<=2.18.3
                    ),
                ):
                    element = read_image_element(f_elem)
                    images[subgroup_name] = element
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if "labels" in selector and "labels" in f:
            with handle_read_errors(
                on_bad_files,
                location="labels",
                exc_types=(JSONDecodeError, MetadataError),
            ):
                group = f["labels"]
                count = 0
                for subgroup_name in group:
                    if is_hidden_zarr_entry(subgroup_name):
                        continue
                    f_elem = group[subgroup_name]
                    with handle_read_errors(
                        on_bad_files,
                        location=f"{group.path}/{subgroup_name}",
                        exc_types=(JSONDecodeError, KeyError, ValueError, ArrayNotFoundError, TypeError),
                    ):
                        labels[subgroup_name] = read_labels_element(f_elem)
                        count += 1
                logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in f:
        with handle_read_errors(
            on_bad_files,
            location="points",
            exc_types=(JSONDecodeError, MetadataError),
        ):
            group = f["points"]
            count = 0
            for subgroup_name in group:
                f_elem = group[subgroup_name]
                if is_hidden_zarr_entry(subgroup_name):
                    continue
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(JSONDecodeError, KeyError, ArrowInvalid),
                ):
                    points[subgroup_name] = read_points_element(f_elem)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in f:
        with handle_read_errors(
            on_bad_files,
            location="shapes",
            exc_types=(JSONDecodeError, MetadataError),
        ):
            group = f["shapes"]
            count = 0
            for subgroup_name in group:
                if is_hidden_zarr_entry(subgroup_name):
                    continue
                f_elem = group[subgroup_name]
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        JSONDecodeError,
                        ValueError,
                        KeyError,
                        ArrayNotFoundError,
                    ),
                ):
                    shapes[subgroup_name] = read_shapes_element(f_elem)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in f:
        with handle_read_errors(
            on_bad_files,
            location="tables",
            exc_types=(JSONDecodeError, MetadataError),
        ):
            group = f["tables"]
            tables = read_tables_element(_store, f, group, tables, on_bad_files=on_bad_files)

    if "table" in selector and "table" in f:
        warnings.warn(
            f"Table group found in zarr store at location {store}. Please update the zarr store to use tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        subgroup_name = "table"
        with handle_read_errors(
            on_bad_files,
            location=subgroup_name,
            exc_types=(JSONDecodeError, MetadataError),
        ):
            group = f[subgroup_name]
            tables = read_tables_element(store, f, group, tables, on_bad_files=on_bad_files)

            logger.debug(f"Found {count} elements in {group}")

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
    sdata.path = _create_upath(_store)
    return sdata
