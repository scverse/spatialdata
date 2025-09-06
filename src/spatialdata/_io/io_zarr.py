import logging
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import zarr.storage
from anndata import AnnData
from pyarrow import ArrowInvalid
from zarr.errors import ArrayNotFoundError, MetadataValidationError

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import (
    BadFileHandleMethod,
    handle_read_errors,
    ome_zarr_logger,
)
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger


# TODO: remove with incoming remote read / write PR
# Not removing this now as it requires substantial extra refactor beyond scope of zarrv3 PR.
def _open_zarr_store(
    store: str | Path | zarr.Group, mode: Literal["r", "r+", "a", "w", "w-"] = "r", use_consolidated: bool | None = None
) -> tuple[zarr.Group, str]:
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
    f = store if isinstance(store, zarr.Group) else zarr.open_group(store, mode=mode, use_consolidated=use_consolidated)
    return f, f.store.root


def read_zarr(
    store: str | Path | zarr.Group,
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
    f, f_store_path = _open_zarr_store(store)

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    # TODO: remove table once deprecated.
    selector = {"images", "labels", "points", "shapes", "tables"} if not selection else set(selection or [])
    logger.debug(f"Reading selection {selector}")

    # We raise OS errors instead for some read errors now as in zarr v3 with some corruptions nothing will be read.
    # related to images / labels.
    if "images" in selector and "images" in f:
        group = f["images"]
        count = 0
        for subgroup_name in group:
            if Path(subgroup_name).name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store = os.path.join(f_store_path, f_elem.path)
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
                f_elem_store = f_store_path / f_elem.path
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        JSONDecodeError,
                        KeyError,
                        ValueError,
                        ArrayNotFoundError,
                        TypeError,
                    ),
                ):
                    labels[subgroup_name] = _read_multiscale(f_elem_store, raster_type="labels")
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in f:
        with handle_read_errors(
            on_bad_files,
            location="points",
            exc_types=(JSONDecodeError, MetadataValidationError),
        ):
            group = f["points"]
            count = 0
            for subgroup_name in group:
                f_elem = group[subgroup_name]
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                f_elem_store = os.path.join(f_store_path, f_elem.path)
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(JSONDecodeError, KeyError, ArrowInvalid),
                ):
                    points[subgroup_name] = _read_points(f_elem_store)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in f:
        with handle_read_errors(
            on_bad_files,
            location="shapes",
            exc_types=(JSONDecodeError, MetadataValidationError),
        ):
            group = f["shapes"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                f_elem = group[subgroup_name]
                f_elem_store = os.path.join(f_store_path, f_elem.path)
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
                    shapes[subgroup_name] = _read_shapes(f_elem_store)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in f:
        with handle_read_errors(
            on_bad_files,
            location="tables",
            exc_types=(JSONDecodeError, MetadataValidationError),
        ):
            group = f["tables"]
            tables = _read_table(f_store_path, group, tables, on_bad_files=on_bad_files)

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
    sdata.path = Path(store)
    return sdata


def _get_groups_for_element(
    zarr_path: Path, element_type: str, element_name: str, use_consolidated: bool = True
) -> tuple[zarr.Group, zarr.Group, zarr.Group]:
    """
    Get the Zarr groups for the root, element_type and element for a specific element.

    The store must exist, but creates the element type group and the element group if they don't exist.

    Parameters
    ----------
    zarr_path
        The path to the Zarr storage.
    element_type
        type of the element; must be in ["images", "labels", "points", "polygons", "shapes", "tables"].
    element_name
        name of the element
    use_consolidated
        whether to open zarr groups using consolidated metadata. This should be false when writing as we open
        zarr groups multiple times when writing an element. If the consolidated metadata store is out of sync with
        what is written on disk this leads to errors.

    Returns
    -------
    either the existing Zarr subgroup or a new one.
    """
    if not isinstance(zarr_path, Path):
        raise ValueError("zarr_path should be a Path object")

    if element_type not in [
        "images",
        "labels",
        "points",
        "polygons",
        "shapes",
        "tables",
    ]:
        raise ValueError(f"Unknown element type {element_type}")
    # TODO: remove local import after remote PR
    from spatialdata._io._utils import _open_zarr_store

    store = _open_zarr_store(zarr_path, mode="r+")

    # When writing, use_consolidated must be set to False. Otherwise, the metadata store
    # can get out of sync with newly added elements (e.g., labels), leading to errors.
    root = zarr.open_group(store=store, mode="a", use_consolidated=use_consolidated)
    element_type_group = root.require_group(element_type)
    element_type_group = zarr.open_group(element_type_group.store_path, mode="a", use_consolidated=use_consolidated)

    element_name_group = element_type_group.require_group(element_name)
    return root, element_type_group, element_name_group


def _group_for_element_exists(zarr_path: Path, element_type: str, element_name: str) -> bool:
    """
    Check if the group for an element exists.

    Parameters
    ----------
    element_type
        type of the element; must be in ["images", "labels", "points", "polygons", "shapes", "tables"].
    element_name
        name of the element

    Returns
    -------
    True if the group exists, False otherwise.
    """
    # TODO: remove local import after remote PR
    from spatialdata._io._utils import _open_zarr_store

    store = _open_zarr_store(zarr_path, mode="r")
    root = zarr.open_group(store=store, mode="r")
    assert element_type in [
        "images",
        "labels",
        "points",
        "polygons",
        "shapes",
        "tables",
    ]
    exists = element_type in root and element_name in root[element_type]
    store.close()
    return exists


def _write_consolidated_metadata(path: Path | str | None) -> None:
    if path is not None:
        f, f_store_path = _open_zarr_store(path, mode="r+", use_consolidated=False)
        zarr.consolidate_metadata(f.store)
        f.store.close()
