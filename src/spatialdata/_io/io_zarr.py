import os
import warnings
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import zarr.storage
from anndata import AnnData
from ome_zarr.format import Format
from pyarrow import ArrowInvalid
from zarr.errors import ArrayNotFoundError

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import (
    BadFileHandleMethod,
    _resolve_zarr_store,
    handle_read_errors,
)
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger


def get_raster_format_for_read(group: zarr.Group, sdata_version: Literal["0.1", "0.2"]) -> Format:
    """Get raster format of stored raster data.

    This checks the image or label element zarr group metadata to retrieve the format that is used by
    ome-zarr's ZarrLocation for reading the data.

    Parameters
    ----------
    group
        The zarr group of the raster element to be read.
    sdata_version
        The version of the SpatialData zarr store retrieved from the spatialdata attributes.

    Returns
    -------
    The ome-zarr format to use for reading the raster element.
    """
    from spatialdata._io.format import SdataVersion_to_Format

    if sdata_version == "0.1":
        group_version = group.metadata.attributes["multiscales"][0]["version"]
    if sdata_version == "0.2":
        group_version = group.metadata.attributes["ome"]["version"]
    return SdataVersion_to_Format[group_version]


def read_zarr(
    store: str | Path,
    selection: None | tuple[str] = None,
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN] = BadFileHandleMethod.ERROR,
) -> SpatialData:
    """
    Read a SpatialData dataset from a zarr store (on-disk or remote).

    Parameters
    ----------
    store
        Path to the zarr store (on-disk or remote).

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
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store)
    root_group = zarr.open_group(resolved_store, mode="r")
    sdata_version = root_group.metadata.attributes["spatialdata_attrs"]["version"]
    root_store_path = root_group.store.root

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    selector = {"images", "labels", "points", "shapes", "tables"} if not selection else set(selection or [])
    logger.debug(f"Reading selection {selector}")

    # We raise OS errors instead for some read errors now as in zarr v3 with some corruptions nothing will be read.
    # related to images / labels.
    with handle_read_errors(
        on_bad_files,
        location="images",
        exc_types=JSONDecodeError,
    ):
        if "images" in selector and "images" in root_group:
            group = root_group["images"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group = group[subgroup_name]
                reader_format = get_raster_format_for_read(elem_group, sdata_version)
                elem_group_path = os.path.join(root_store_path, elem_group.path)
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        KeyError,
                        ArrayNotFoundError,
                        OSError,
                    ),
                ):
                    element = _read_multiscale(elem_group_path, raster_type="image", reader_format=reader_format)
                    images[subgroup_name] = element
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with handle_read_errors(
        on_bad_files,
        location="labels",
        exc_types=JSONDecodeError,
    ):
        if "labels" in selector and "labels" in root_group:
            group = root_group["labels"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group = group[subgroup_name]
                reader_format = get_raster_format_for_read(elem_group, sdata_version)
                elem_group_path = root_store_path / elem_group.path
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        KeyError,
                        ArrayNotFoundError,
                        OSError,
                    ),
                ):
                    labels[subgroup_name] = _read_multiscale(
                        elem_group_path, raster_type="labels", reader_format=reader_format
                    )
                    count += 1
            logger.debug(f"Found {count} elements in {group}")
    # now read rest of the data
    with handle_read_errors(
        on_bad_files,
        location="points",
        exc_types=JSONDecodeError,
    ):
        if "points" in selector and "points" in root_group:
            group = root_group["points"]
            count = 0
            for subgroup_name in group:
                elem_group = group[subgroup_name]
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group_path = os.path.join(root_store_path, elem_group.path)
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(KeyError, ArrowInvalid, JSONDecodeError),
                ):
                    points[subgroup_name] = _read_points(elem_group_path)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    with handle_read_errors(
        on_bad_files,
        location="shapes",
        exc_types=JSONDecodeError,
    ):
        if "shapes" in selector and "shapes" in root_group:
            group = root_group["shapes"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group = group[subgroup_name]
                elem_group_path = os.path.join(root_store_path, elem_group.path)
                with handle_read_errors(
                    on_bad_files,
                    location=f"{group.path}/{subgroup_name}",
                    exc_types=(
                        JSONDecodeError,
                        KeyError,
                        ArrayNotFoundError,
                    ),
                ):
                    shapes[subgroup_name] = _read_shapes(elem_group_path)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")
    if "tables" in selector and "tables" in root_group:
        with handle_read_errors(
            on_bad_files,
            location="tables",
            exc_types=JSONDecodeError,
        ):
            group = root_group["tables"]
            tables = _read_table(root_store_path, group, tables, on_bad_files=on_bad_files)

    # read attrs metadata
    attrs = root_group.attrs.asdict()
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

    The store must exist, but creates the element type group and the element group if they don't exist. In all cases
    the zarr group will also be opened. When writing data to disk this should always be done with 'use_consolidated'
    being 'False'. If a user wrote the data previously with consolidation of the metadata and then they write new data
    in the zarr store, it can give errors otherwise, due to partially written elements not yet being present in the
    consolidated metadata store, e.g. when first writing the element and then opening the zarr group again for writing
    transformations.

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
    The Zarr groups for the root, element_type and element for a specific element.
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
    resolved_store = _resolve_zarr_store(zarr_path)

    # When writing, use_consolidated must be set to False. Otherwise, the metadata store
    # can get out of sync with newly added elements (e.g., labels), leading to errors.
    root_group = zarr.open_group(store=resolved_store, mode="r+", use_consolidated=use_consolidated)
    element_type_group = root_group.require_group(element_type)
    element_type_group = zarr.open_group(element_type_group.store_path, mode="a", use_consolidated=use_consolidated)

    element_name_group = element_type_group.require_group(element_name)
    return root_group, element_type_group, element_name_group


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
    store = _resolve_zarr_store(zarr_path)
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
        f = zarr.open_group(path, mode="r+", use_consolidated=False)
        # .parquet files are not recognized as proper zarr and thus throw a warning. This does not affect SpatialData.
        # and therefore we silence it for our users as they can't do anything about this.
        # TODO check with remote PR whether we can prevent this warning at least for points data and whether with zarrv3
        # that pr would still work.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)
            zarr.consolidate_metadata(f.store)
        f.store.close()
