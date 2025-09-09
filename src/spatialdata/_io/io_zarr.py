import logging
import os
import warnings
from json import JSONDecodeError
from pathlib import Path
from typing import Literal

import zarr.storage
from anndata import AnnData
from pyarrow import ArrowInvalid
from zarr.errors import ArrayNotFoundError, MetadataValidationError

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import BadFileHandleMethod, _resolve_zarr_store, handle_read_errors, ome_zarr_logger
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._io.io_table import _read_table
from spatialdata._logging import logger


# TODO: remove with incoming remote read / write PR
# Not removing this now as it requires substantial extra refactor beyond scope of zarrv3 PR.
def _open_zarr(
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
    from spatialdata._io._utils import _resolve_zarr_store as rzs

    resolved_store = rzs(store)
    root_group, root_store_path = _open_zarr(resolved_store)

    images = {}
    labels = {}
    points = {}
    tables: dict[str, AnnData] = {}
    shapes = {}

    selector = {"images", "labels", "points", "shapes", "tables"} if not selection else set(selection or [])
    logger.debug(f"Reading selection {selector}")

    # We raise OS errors instead for some read errors now as in zarr v3 with some corruptions nothing will be read.
    # related to images / labels.
    if "images" in selector and "images" in root_group:
        group = root_group["images"]
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
                    JSONDecodeError,  # JSON parse error
                    ValueError,  # ome_zarr: Unable to read the NGFF file
                    KeyError,  # Missing JSON key
                    ArrayNotFoundError,  # Image chunks missing
                    TypeError,  # instead of ArrayNotFoundError, with dask>=2024.10.0 zarr<=2.18.3
                ),
            ):
                element = _read_multiscale(elem_group_path, raster_type="image")
                images[subgroup_name] = element
                count += 1
        logger.debug(f"Found {count} elements in {group}")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if "labels" in selector and "labels" in root_group:
            group = root_group["labels"]
            count = 0
            for subgroup_name in group:
                if Path(subgroup_name).name.startswith("."):
                    # skip hidden files like .zgroup or .zmetadata
                    continue
                elem_group = group[subgroup_name]
                elem_group_path = root_store_path / elem_group.path
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
                    labels[subgroup_name] = _read_multiscale(elem_group_path, raster_type="labels")
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    # now read rest of the data
    if "points" in selector and "points" in root_group:
        with handle_read_errors(
            on_bad_files,
            location="points",
            exc_types=(JSONDecodeError, MetadataValidationError),
        ):
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
                    exc_types=(JSONDecodeError, KeyError, ArrowInvalid),
                ):
                    points[subgroup_name] = _read_points(elem_group_path)
                    count += 1
            logger.debug(f"Found {count} elements in {group}")

    if "shapes" in selector and "shapes" in root_group:
        with handle_read_errors(
            on_bad_files,
            location="shapes",
            exc_types=(JSONDecodeError, MetadataValidationError),
        ):
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
                        ValueError,
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
            exc_types=(JSONDecodeError, MetadataValidationError),
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
    store = _resolve_zarr_store(zarr_path, mode="r")
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
        f, f_store_path = _open_zarr(path, mode="r+", use_consolidated=False)
        # .parquet files are not recognized as proper zarr and thus throw a warning. This does not affect SpatialData.
        # and therefore we silence it for our users as they can't do anything about this.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)
            zarr.consolidate_metadata(f.store)
        f.store.close()
