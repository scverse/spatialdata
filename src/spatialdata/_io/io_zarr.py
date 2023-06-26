import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr

from spatialdata import SpatialData
from spatialdata._io._utils import ome_zarr_logger
from spatialdata._io.io_points import _read_points
from spatialdata._io.io_raster import _read_multiscale
from spatialdata._io.io_shapes import _read_shapes
from spatialdata._logging import logger
from spatialdata.models import TableModel


def read_zarr(store: Union[str, Path, zarr.Group], selection: Optional[tuple[str]] = None) -> SpatialData:
    f = store if isinstance(store, zarr.Group) else zarr.open(store, mode="r")

    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    shapes = {}

    selector = {"images", "labels", "points", "shapes", "table"} if not selection else set(selection or [])
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
            f_elem_store = os.path.join(group._store.path, f_elem.path)
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
                f_elem_store = os.path.join(group._store.path, f_elem.path)
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
            f_elem_store = os.path.join(group._store.path, f_elem.path)
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
            f_elem_store = os.path.join(group._store.path, f_elem.path)
            shapes[subgroup_name] = _read_shapes(f_elem_store)
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    if "table" in selector and "table" in f:
        group = f["table"]
        count = 0
        for subgroup_name in group:
            if Path(subgroup_name).name.startswith("."):
                # skip hidden files like .zgroup or .zmetadata
                continue
            f_elem = group[subgroup_name]
            f_elem_store = os.path.join(group._store.path, f_elem.path)
            table = read_anndata_zarr(f_elem_store)
            if TableModel.ATTRS_KEY in table.uns:
                # fill out eventual missing attributes that has been omitted because their value was None
                attrs = table.uns[TableModel.ATTRS_KEY]
                if "region" not in attrs:
                    attrs["region"] = None
                if "region_key" not in attrs:
                    attrs["region_key"] = None
                if "instance_key" not in attrs:
                    attrs["instance_key"] = None
                # fix type for region
                if "region" in attrs and isinstance(attrs["region"], np.ndarray):
                    attrs["region"] = attrs["region"].tolist()
            count += 1
        logger.debug(f"Found {count} elements in {group}")

    sdata = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        table=table,
    )
    sdata.path = str(store)
    return sdata
