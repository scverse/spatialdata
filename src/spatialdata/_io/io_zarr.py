import logging
from pathlib import Path
from urlpath import URL
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
from spatialdata.models import TableModel

from spatialdata._logging import logger


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:
    if isinstance(store, zarr.Group):
        f = store
    else:
        f = zarr.open(store, mode="r")

    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    shapes = {}

    # read multiscale images
    if 'images' in f:
        images_store = f["images"]
        for k in images_store:
            f_elem = images_store[k]
            f_elem_store = Path(images_store._store.path) / f_elem.path
            logger.debug(f"Store path for multiscale {f_elem}")
            element = _read_multiscale(f_elem_store, raster_type="image")
            print(element)
            images[k] = element

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        if 'labels' in f:
            labels_store = f["labels"]
            for k in labels_store:
                f_elem = labels_store[k]
                f_elem_store = Path(labels_store._store.path) / f_elem.path
                logger.debug(f"Store path for multiscale {f_elem}")
                labels[k] = _read_multiscale(f_elem_store, raster_type="labels")

    # now read rest of the data
    if 'points' in f:
        points_store = f["points"]
        for k in points_store:
            f_elem = points_store[k]
            f_elem_store = Path(points_store._store.path) / f_elem.path
            logger.debug(f"Store path for points {f_elem_store}")
            points[k] = _read_points(f_elem_store)

    if 'shapes' in f:
        shapes_store = f["shapes"]
        for k in shapes_store:
            f_elem = shapes_store[k]
            f_elem_store = Path(shapes_store._store.path) / f_elem.path
            shapes[k] = _read_shapes(f_elem_store)

    if 'table' in f:
        table_store = f["table"]
        for k in table_store:
            f_elem = table_store[k]
            f_elem_store = Path(table_store._store.path) / f_elem.path
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

    sdata = SpatialData(
        images=images,
        labels=labels,
        points=points,
        shapes=shapes,
        table=table,
    )
    sdata.path = str(store)
    return sdata
