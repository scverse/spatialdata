import logging
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
from spatialdata.models import TableModel


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:
    if isinstance(store, str):
        store = Path(store)

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    shapes = {}

    # read multiscale images
    images_store = store / "images"
    if images_store.exists():
        f = zarr.open(images_store, mode="r")
        for k in f:
            f_elem = f[k].name
            f_elem_store = f"{images_store}{f_elem}"
            images[k] = _read_multiscale(f_elem_store, raster_type="image")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
        labels_store = store / "labels"
        if labels_store.exists():
            f = zarr.open(labels_store, mode="r")
            for k in f:
                f_elem = f[k].name
                f_elem_store = f"{labels_store}{f_elem}"
                labels[k] = _read_multiscale(f_elem_store, raster_type="labels")

    # now read rest of the data
    points_store = store / "points"
    if points_store.exists():
        f = zarr.open(points_store, mode="r")
        for k in f:
            f_elem = f[k].name
            f_elem_store = f"{points_store}{f_elem}"
            points[k] = _read_points(f_elem_store)

    shapes_store = store / "shapes"
    if shapes_store.exists():
        f = zarr.open(shapes_store, mode="r")
        for k in f:
            f_elem = f[k].name
            f_elem_store = f"{shapes_store}{f_elem}"
            shapes[k] = _read_shapes(f_elem_store)

    table_store = store / "table"
    if table_store.exists():
        f = zarr.open(table_store, mode="r")
        for k in f:
            f_elem = f[k].name
            f_elem_store = f"{table_store}{f_elem}"
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
