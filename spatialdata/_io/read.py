import os
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Reader

from spatialdata._core.spatialdata import SpatialData
from spatialdata._io.format import SpatialDataFormat


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, Path):
        store = str(store)

    fmt = SpatialDataFormat()

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    polygons: Mapping[str, Any] = {}
    tables = {}

    for k in f.keys():
        f_elem = f[k].name
        f_elem_store = f"{store}{f_elem}"
        loc = ZarrLocation(f_elem_store)
        reader = Reader(loc)()
        nodes = list(reader)
        if len(nodes):
            for node in nodes:
                if np.any([isinstance(spec, Multiscales) for spec in node.specs]):
                    if np.any([isinstance(spec, Label) for spec in node.specs]):
                        labels[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
                    else:
                        images[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
        # read all images/labels for the level
        # now read rest
        g = zarr.open(f_elem_store, mode="r")
        for j in g.keys():
            g_elem = g[j].name
            g_elem_store = f"{f_elem_store}{g_elem}{f_elem}"
            if g_elem == "/points":
                points[k] = read_anndata_zarr(g_elem_store)
            if g_elem == "/table":
                tables[k] = read_anndata_zarr(g_elem_store)

    return SpatialData(images=images, labels=labels, points=points, polygons=polygons, tables=tables)


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))
