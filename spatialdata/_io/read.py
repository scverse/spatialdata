import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Reader

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.transform import Transform
from spatialdata._io.format import SpatialDataFormat
from spatialdata._types import ArrayLike


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, Path):
        store = str(store)

    fmt = SpatialDataFormat()

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    polygons: Mapping[str, Any] = {}
    images_transform = {}
    labels_transform = {}
    points_transform = {}
    polygons_transform: Optional[Mapping[str, Any]] = {}

    def _get_transform_from_group(group: zarr.Group) -> Transform:
        multiscales = group.attrs["multiscales"]
        # TODO: parse info from multiscales['axes']
        assert (
            len(multiscales) == 1
        ), "TODO: expecting only one transformation, but found more. Probably one for each pyramid level"
        datasets = multiscales[0]["datasets"]
        assert len(datasets) == 1, "Expecting only one dataset"
        coordinate_transformations = datasets[0]["coordinateTransformations"]
        assert len(coordinate_transformations) in [1, 2]
        assert coordinate_transformations[0]["type"] == "scale"
        scale: ArrayLike = np.array(coordinate_transformations[0]["scale"])
        translation: ArrayLike
        if len(coordinate_transformations) == 2:
            assert coordinate_transformations[1]["type"] == "translation"
            translation = np.array(coordinate_transformations[1]["translation"])
        else:
            # TODO: assuming ndim=2 for now
            translation = np.array([0, 0])
        transform = Transform(translation=translation, scale_factors=scale)
        return transform

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
                        labels_transform[k] = _get_transform_from_group(zarr.open(loc.path, mode="r"))
                    else:
                        images[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
                        images_transform[k] = _get_transform_from_group(zarr.open(loc.path, mode="r"))
        # read all images/labels for the level
        # now read rest
        g = zarr.open(f_elem_store, mode="r")
        for j in g.keys():
            g_elem = g[j].name
            g_elem_store = f"{f_elem_store}{g_elem}{f_elem}"
            if g_elem == "/points":
                points[k] = read_anndata_zarr(g_elem_store)
                points_transform[k] = _get_transform_from_group(zarr.open(g_elem_store, mode="r"))

            if g_elem == "/table":
                table = read_anndata_zarr(f"{f_elem_store}{g_elem}")

    return SpatialData(
        images=images,
        labels=labels,
        points=points,
        polygons=polygons,
        table=table,
        images_transform=images_transform,
        labels_transform=labels_transform,
        points_transform=points_transform,
        polygons_transform=polygons_transform,
    )


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))
