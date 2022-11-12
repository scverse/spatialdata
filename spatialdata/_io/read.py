import os
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from shapely import GeometryType
from shapely.io import from_ragged_array
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.transformations import (
    BaseTransformation,
    get_transformation_from_dict,
)
from spatialdata._io.format import SpatialDataFormat


def _read_multiscale(node: Node, fmt: SpatialDataFormat) -> Union[SpatialImage, MultiscaleSpatialImage]:
    datasets = node.load(Multiscales).datasets
    transformations = [get_transformation_from_dict(t[0]) for t in node.metadata["coordinateTransformations"]]
    name = node.metadata["name"]
    axes = [i["name"] for i in node.metadata["axes"]]
    assert len(transformations) == len(datasets), "Expecting one transformation per dataset."
    if len(datasets) > 1:
        multiscale_image = {}
        for i, (t, d) in enumerate(zip(transformations, datasets)):
            data = node.load(Multiscales).array(resolution=d, version=fmt.version)
            multiscale_image[f"scale{i}"] = DataArray(
                data,
                name=name,
                dims=axes,
                attrs={"transform": t},
            )
        return MultiscaleSpatialImage.from_dict(multiscale_image)
    else:
        t = transformations[0]
        data = node.load(Multiscales).array(resolution=datasets[0], version=fmt.version)
        return SpatialImage(
            data,
            name=node.metadata["name"],
            dims=axes,
            attrs={"transform": t},
        )


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, Path):
        store = str(store)

    fmt = SpatialDataFormat()

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    polygons = {}
    labels_transform: Dict[str, Any] = {}
    points_transform = {}

    def _get_transform_from_group(group: zarr.Group) -> BaseTransformation:
        multiscales = group.attrs["multiscales"]
        print(multiscales)
        # TODO: parse info from multiscales['axes']
        assert len(multiscales) == 1, f"TODO: expecting only one multiscale, got {len(multiscales)}"
        datasets = multiscales[0]["datasets"]
        print(datasets)
        assert len(datasets) == 1, "Expecting only one dataset"
        coordinate_transformations = datasets[0]["coordinateTransformations"]
        transformations = [get_transformation_from_dict(t) for t in coordinate_transformations]
        assert len(transformations) == 1, "Expecting only one transformation per multiscale"
        return transformations[0]

    for k in f.keys():
        print(k)
        f_elem = f[k].name
        f_elem_store = f"{store}{f_elem}"
        image_loc = ZarrLocation(f_elem_store)
        image_reader = Reader(image_loc)()
        image_nodes = list(image_reader)
        # read multiscale images that are not labels
        start = time.time()
        if len(image_nodes):
            for node in image_nodes:
                if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and np.all(
                    [not isinstance(spec, Label) for spec in node.specs]
                ):
                    print(f"action0: {time.time() - start}")
                    start = time.time()
                    images[k] = _read_multiscale(node, fmt)
                    print(f"action1: {time.time() - start}")
        # read multiscale labels for the level
        # `WARNING  ome_zarr.reader:reader.py:225 no parent found for` is expected
        # since we don't link the image and the label inside .zattrs['image-label']
        labels_loc = ZarrLocation(f"{f_elem_store}/labels")
        if labels_loc.exists():
            labels_reader = Reader(labels_loc)()
            labels_nodes = list(labels_reader)
            start = time.time()
            if len(labels_nodes):
                for node in labels_nodes:
                    if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and np.any(
                        [isinstance(spec, Label) for spec in node.specs]
                    ):
                        print(f"action0: {time.time() - start}")
                        start = time.time()
                        labels[k] = _read_multiscale(node, fmt)
                        print(f"action1: {time.time() - start}")
        # now read rest
        start = time.time()
        g = zarr.open(f_elem_store, mode="r")
        for j in g.keys():
            g_elem = g[j].name
            g_elem_store = f"{f_elem_store}{g_elem}{f_elem}"

            if g_elem == "/points":
                points[k] = read_anndata_zarr(g_elem_store)
                points_transform[k] = _get_transform_from_group(zarr.open(g_elem_store, mode="r"))

            if g_elem == "/polygons":
                polygons[k] = _read_polygons(g_elem_store)

            if g_elem == "/table":
                table = read_anndata_zarr(f"{f_elem_store}{g_elem}")
        print(f"rest: {time.time() - start}")

    return SpatialData(
        images=images,
        labels=labels,
        points=points,
        polygons=polygons,
        table=table,
        labels_transform=labels_transform,
        points_transform=points_transform,
    )


def _read_polygons(store: Union[str, Path, MutableMapping, zarr.Group]) -> GeoDataFrame:  # type: ignore[type-arg]
    """Read polygons from a zarr store."""

    f = zarr.open(store, mode="r")

    coords = np.array(f["coords"])
    offsets = tuple(x.flatten() for x in np.split(np.array(f["offsets"]), 2))  # type: ignore[var-annotated]

    attrs = f.attrs.asdict()["multiscales"][0]["datasets"][0]
    typ = GeometryType(attrs["path"]["geos"]["geometry_type"])
    assert typ.name == attrs["path"]["geos"]["geometry_name"]

    transforms = get_transformation_from_dict(attrs["coordinateTransformations"][0])

    geometry = from_ragged_array(typ, coords, offsets)

    geo_df = GeoDataFrame({"geometry": geometry})
    geo_df.attrs = {"transform": transforms}

    return geo_df


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))


if __name__ == "__main__":
    sdata = SpatialData.read("../../spatialdata-sandbox/nanostring_cosmx/data_small.zarr")
    print(sdata)
    from napari_spatialdata import Interactive

    Interactive(sdata)
