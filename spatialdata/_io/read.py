import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.transform import BaseTransformation, get_transformation_from_dict
from spatialdata._io.format import SpatialDataFormat


def _read_multiscale(node: Node, fmt: SpatialDataFormat) -> Union[SpatialImage, MultiscaleSpatialImage]:
    datasets = node.load(Multiscales).datasets
    transformations = [get_transformation_from_dict(t[0]) for t in node.metadata["coordinateTransformations"]]
    axes = [i["name"] for i in node.metadata["axes"]]
    assert len(transformations) == len(datasets), "Expecting one transformation per dataset."
    if len(datasets) > 1:
        multiscale_image = {}
        for i, (t, d) in enumerate(zip(transformations, datasets)):
            multiscale_image[f"scale{i}"] = DataArray(
                node.load(Multiscales).array(resolution=d, version=fmt.version),
                name=node.metadata["name"],
                dims=axes,
                attrs={"transformation": t},
            )
        return MultiscaleSpatialImage.from_dict(multiscale_image)
    else:
        return SpatialImage(
            node.load(Multiscales).array(resolution=datasets[0], version=fmt.version),
            name=node.metadata["name"],
            dims=axes,
            attrs={"transformation": t},
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
    labels_transform = {}
    points_transform = {}
    polygons_transform = {}

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
        # read all images/labels for the level
        # warnings like "no parent found for <ome_zarr.reader.Label object at 0x1c789f310>: None" are expected,
        # since we don't link the image and the label inside .zattrs['image-label']
        labels_loc = ZarrLocation(f"{f_elem_store}/labels")
        start = time.time()
        if labels_loc.exists():
            labels_reader = Reader(labels_loc)()
            labels_nodes = list(labels_reader)
            if len(labels_nodes):
                for node in labels_nodes:
                    if np.any([isinstance(spec, Label) for spec in node.specs]):
                        print(f"action0: {time.time() - start}")
                        start = time.time()
                        labels[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
                        labels_transform[k] = _get_transform_from_group(zarr.open(node.zarr.path, mode="r"))
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
                polygons[k] = read_anndata_zarr(g_elem_store)
                polygons_transform[k] = _get_transform_from_group(zarr.open(g_elem_store, mode="r"))

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
        polygons_transform=polygons_transform,
    )


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))


if __name__ == "__main__":
    sdata = SpatialData.read("../../spatialdata-sandbox/nanostring_cosmx/data_small.zarr")
    print(sdata)
    from napari_spatialdata import Interactive

    Interactive(sdata)
