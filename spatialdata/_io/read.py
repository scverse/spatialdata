import os
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from shapely.io import from_ragged_array
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.core_utils import TRANSFORM_KEY, _set_transform
from spatialdata._core.models import TableModel
from spatialdata._core.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata._core.transformations import BaseTransformation
from spatialdata._io.format import (
    PointsFormat,
    PolygonsFormat,
    ShapesFormat,
    SpatialDataFormatV01,
)
from spatialdata._logging import logger


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, str):
        store = Path(store)

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    polygons = {}
    shapes = {}

    # read multiscale images
    images_store = store / "images"
    if images_store.exists():
        f = zarr.open(images_store, mode="r")
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{images_store}{f_elem}"
            images[k] = _read_multiscale(f_elem_store, raster_type="image")

    # read multiscale labels
    labels_store = store / "labels"
    if labels_store.exists():
        f = zarr.open(labels_store, mode="r")
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{labels_store}{f_elem}"
            labels[k] = _read_multiscale(f_elem_store, raster_type="labels")

    # now read rest of the data
    points_store = store / "points"
    if points_store.exists():
        f = zarr.open(points_store, mode="r")
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{points_store}{f_elem}"
            points[k] = _read_points(f_elem_store)

    shapes_store = store / "shapes"
    if shapes_store.exists():
        f = zarr.open(shapes_store, mode="r")
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{shapes_store}{f_elem}"
            shapes[k] = _read_shapes(f_elem_store)

    polygons_store = store / "polygons"
    if polygons_store.exists():
        f = zarr.open(polygons_store, mode="r")
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{polygons_store}{f_elem}"
            polygons[k] = _read_polygons(f_elem_store)

    table_store = store / "table"
    if table_store.exists():
        f = zarr.open(table_store, mode="r")
        for k in f.keys():
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
        polygons=polygons,
        shapes=shapes,
        table=table,
    )
    sdata.path = str(store)
    return sdata


def _read_multiscale(
    store: str, raster_type: Literal["image", "labels"], fmt: SpatialDataFormatV01 = SpatialDataFormatV01()
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    assert isinstance(store, str)
    assert raster_type in ["image", "labels"]
    nodes: list[Node] = []
    image_loc = ZarrLocation(store)
    if image_loc.exists():
        image_reader = Reader(image_loc)()
        image_nodes = list(image_reader)
        if len(image_nodes):
            for node in image_nodes:
                if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and (
                    raster_type == "image"
                    and np.all([not isinstance(spec, Label) for spec in node.specs])
                    or raster_type == "labels"
                    and np.any([isinstance(spec, Label) for spec in node.specs])
                ):
                    nodes.append(node)
    assert len(nodes) == 1
    node = nodes[0]
    datasets = node.load(Multiscales).datasets
    ngff_transformations = [NgffBaseTransformation.from_dict(t[0]) for t in node.metadata["coordinateTransformations"]]
    transformations = [BaseTransformation.from_ngff(t) for t in ngff_transformations]
    assert len(transformations) == len(datasets), "Expecting one transformation per dataset."
    name = node.metadata["name"]
    if type(name) == list:
        assert len(name) == 1
        name = name[0]
        logger.warning(
            "omero metadata is not fully supported yet, using a workaround. If you encounter bugs related "
            "to omero metadata please follow the discussion at https://github.com/scverse/spatialdata/issues/60"
        )
    axes = [i["name"] for i in node.metadata["axes"]]
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
        msi = MultiscaleSpatialImage.from_dict(multiscale_image)
        return msi
    else:
        t = transformations[0]
        data = node.load(Multiscales).array(resolution=datasets[0], version=fmt.version)
        return SpatialImage(
            data,
            name=name,
            dims=axes,
            attrs={TRANSFORM_KEY: t},
        )


def _read_polygons(store: str, fmt: SpatialDataFormatV01 = PolygonsFormat()) -> GeoDataFrame:  # type: ignore[type-arg]
    """Read polygons from a zarr store."""
    assert isinstance(store, str)
    f = zarr.open(store, mode="r")

    coords = np.array(f["coords"])
    index = np.array(f["Index"])
    offsets_keys = [k for k in f.keys() if k.startswith("offset")]
    offsets = tuple(np.array(f[k]).flatten() for k in offsets_keys)

    typ = fmt.attrs_from_dict(f.attrs.asdict())

    ngff_transform = NgffBaseTransformation.from_dict(f.attrs.asdict()["coordinateTransformations"][0])
    transform = BaseTransformation.from_ngff(ngff_transform)

    geometry = from_ragged_array(typ, coords, offsets)

    geo_df = GeoDataFrame({"geometry": geometry}, index=index)
    _set_transform(geo_df, transform)
    return geo_df


def _read_shapes(store: str, fmt: SpatialDataFormatV01 = ShapesFormat()) -> AnnData:  # type: ignore[type-arg]
    """Read shapes from a zarr store."""
    assert isinstance(store, str)
    f = zarr.open(store, mode="r")

    ngff_transform = NgffBaseTransformation.from_dict(f.attrs.asdict()["coordinateTransformations"][0])
    transform = BaseTransformation.from_ngff(ngff_transform)
    attrs = fmt.attrs_from_dict(f.attrs.asdict())

    adata = read_anndata_zarr(store)

    _set_transform(adata, transform)
    assert adata.uns["spatialdata_attrs"] == attrs

    return adata


def _read_points(store: str, fmt: SpatialDataFormatV01 = PointsFormat()) -> pa.Table:  # type: ignore[type-arg]
    """Read points from a zarr store."""
    assert isinstance(store, str)
    f = zarr.open(store, mode="r")

    path = os.path.join(f._store.path, f.path, "points.parquet")
    table = pq.read_table(path)

    ngff_transform = NgffBaseTransformation.from_dict(f.attrs.asdict()["coordinateTransformations"][0])
    transform = BaseTransformation.from_ngff(ngff_transform)

    new_table = _set_transform(table, transform)
    return new_table
