import logging
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from dask.dataframe import read_parquet  # type: ignore[attr-defined]
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from shapely.io import from_ragged_array
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.core_utils import (
    MappingToCoordinateSystem_t,
    _set_transformations,
    compute_coordinates,
)
from spatialdata._core.models import TableModel
from spatialdata._core.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata._core.transformations import BaseTransformation
from spatialdata._io._utils import ome_zarr_logger
from spatialdata._io.format import PointsFormat, ShapesFormat, SpatialDataFormatV01


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
        for k in f.keys():
            f_elem = f[k].name
            f_elem_store = f"{images_store}{f_elem}"
            images[k] = _read_multiscale(f_elem_store, raster_type="image")

    # read multiscale labels
    with ome_zarr_logger(logging.ERROR):
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
        shapes=shapes,
        table=table,
    )
    sdata.path = str(store)
    return sdata


def _get_transformations_from_ngff_dict(
    list_of_encoded_ngff_transformations: list[dict[str, Any]]
) -> MappingToCoordinateSystem_t:
    list_of_ngff_transformations = [NgffBaseTransformation.from_dict(d) for d in list_of_encoded_ngff_transformations]
    list_of_transformations = [BaseTransformation.from_ngff(t) for t in list_of_ngff_transformations]
    transformations = {}
    for ngff_t, t in zip(list_of_ngff_transformations, list_of_transformations):
        assert ngff_t.output_coordinate_system is not None
        transformations[ngff_t.output_coordinate_system.name] = t
    return transformations


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
    if len(nodes) != 1:
        raise ValueError(
            f"len(nodes) = {len(nodes)}, expected 1. Unable to read the NGFF file. Please report this "
            f"bug and attach a minimal data example."
        )
    node = nodes[0]
    datasets = node.load(Multiscales).datasets
    multiscales = node.load(Multiscales).zarr.root_attrs["multiscales"]
    assert len(multiscales) == 1
    # checking for multiscales[0]["coordinateTransformations"] would make fail
    # something that doesn't have coordinateTransformations in top level
    # which is true for the current version of the spec
    # and for instance in the xenium example
    encoded_ngff_transformations = multiscales[0]["coordinateTransformations"]
    transformations = _get_transformations_from_ngff_dict(encoded_ngff_transformations)
    name = os.path.basename(node.metadata["name"])
    # if image, read channels metadata
    if raster_type == "image":
        omero = multiscales[0]["omero"]
        channels = fmt.channels_from_metadata(omero)
    axes = [i["name"] for i in node.metadata["axes"]]
    if len(datasets) > 1:
        multiscale_image = {}
        for i, d in enumerate(datasets):
            data = node.load(Multiscales).array(resolution=d, version=fmt.version)
            multiscale_image[f"scale{i}"] = DataArray(
                data,
                name=name,
                dims=axes,
                coords={"c": channels} if raster_type == "image" else {},
                # attrs={"transform": t},
            )
        msi = MultiscaleSpatialImage.from_dict(multiscale_image)
        _set_transformations(msi, transformations)
        return compute_coordinates(msi)
    else:
        data = node.load(Multiscales).array(resolution=datasets[0], version=fmt.version)
        si = SpatialImage(
            data,
            name=name,
            dims=axes,
            coords={"c": channels} if raster_type == "image" else {},
            # attrs={TRANSFORM_KEY: t},
        )
        _set_transformations(si, transformations)
        return compute_coordinates(si)


def _read_shapes(store: Union[str, Path, MutableMapping, zarr.Group], fmt: SpatialDataFormatV01 = ShapesFormat()) -> GeoDataFrame:  # type: ignore[type-arg]
    """Read shapes from a zarr store."""
    assert isinstance(store, str)
    f = zarr.open(store, mode="r")

    coords = np.array(f["coords"])
    index = np.array(f["Index"])
    typ = fmt.attrs_from_dict(f.attrs.asdict())
    if typ.name == "POINT":
        radius = np.array(f["radius"])
        geometry = from_ragged_array(typ, coords)
        geo_df = GeoDataFrame({"geometry": geometry, "radius": radius}, index=index)
    else:
        offsets_keys = [k for k in f.keys() if k.startswith("offset")]
        offsets = tuple(np.array(f[k]).flatten() for k in offsets_keys)
        geometry = from_ragged_array(typ, coords, offsets)
        geo_df = GeoDataFrame({"geometry": geometry}, index=index)

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(geo_df, transformations)
    return geo_df


def _read_points(
    store: Union[str, Path, MutableMapping, zarr.Group], fmt: SpatialDataFormatV01 = PointsFormat()  # type: ignore[type-arg]
) -> DaskDataFrame:
    """Read points from a zarr store."""
    assert isinstance(store, str)
    f = zarr.open(store, mode="r")

    path = Path(f._store.path) / f.path / "points.parquet"
    table = read_parquet(path)
    assert isinstance(table, DaskDataFrame)

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(table, transformations)

    attrs = fmt.attrs_from_dict(f.attrs.asdict())
    if len(attrs):
        table.attrs["spatialdata_attrs"] = attrs
    return table
