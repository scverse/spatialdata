import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Optional, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from dask.dataframe import read_parquet
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from shapely.io import from_ragged_array
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.core_utils import TRANSFORM_KEY, set_transform
from spatialdata._core.models import TableModel
from spatialdata._core.transformations import BaseTransformation
from spatialdata._io._utils import ome_zarr_logger
from spatialdata._io.format import PointsFormat, ShapesFormat, SpatialDataFormatV01
from spatialdata._logging import logger


def _read_multiscale(node: Node, fmt: SpatialDataFormatV01) -> Union[SpatialImage, MultiscaleSpatialImage]:
    datasets = node.load(Multiscales).datasets
    transformations = [BaseTransformation.from_dict(t[0]) for t in node.metadata["coordinateTransformations"]]
    name = node.metadata["name"]
    if type(name) == list:
        assert len(name) == 1
        name = name[0]
        logger.warning(
            "omero metadata is not fully supported yet, using a workaround. If you encounter bugs related "
            "to omero metadata please follow the discussion at https://github.com/scverse/spatialdata/issues/60"
        )
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
        msi = MultiscaleSpatialImage.from_dict(multiscale_image)
        # for some reasons if we put attrs={"transform": t} in the dict above, it does not get copied to
        # MultiscaleSpatialImage. We put it also above otherwise we get a schema error
        # TODO: think if we can/want to do something about this
        t = transformations[0]
        set_transform(msi, t)
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


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, str):
        store = Path(store)

    fmt = SpatialDataFormatV01()

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
            image_loc = ZarrLocation(f_elem_store)
            if image_loc.exists():
                image_reader = Reader(image_loc)()
                image_nodes = list(image_reader)
                if len(image_nodes):
                    for node in image_nodes:
                        if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and np.all(
                            [not isinstance(spec, Label) for spec in node.specs]
                        ):
                            images[k] = _read_multiscale(node, fmt)

    # read multiscale labels
    # `WARNING  ome_zarr.reader:reader.py:225 no parent found for` is expected
    # since we don't link the image and the label inside .zattrs['image-label']
    with ome_zarr_logger(logging.ERROR):
        labels_store = store / "labels"
        if labels_store.exists():
            f = zarr.open(labels_store, mode="r")
            for k in f.keys():
                f_elem = f[k].name
                f_elem_store = f"{labels_store}{f_elem}"
                labels_loc = ZarrLocation(f_elem_store)
                if labels_loc.exists():
                    labels_reader = Reader(labels_loc)()
                    labels_nodes = list(labels_reader)
                    # time.time()
                    if len(labels_nodes):
                        for node in labels_nodes:
                            if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and np.any(
                                [isinstance(spec, Label) for spec in node.specs]
                            ):
                                labels[k] = _read_multiscale(node, fmt)

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


def _read_shapes(store: Union[str, Path, MutableMapping, zarr.Group], fmt: SpatialDataFormatV01 = ShapesFormat()) -> GeoDataFrame:  # type: ignore[type-arg]
    """Read shapes from a zarr store."""

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

    transforms = BaseTransformation.from_dict(f.attrs.asdict()["coordinateTransformations"][0])
    set_transform(geo_df, transforms)
    return geo_df


def _read_points(
    store: Union[str, Path, MutableMapping, zarr.Group], fmt: SpatialDataFormatV01 = PointsFormat()  # type: ignore[type-arg]
) -> DaskDataFrame:
    """Read points from a zarr store."""
    f = zarr.open(store, mode="r")

    path = Path(f._store.path) / f.path / "points.parquet"
    table = read_parquet(path)

    transforms = BaseTransformation.from_dict(f.attrs.asdict()["coordinateTransformations"][0])

    set_transform(table, transforms)
    attrs = fmt.attrs_from_dict(f.attrs.asdict())
    table.attrs["spatialdata_attrs"] = attrs
    return table
