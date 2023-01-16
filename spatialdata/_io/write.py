import os
from collections.abc import Mapping
from typing import Any, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from shapely.io import to_ragged_array
from spatial_image import SpatialImage

from spatialdata._core.core_utils import get_dims, get_transform
from spatialdata._core.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata._io.format import (
    PointsFormat,
    PolygonsFormat,
    ShapesFormat,
    SpatialDataFormatV01,
)

__all__ = ["write_image", "write_labels", "write_points", "write_polygons", "write_table"]


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    coordinate_transformations: list[dict[str, Any]],
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
    fmt: Format = SpatialDataFormatV01(),
) -> None:
    """Write metdata to a group."""
    axes = _get_valid_axes(axes=axes, fmt=fmt)

    group.attrs["@type"] = group_type
    group.attrs["axes"] = axes
    group.attrs["coordinateTransformations"] = coordinate_transformations
    group.attrs["spatialdata_attrs"] = attrs


def write_image(
    image: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    fmt: Format = SpatialDataFormatV01(),
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:

    subgroup = group.require_group(name)
    if isinstance(image, SpatialImage):
        data = image.data
        t = get_transform(image)
        assert isinstance(t, NgffBaseTransformation)
        coordinate_transformations = [[t.to_dict()]]
        chunks = image.chunks
        axes = image.dims
        axes = _get_valid_axes(axes=axes, fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        write_image_ngff(
            image=data,
            group=subgroup,
            scaler=None,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            **metadata,
        )
    elif isinstance(image, MultiscaleSpatialImage):
        data = _iter_multiscale(image, "data")
        coordinate_transformations = [[x.to_dict()] for x in _iter_multiscale(image, "attrs", "transform")]
        chunks = _iter_multiscale(image, "chunks")
        axes_ = _iter_multiscale(image, "dims")
        # TODO: how should axes be handled with multiscale?
        axes = _get_valid_axes(axes=axes_[0], fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multiscale_ngff(
            pyramid=data,
            group=subgroup,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            **metadata,
        )


def write_labels(
    labels: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    fmt: Format = SpatialDataFormatV01(),
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if isinstance(labels, SpatialImage):
        data = labels.data
        t = get_transform(labels)
        assert isinstance(t, NgffBaseTransformation)
        coordinate_transformations = [[t.to_dict()]]
        chunks = labels.chunks
        axes = labels.dims
        axes = _get_valid_axes(axes=axes, fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        write_labels_ngff(
            labels=data,
            name=name,
            group=group,
            scaler=None,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            label_metadata=label_metadata,
            **metadata,
        )
    elif isinstance(labels, MultiscaleSpatialImage):
        data = _iter_multiscale(labels, "data")
        # TODO: nitpick, rewrite the next line to use get_transform()
        coordinate_transformations = [[x.to_dict()] for x in _iter_multiscale(labels, "attrs", "transform")]
        chunks = _iter_multiscale(labels, "chunks")
        axes_ = _iter_multiscale(labels, "dims")
        # TODO: how should axes be handled with multiscale?
        axes = _get_valid_axes(axes=axes_[0], fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multiscale_labels_ngff(
            pyramid=data,
            group=group,
            fmt=fmt,
            axes=axes,
            coordinate_transformations=coordinate_transformations,
            storage_options=storage_options,
            name=name,
            label_metadata=label_metadata,
            **metadata,
        )


def write_polygons(
    polygons: GeoDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:polygons",
    fmt: Format = PolygonsFormat(),
) -> None:
    polygons_groups = group.require_group(name)
    t = get_transform(polygons)
    assert isinstance(t, NgffBaseTransformation)
    coordinate_transformations = [t.to_dict()]

    geometry, coords, offsets = to_ragged_array(polygons.geometry)
    polygons_groups.create_dataset(name="coords", data=coords)
    for i, o in enumerate(offsets):
        polygons_groups.create_dataset(name=f"offset{i}", data=o)
    polygons_groups.create_dataset(name="Index", data=polygons.index.values)

    attrs = fmt.attrs_to_dict(geometry)
    attrs["version"] = fmt.spatialdata_version

    axes = list(get_dims(polygons))

    _write_metadata(
        polygons_groups,
        group_type=group_type,
        coordinate_transformations=coordinate_transformations,
        axes=axes,
        attrs=attrs,
        fmt=fmt,
    )


def write_shapes(
    shapes: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = ShapesFormat(),
) -> None:

    transform = shapes.uns.pop("transform")
    coordinate_transformations = [transform.to_dict()]
    write_adata(group, name, shapes)  # creates group[name]
    shapes.uns["transform"] = transform

    attrs = fmt.attrs_to_dict(shapes.uns)
    attrs["version"] = fmt.spatialdata_version

    axes = list(get_dims(shapes))

    shapes_group = group[name]
    _write_metadata(
        shapes_group,
        group_type=group_type,
        coordinate_transformations=coordinate_transformations,
        axes=axes,
        attrs=attrs,
        fmt=fmt,
    )


def write_points(
    points: pa.Table,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = PointsFormat(),
) -> None:
    points_groups = group.require_group(name)
    t = get_transform(points)
    assert isinstance(t, NgffBaseTransformation)
    coordinate_transformations = [t.to_dict()]

    path = os.path.join(points_groups._store.path, points_groups.path, "points.parquet")
    pq.write_table(points, path)

    axes = list(get_dims(points))

    attrs = {}
    attrs["version"] = fmt.spatialdata_version
    (0, get_dims(points))

    _write_metadata(
        points_groups,
        group_type=group_type,
        coordinate_transformations=coordinate_transformations,
        axes=axes,
        attrs=attrs,
        fmt=fmt,
    )


def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    fmt: Format = SpatialDataFormatV01(),
) -> None:
    region = table.uns["spatialdata_attrs"]["region"]
    region_key = table.uns["spatialdata_attrs"].get("region_key", None)
    instance_key = table.uns["spatialdata_attrs"].get("instance_key", None)
    fmt.validate_table(table, region_key, instance_key)
    write_adata(group, name, table)  # creates group[name]
    tables_group = group[name]
    tables_group.attrs["@type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key
    tables_group.attrs["version"] = fmt.spatialdata_version


def _iter_multiscale(
    data: MultiscaleSpatialImage,
    attr: str,
    key: Optional[str] = None,
) -> list[Any]:
    # TODO: put this check also in the validator for raster multiscales
    name = None
    for i in data.keys():
        variables = list(data[i].variables)
        if len(variables) != 1:
            raise ValueError("MultiscaleSpatialImage must have exactly one variable (the variable name is arbitrary)")
        if name is not None:
            if name != variables[0]:
                raise ValueError("MultiscaleSpatialImage must have the same variable name across all levels")
        name = variables[0]
    if key is None:
        return [getattr(data[i][name], attr) for i in data.keys()]
    else:
        return [getattr(data[i][name], attr).get(key) for i in data.keys()]
