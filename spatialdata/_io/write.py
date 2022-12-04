from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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

from spatialdata._core.core_utils import get_transform
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
    shape: Tuple[int, ...],
    coordinate_transformations: List[Dict[str, Any]],
    attrs: Optional[Mapping[str, Any]] = None,
    fmt: Format = SpatialDataFormatV01(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
) -> None:
    """Write metdata to a group."""
    dims = len(shape)
    axes = _get_valid_axes(dims, axes, fmt)

    if axes is not None:
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
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:

    subgroup = group.require_group(name)
    if isinstance(image, SpatialImage):
        data = image.data
        coordinate_transformations = [[get_transform(image).to_dict()]]
        chunks = image.chunks
        axes = image.dims
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
        data = _iter_multiscale(image, name, "data")
        coordinate_transformations = [[x.to_dict()] for x in _iter_multiscale(image, name, "attrs", "transform")]
        chunks = _iter_multiscale(image, name, "chunks")
        axes_ = _iter_multiscale(image, name, "dims")
        # TODO: how should axes be handled with multiscale?
        axes = _get_valid_axes(ndim=data[0].ndim, axes=axes_[0])
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
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if isinstance(labels, SpatialImage):
        data = labels.data
        coordinate_transformations = [[get_transform(labels).to_dict()]]
        chunks = labels.chunks
        axes = labels.dims
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
        data = _iter_multiscale(labels, name, "data")
        # TODO: nitpick, rewrite the next line to use get_transform()
        coordinate_transformations = [[x.to_dict()] for x in _iter_multiscale(labels, name, "attrs", "transform")]
        chunks = _iter_multiscale(labels, name, "chunks")
        axes_ = _iter_multiscale(labels, name, "dims")
        # TODO: how should axes be handled with multiscale?
        axes = _get_valid_axes(ndim=data[0].ndim, axes=axes_[0])
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
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
) -> None:
    polygons_groups = group.require_group(name)
    coordinate_transformations = [get_transform(polygons).to_dict()]

    geometry, coords, offsets = to_ragged_array(polygons.geometry)
    polygons_groups.create_dataset(name="coords", data=coords)
    for i, o in enumerate(offsets):
        polygons_groups.create_dataset(name=f"offset{i}", data=o)
    polygons_groups.create_dataset(name="Index", data=polygons.index.values)

    attrs = fmt.attrs_to_dict(geometry)
    attrs["version"] = fmt.spatialdata_version

    _write_metadata(
        polygons_groups,
        group_type=group_type,
        shape=coords.shape,
        coordinate_transformations=coordinate_transformations,
        attrs=attrs,
        fmt=fmt,
        axes=axes,
    )


def write_shapes(
    shapes: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = ShapesFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
) -> None:

    transform = shapes.uns.pop("transform")
    coordinate_transformations = [transform.to_dict()]
    write_adata(group, name, shapes)  # creates group[name]
    shapes.uns["transform"] = transform

    attrs = fmt.attrs_to_dict(shapes.uns)
    attrs["version"] = fmt.spatialdata_version

    shapes_group = group[name]
    _write_metadata(
        shapes_group,
        group_type=group_type,
        shape=shapes.obsm["spatial"].shape,
        coordinate_transformations=coordinate_transformations,
        attrs=attrs,
        fmt=fmt,
        axes=axes,
    )


def write_points(
    points: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = PointsFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
) -> None:
    transform = points.uns.pop("transform")
    coordinate_transformations = [transform.to_dict()]
    write_adata(group, name, points)  # creates group[name]
    points.uns["transform"] = transform
    points_group = group[name]

    attrs = {"version": fmt.spatialdata_version}

    _write_metadata(
        group=points_group,
        group_type=group_type,
        shape=points.obsm["spatial"].shape,
        coordinate_transformations=coordinate_transformations,
        attrs=attrs,
        fmt=fmt,
        axes=axes,
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
    name: str,
    attr: str,
    key: Optional[str] = None,
) -> List[Any]:
    if key is None:
        return [getattr(data[i][name], attr) for i in data.keys()]
    else:
        return [getattr(data[i][name], attr).get(key) for i in data.keys()]
