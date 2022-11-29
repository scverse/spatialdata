from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes, _validate_datasets
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from shapely.io import to_ragged_array
from spatial_image import SpatialImage

from spatialdata._io.format import SpatialDataFormat

__all__ = ["write_image", "write_labels", "write_points", "write_polygons", "write_table"]


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    shape: Tuple[int, ...],
    name: str,
    attr: Optional[Mapping[str, Optional[str]]] = None,
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    """Write metdata to a group."""
    dims = len(shape)
    axes = _get_valid_axes(dims, axes, fmt)

    datasets: List[Dict[str, Any]] = []
    datasets.append({"path": name})

    if coordinate_transformations is None:
        # TODO: temporary workaround, report bug to handle empty shapes
        if shape[0] == 0:
            shape = (1, *shape[1:])
        shape = [shape]  # type: ignore[assignment]
        coordinate_transformations = fmt.generate_coordinate_transformations(shape)

    fmt.validate_coordinate_transformations(dims, 1, coordinate_transformations)
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    if axes is not None:
        axes = _get_valid_axes(axes=axes, fmt=fmt)
        if axes is not None:
            ndim = len(axes)

    multiscales = [
        dict(
            version=fmt.version,
            datasets=_validate_datasets(datasets, ndim, fmt),
            **metadata,
        )
    ]
    if axes is not None:
        multiscales[0]["axes"] = axes

    group.attrs["@type"] = group_type
    group.attrs["multiscales"] = multiscales
    group.attrs["spatialdata_attrs"] = attr


def write_image(
    image: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:

    if isinstance(image, SpatialImage):
        data = image.data
        coordinate_transformations = [[image.attrs.get("transform").to_dict()]]
        chunks = image.chunks
        axes = image.dims
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        write_image_ngff(
            image=data,
            group=group,
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
            group=group,
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
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if isinstance(labels, SpatialImage):
        data = labels.data
        coordinate_transformations = [[labels.attrs.get("transform").to_dict()]]
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
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("polygons")
    polygons_groups = sub_group.require_group(name)
    coordinate_transformations = [[polygons.attrs.get("transform").to_dict()]]
    print(coordinate_transformations)

    # TODO: save everything or just polygons?
    geometry, coords, offsets = to_ragged_array(polygons.geometry)
    polygons_groups.create_dataset(name="coords", data=coords)
    polygons_groups.create_dataset(name="offsets", data=offsets)
    # polygons_groups = sub_group[name]
    attr = {"geos": {"geometry_name": geometry.name, "geometry_type": geometry.value}}

    _write_metadata(
        polygons_groups,
        group_type=group_type,
        shape=coords.shape,
        attr=attr,  # type: ignore[arg-type]
        name=name,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,
    )


def write_shapes(
    shapes: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("shapes")
    transform = shapes.uns.pop("transform")
    coordinate_transformations = [[transform.to_dict()]]
    attr = shapes.uns.get("spatialdata_attrs")
    write_adata(sub_group, name, shapes)
    shapes_group = sub_group[name]
    shapes.uns["transform"] = transform

    _write_metadata(
        shapes_group,
        group_type=group_type,
        shape=shapes.obsm["spatial"].shape,
        attr=attr,
        name=name,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,
    )


def write_points(
    points: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("points")
    transform = points.uns.pop("transform")
    coordinate_transformations = [[transform.to_dict()]]
    # attr = points.uns.get("spatialdata_attrs")
    write_adata(sub_group, name, points)
    points.uns["transform"] = transform

    points_group = sub_group[name]
    _write_metadata(
        group=points_group,
        group_type=group_type,
        shape=points.obsm["spatial"].shape,
        # attr=attr,
        name=name,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,  # type: ignore[arg-type]
    )


def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    fmt: Format = SpatialDataFormat(),
) -> None:
    region = table.uns["spatialdata_attrs"]["region"]
    region_key = table.uns["spatialdata_attrs"].get("region_key", None)
    instance_key = table.uns["spatialdata_attrs"].get("instance_key", None)
    fmt.validate_table(table, region_key, instance_key)
    sub_group = group.require_group("table")
    write_adata(sub_group, name, table)
    tables_group = sub_group[name]
    tables_group.attrs["@type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key


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
