from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from ome_zarr.format import CurrentFormat, Format
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes, _validate_datasets
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff

from spatialdata._io.format import SpatialDataFormat
from spatialdata._types import ArrayLike

__all__ = ["write_image", "write_labels", "write_points", "write_polygons", "write_table"]


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    shape: Tuple[int, ...],
    attr: Optional[Mapping[str, Optional[str]]] = MappingProxyType({"attr": "X", "key": None}),
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    """Write metdata to a group."""
    dims = len(shape)
    axes = _get_valid_axes(dims, axes, fmt)

    datasets: List[Dict[str, Any]] = []
    datasets.append({"path": attr})

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


def write_points(
    points: AnnData,
    group: zarr.Group,
    name: str,
    points_parameters: Optional[Mapping[str, Any]] = None,
    group_type: str = "ngff:points",
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("points")
    write_adata(sub_group, name, points)
    points_group = sub_group[name]
    # TODO: decide what to do here with additional params for
    if points_parameters is not None:
        points_group.attrs["points_parameters"] = points_parameters
    _write_metadata(
        points_group,
        group_type=group_type,
        shape=points.obsm["spatial"].shape,
        attr={"attr": "X", "key": None},
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,
    )


def write_table(
    tables: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    fmt: Format = SpatialDataFormat(),
    region: Union[str, List[str]] = "features",  # TODO: remove default?
    region_key: Optional[str] = None,
    instance_key: Optional[str] = None,
) -> None:
    fmt.validate_tables(tables, region_key, instance_key)
    # anndata create subgroup from name
    # ome-ngff doesn't, hence difference
    sub_group = group.require_group("table")
    write_adata(sub_group, name, tables)
    tables_group = sub_group[name]
    tables_group.attrs["@type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key


def write_image(
    image: ArrayLike,
    group: zarr.Group,
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = CurrentFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    write_image_ngff(
        image=image,
        group=group,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        **metadata,
    )


def write_labels(
    labels: ArrayLike,
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = CurrentFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if np.prod(labels.shape) == 0:
        # TODO: temporary workaround, report bug to handle empty shapes
        # TODO: consider the different axes, now assuming a 2D image
        coordinate_transformations = fmt.generate_coordinate_transformations(shapes=[(1, 1)])
    write_labels_ngff(
        labels=labels,
        group=group,
        name=name,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )


def write_polygons(
    polygons: ArrayLike,
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = CurrentFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: JSONDict,
) -> None:
    raise NotImplementedError("Polygons IO not implemented yet.")
