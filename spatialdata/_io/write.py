from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from ome_zarr.format import Format
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transform import BaseTransformation
from spatialdata._io.format import SpatialDataFormat
from spatialdata._types import ArrayLike

__all__ = ["write_image", "write_labels", "write_points", "write_polygons", "write_table"]


def _get_coordinate_transformations_list(
    coordinate_transformations: Dict[str, BaseTransformation], transformation_input: str
) -> List[Dict[str, Any]]:
    ct_formatted = []
    for target, ct in coordinate_transformations.items():
        d = ct.to_dict()
        d["input"] = transformation_input
        d["output"] = target
        ct_formatted.append(d)
    return ct_formatted


def _get_coordinate_systems_list(coordinate_systems: Dict[str, CoordinateSystem]) -> List[Dict[str, Any]]:
    cs_formatted = [cs.to_dict() for cs in coordinate_systems.values()]
    return cs_formatted


def _write_metadata_points_polygons(
    group: zarr.Group,
    group_type: str,
    shape: Tuple[int, ...],
    coordinate_transformations: Dict[str, BaseTransformation],
    # coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    coordinate_systems: Dict[str, CoordinateSystem],
    implicit_coordinate_system: CoordinateSystem,
    attr: Optional[Mapping[str, Optional[str]]] = MappingProxyType({"attr": "X", "key": None}),
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    """Write metdata to a group."""
    len(shape)
    # axes = _get_valid_axes(dims, axes, fmt)

    assert implicit_coordinate_system.name not in coordinate_systems
    cs_formatted = _get_coordinate_systems_list(
        {implicit_coordinate_system.name: implicit_coordinate_system} | coordinate_systems
    )
    ct_formatted = _get_coordinate_transformations_list(
        coordinate_transformations, transformation_input=f"/{group.path}"
    )
    # TODO: do we need also to add a name?
    multiscales = [
        dict(version="0.5-dev", coordinateSystems=cs_formatted, coordinateTransformations=ct_formatted, **metadata)
    ]
    group.attrs["@type"] = group_type
    group.attrs["multiscales"] = multiscales


def _update_metadata_images_labels(
    group: zarr.Group,
    coordinate_transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    implicit_coordinate_system: CoordinateSystem,
    fmt: Format = SpatialDataFormat(),
) -> None:
    assert "multiscales" in group.attrs
    multiscales = group.attrs["multiscales"].copy()
    assert len(multiscales) == 1
    # this should be deleted as it does not appear in the specs but I am keeping for back compatibility with
    # ome-zarr-py until it is updated
    # del multiscales[0]["axes"]
    assert implicit_coordinate_system.name not in coordinate_systems
    cs_formatted = _get_coordinate_systems_list(
        {implicit_coordinate_system.name: implicit_coordinate_system} | coordinate_systems
    )
    ct_formatted = _get_coordinate_transformations_list(
        coordinate_transformations, transformation_input=f"/{group.path}"
    )
    multiscales[0]["coordinateSystems"] = cs_formatted
    multiscales[0]["coordinateTransformations"] = ct_formatted
    group.attrs["multiscales"] = multiscales
    dict(group.attrs)
    # the list of transformations in datasets (that are exclusiely the ones for the pyramid) is kept as the previous
    # version, because otherwise the reader (ome-zarr-py) would not be able to read the pyramid data


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
    coordinate_transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    # coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    ##
    write_image_ngff(
        image=image,
        group=group,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=[[]],
        storage_options=storage_options,
        **metadata,
    )
    assert not group.path.startswith("/")
    implicit_coordinate_system = CoordinateSystem()
    implicit_coordinate_system.from_dict(
        {"name": f"/{group.path}", "axes": [{"name": ax, "type": "array"} for ax in axes]}
    )
    _update_metadata_images_labels(group, coordinate_transformations, coordinate_systems, implicit_coordinate_system)
    ##


def write_labels(
    labels: ArrayLike,
    group: zarr.Group,
    name: str,
    coordinate_transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    # coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    if np.prod(labels.shape) == 0:
        # TODO: temporary workaround, report bug to handle empty shapes
        # TODO: consider the different axes, now assuming a 2D image
        raise NotImplementedError("legacy code not tested, debug")
        coordinate_transformations = fmt.generate_coordinate_transformations(shapes=[(1, 1)])
    write_labels_ngff(
        labels=labels,
        group=group,
        name=name,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=[[]],
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )
    sub_group = group.require_group(f"labels/{name}")
    assert not sub_group.path.startswith("/")
    implicit_coordinate_system = CoordinateSystem()
    implicit_coordinate_system.from_dict(
        {"name": f"/{sub_group.path}", "axes": [{"name": ax, "type": "array"} for ax in axes]}
    )
    _update_metadata_images_labels(
        sub_group, coordinate_transformations, coordinate_systems, implicit_coordinate_system
    )


def write_points(
    points: AnnData,
    group: zarr.Group,
    name: str,
    coordinate_transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    # coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    points_parameters: Optional[Mapping[str, Any]] = None,
    group_type: str = "ngff:points",
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("points")
    write_adata(sub_group, name, points)
    points_group = sub_group[name]
    # TODO: decide what to do here with additional params for
    if points_parameters is not None:
        points_group.attrs["points_parameters"] = points_parameters
    implicit_coordinate_system = CoordinateSystem()
    implicit_coordinate_system.from_dict(
        {"name": f"/{points_group.path}", "axes": [{"name": ax, "type": "array"} for ax in axes]}
    )
    _write_metadata_points_polygons(
        points_group,
        group_type=group_type,
        shape=points.obsm["spatial"].shape,
        attr={"attr": "X", "key": None},
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        coordinate_systems=coordinate_systems,
        implicit_coordinate_system=implicit_coordinate_system,
        **metadata,
    )


def write_polygons(
    polygons: AnnData,
    group: zarr.Group,
    name: str,
    coordinate_transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    group_type: str = "ngff:polygons",
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    # coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    sub_group = group.require_group("polygons")
    write_adata(sub_group, name, polygons)
    polygons_group = sub_group[name]
    implicit_coordinate_system = CoordinateSystem()
    implicit_coordinate_system.from_dict(
        {"name": f"/{polygons_group.path}", "axes": [{"name": ax, "type": "array"} for ax in axes]}
    )
    _write_metadata_points_polygons(
        polygons_group,
        group_type=group_type,
        shape=(1, 2),  # assuming 2d
        attr={"attr": "X", "key": None},
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        coordinate_systems=coordinate_systems,
        implicit_coordinate_system=implicit_coordinate_system,
        **metadata,
    )
