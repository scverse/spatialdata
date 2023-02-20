from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any, Literal, Optional, Union

import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from shapely.io import to_ragged_array
from spatial_image import SpatialImage

from spatialdata._core.core_utils import (
    MappingToCoordinateSystem_t,
    ValidAxis_t,
    _get_transformations,
    _get_transformations_xarray,
    _validate_mapping_to_coordinate_system_type,
    get_dims,
)
from spatialdata._core.models import ShapesModel
from spatialdata._core.transformations import _get_current_output_axes
from spatialdata._io.format import PointsFormat, ShapesFormat, SpatialDataFormatV01

__all__ = [
    "write_image",
    "write_labels",
    "write_points",
    "write_table",
    "overwrite_coordinate_transformations_non_raster",
    "overwrite_coordinate_transformations_raster",
]


def overwrite_coordinate_transformations_non_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformations: MappingToCoordinateSystem_t
) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    ngff_transformations = []
    for target_coordinate_system, t in transformations.items():
        output_axes = _get_current_output_axes(transformation=t, input_axes=tuple(axes))
        ngff_transformations.append(
            t.to_ngff(
                input_axes=tuple(axes),
                output_axes=tuple(output_axes),
                output_coordinate_system_name=target_coordinate_system,
            ).to_dict()
        )
    group.attrs["coordinateTransformations"] = ngff_transformations


def overwrite_coordinate_transformations_raster(
    group: zarr.Group, axes: tuple[ValidAxis_t, ...], transformations: MappingToCoordinateSystem_t
) -> None:
    _validate_mapping_to_coordinate_system_type(transformations)
    # prepare the transformations in the dict representation
    ngff_transformations = []
    for target_coordinate_system, t in transformations.items():
        output_axes = _get_current_output_axes(transformation=t, input_axes=tuple(axes))
        ngff_transformations.append(
            t.to_ngff(
                input_axes=tuple(axes),
                output_axes=tuple(output_axes),
                output_coordinate_system_name=target_coordinate_system,
            )
        )
    coordinate_transformations = [t.to_dict() for t in ngff_transformations]
    # replace the metadata storage
    multiscales = group.attrs["multiscales"]
    assert len(multiscales) == 1
    multiscale = multiscales[0]
    # the transformation present in multiscale["datasets"] are the ones for the multiscale, so and we leave them intact
    # we update multiscale["coordinateTransformations"] and multiscale["coordinateSystems"]
    # see the first post of https://github.com/scverse/spatialdata/issues/39 for an overview
    # fix the io to follow the NGFF specs, see https://github.com/scverse/spatialdata/issues/114
    multiscale["coordinateTransformations"] = coordinate_transformations
    # multiscale["coordinateSystems"] = [t.output_coordinate_system_name for t in ngff_transformations]
    group.attrs["multiscales"] = multiscales


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    # coordinate_transformations: list[dict[str, Any]],
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
    fmt: Format = SpatialDataFormatV01(),
) -> None:
    """Write metdata to a group."""
    axes = _get_valid_axes(axes=axes, fmt=fmt)

    group.attrs["@type"] = group_type
    group.attrs["axes"] = axes
    # we write empty coordinateTransformations and then overwrite them with overwrite_coordinate_transformations_non_raster()
    group.attrs["coordinateTransformations"] = []
    # group.attrs["coordinateTransformations"] = coordinate_transformations
    group.attrs["spatialdata_attrs"] = attrs


def _write_raster(
    raster_type: Literal["image", "labels"],
    raster_data: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = SpatialDataFormatV01(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    channels_metadata: Optional[JSONDict] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:
    assert raster_type in ["image", "labels"]
    # the argument "name" and "label_metadata" are only used for labels (to be precise, name is used in
    # write_multiscale_ngff() when writing metadata, but name is not used in write_image_ngff(). Maybe this is bug of
    # ome-zarr-py. In any case, we don't need that metadata and we use the argument name so that when we write labels
    # the correct group is created by the ome-zarr-py APIs. For images we do it manually in the function
    # _get_group_for_writing_data()
    if raster_type == "image":
        assert label_metadata is None
    else:
        metadata["name"] = name
        metadata["label_metadata"] = label_metadata

    write_single_scale_ngff = write_image_ngff if raster_type == "image" else write_labels_ngff
    write_multi_scale_ngff = write_multiscale_ngff if raster_type == "image" else write_multiscale_labels_ngff

    def _get_group_for_writing_data() -> zarr.Group:
        if raster_type == "image":
            return group.require_group(name)
        else:
            return group

    def _get_group_for_writing_transformations() -> zarr.Group:
        if raster_type == "image":
            return group.require_group(name)
        else:
            return group["labels"][name]

    # convert channel names to channel metadata
    metadata["omero"] = fmt.channels_to_metadata(raster_data, channels_metadata)

    if isinstance(raster_data, SpatialImage):
        data = raster_data.data
        transformations = _get_transformations(raster_data)
        input_axes: tuple[str, ...] = tuple(raster_data.dims)
        chunks = raster_data.chunks
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        # Scaler needs to be None since we are passing the data already downscaled for the multiscale case.
        # We need this because the argument of write_image_ngff is called image while the argument of
        # write_labels_ngff is called label.
        metadata[raster_type] = data
        write_single_scale_ngff(
            group=_get_group_for_writing_data(),
            scaler=None,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            **metadata,
        )
        assert transformations is not None
        overwrite_coordinate_transformations_raster(
            group=_get_group_for_writing_transformations(), transformations=transformations, axes=input_axes
        )
    elif isinstance(raster_data, MultiscaleSpatialImage):
        data = _iter_multiscale(raster_data, "data")
        list_of_input_axes: list[Any] = _iter_multiscale(raster_data, "dims")
        assert len(set(list_of_input_axes)) == 1
        input_axes = list_of_input_axes[0]
        # saving only the transformations of the first scale
        d = dict(raster_data["scale0"])
        assert len(d) == 1
        xdata = d.values().__iter__().__next__()
        transformations = _get_transformations_xarray(xdata)
        assert transformations is not None
        assert len(transformations) > 0
        chunks = _iter_multiscale(raster_data, "chunks")
        # coords = _iter_multiscale(raster_data, "coords")
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multi_scale_ngff(
            pyramid=data,
            group=_get_group_for_writing_data(),
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            **metadata,
        )
        assert transformations is not None
        overwrite_coordinate_transformations_raster(
            group=_get_group_for_writing_transformations(), transformations=transformations, axes=tuple(input_axes)
        )
    else:
        raise ValueError("Not a valid labels object")


def write_image(
    image: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = SpatialDataFormatV01(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:
    _write_raster(
        raster_type="image",
        raster_data=image,
        group=group,
        name=name,
        fmt=fmt,
        storage_options=storage_options,
        **metadata,
    )


def write_labels(
    labels: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = SpatialDataFormatV01(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    _write_raster(
        raster_type="labels",
        raster_data=labels,
        group=group,
        name=name,
        fmt=fmt,
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )


def write_shapes(
    shapes: GeoDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = ShapesFormat(),
) -> None:
    axes = get_dims(shapes)
    t = _get_transformations(shapes)

    shapes_group = group.require_group(name)
    geometry, coords, offsets = to_ragged_array(shapes.geometry)
    shapes_group.create_dataset(name="coords", data=coords)
    for i, o in enumerate(offsets):
        shapes_group.create_dataset(name=f"offset{i}", data=o)
    shapes_group.create_dataset(name="Index", data=shapes.index.values)
    if geometry.name == "POINT":
        shapes_group.create_dataset(name=ShapesModel.RADIUS_KEY, data=shapes[ShapesModel.RADIUS_KEY].values)

    attrs = fmt.attrs_to_dict(geometry)
    attrs["version"] = fmt.spatialdata_version

    _write_metadata(
        shapes_group,
        group_type=group_type,
        # coordinate_transformations=coordinate_transformations,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    overwrite_coordinate_transformations_non_raster(group=shapes_group, axes=axes, transformations=t)


def write_points(
    points: DaskDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = PointsFormat(),
) -> None:
    axes = get_dims(points)
    t = _get_transformations(points)

    points_groups = group.require_group(name)
    path = Path(points_groups._store.path) / points_groups.path / "points.parquet"
    points.to_parquet(path)

    attrs = fmt.attrs_to_dict(points.attrs)
    attrs["version"] = fmt.spatialdata_version

    _write_metadata(
        points_groups,
        group_type=group_type,
        # coordinate_transformations=coordinate_transformations,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    overwrite_coordinate_transformations_non_raster(group=points_groups, axes=axes, transformations=t)


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
) -> list[Any]:
    # TODO: put this check also in the validator for raster multiscales
    for i in data.keys():
        variables = set(data[i].variables.keys())
        names: Set[str] = variables.difference({"c", "z", "y", "x"})
        if len(names) != 1:
            raise ValueError(f"Invalid variable name: `{names}`.")
    name: str = next(iter(names))
    return [getattr(data[i][name], attr) for i in data.keys()]
