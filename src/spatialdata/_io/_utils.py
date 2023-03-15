import logging
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any, Optional, Union

import zarr
from multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.writer import _get_valid_axes

from spatialdata._core.core_utils import (
    MappingToCoordinateSystem_t,
    ValidAxis_t,
    _validate_mapping_to_coordinate_system_type,
)
from spatialdata._core.ngff.ngff_transformations import NgffBaseTransformation
from spatialdata._core.transformations import (
    BaseTransformation,
    _get_current_output_axes,
)


# suppress logger debug from ome_zarr with context manager
@contextmanager
def ome_zarr_logger(level: Any) -> Generator[None, None, None]:
    logger = logging.getLogger("ome_zarr")
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)


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
    fmt: Format,
    axes: Optional[Union[str, list[str], list[dict[str, str]]]] = None,
    attrs: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write metdata to a group."""
    axes = _get_valid_axes(axes=axes, fmt=fmt)

    group.attrs["encoding-type"] = group_type
    group.attrs["axes"] = axes
    # we write empty coordinateTransformations and then overwrite them with overwrite_coordinate_transformations_non_raster()
    group.attrs["coordinateTransformations"] = []
    # group.attrs["coordinateTransformations"] = coordinate_transformations
    group.attrs["spatialdata_attrs"] = attrs


def _iter_multiscale(
    data: MultiscaleSpatialImage,
    attr: Optional[str],
) -> list[Any]:
    # TODO: put this check also in the validator for raster multiscales
    for i in data.keys():
        variables = set(data[i].variables.keys())
        names: set[str] = variables.difference({"c", "z", "y", "x"})
        if len(names) != 1:
            raise ValueError(f"Invalid variable name: `{names}`.")
    name: str = next(iter(names))
    if attr is not None:
        return [getattr(data[i][name], attr) for i in data.keys()]
    else:
        return [data[i][name] for i in data.keys()]
