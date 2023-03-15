from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Union

from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata.transformations.transformations import BaseTransformation

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame]
TRANSFORM_KEY = "transform"
DEFAULT_COORDINATE_SYSTEM = "global"
# ValidAxis_t = Literal["c", "x", "y", "z"]
ValidAxis_t = str
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
C = "c"
Z = "z"
Y = "y"
X = "x"


# mypy says that we can't do isinstance(something, SpatialElement), even if the code works fine in my machine. Since the solution described here don't work: https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6, I am just using the function below
def has_type_spatial_element(e: Any) -> bool:
    """
    Check if the object has the type of a SpatialElement

    Parameters
    ----------
    e
        The input object

    Returns
    -------
    Whether the object is a SpatialElement (i.e in Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame])
    """
    return isinstance(e, (SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame))


# added this code as part of a refactoring to catch errors earlier
def _validate_mapping_to_coordinate_system_type(transformations: Optional[MappingToCoordinateSystem_t]) -> None:
    if not (
        transformations is None
        or isinstance(transformations, dict)
        and all(isinstance(k, str) and isinstance(v, BaseTransformation) for k, v in transformations.items())
    ):
        raise TypeError(
            f"Transform must be of type {MappingToCoordinateSystem_t} or None, but is of type {type(transformations)}."
        )


def validate_axis_name(axis: ValidAxis_t) -> None:
    """
    Check if the axis name is valid

    Parameters
    ----------
    axis
        The axis name

    Raises
    ------
    TypeError
        If the axis name not in ["c", "x", "y", "z"]
    """
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
    """
    Check if the names of the axes are valid

    Parameters
    ----------
    axis
        The names of the axes

    Raises
    ------
    TypeError
        If not in ["c", "x", "y", "z"]
    """
    for ax in axes:
        validate_axis_name(ax)
    if len(axes) != len(set(axes)):
        raise ValueError("Axes must be unique.")


def get_spatial_axes(axes: tuple[ValidAxis_t, ...]) -> tuple[ValidAxis_t, ...]:
    """
    Get the spatial axes of interest

    Parameters
    ----------
    axes
        Should be a subset of ['x', 'y', 'z', 'c']

    Returns
    -------
    The spatial axes, i.e. the input axes but without 'c'
    """
    validate_axes(axes)
    return tuple(ax for ax in axes if ax in [X, Y, Z])


@singledispatch
def get_axis_names(e: SpatialElement) -> tuple[str, ...]:
    """
    Get the dimensions of a spatial element.

    Parameters
    ----------
    e
        Spatial element

    Returns
    -------
    Dimensions of the spatial element (e.g. ("z", "y", "x"))
    """
    raise TypeError(f"Unsupported type: {type(e)}")
