import copy
import json
from functools import singledispatch
from typing import Literal, Optional, Union

import pyarrow as pa
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis, NgffCoordinateSystem
from spatialdata._core.ngff.ngff_transformations import NgffBaseTransformation

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, AnnData, pa.Table]

__all__ = [
    "SpatialElement",
    "TRANSFORM_KEY",
    "get_transform",
    "set_transform",
    "get_default_coordinate_system",
    "get_dims",
    "C",
    "Z",
    "Y",
    "X",
]


TRANSFORM_KEY = "transform"
C, Z, Y, X = "c", "z", "y", "x"
ValidAxis_t = Literal["c", "x", "y", "z"]


def validate_axis_name(axis: ValidAxis_t) -> None:
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


@singledispatch
def get_transform(e: SpatialElement) -> Optional[NgffBaseTransformation]:
    raise TypeError(f"Unsupported type: {type(e)}")


@get_transform.register(SpatialImage)
def _(e: SpatialImage) -> Optional[NgffBaseTransformation]:
    t = e.attrs.get(TRANSFORM_KEY)
    # this double return is to make mypy happy
    if t is not None:
        assert isinstance(t, NgffBaseTransformation)
        return t
    else:
        return t


@get_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> Optional[NgffBaseTransformation]:
    t = e.attrs.get(TRANSFORM_KEY)
    if t is not None:
        assert isinstance(t, NgffBaseTransformation)
        return t
    else:
        return t


@get_transform.register(GeoDataFrame)
def _(e: GeoDataFrame) -> Optional[NgffBaseTransformation]:
    t = e.attrs.get(TRANSFORM_KEY)
    if t is not None:
        assert isinstance(t, NgffBaseTransformation)
        return t
    else:
        return t


@get_transform.register(AnnData)
def _(e: AnnData) -> Optional[NgffBaseTransformation]:
    t = e.uns[TRANSFORM_KEY]
    if t is not None:
        assert isinstance(t, NgffBaseTransformation)
        return t
    else:
        return t


# we need the return type because pa.Table is immutable
@get_transform.register(pa.Table)
def _(e: pa.Table) -> Optional[NgffBaseTransformation]:
    t_bytes = e.schema.metadata[TRANSFORM_KEY.encode("utf-8")]
    t = NgffBaseTransformation.from_dict(json.loads(t_bytes.decode("utf-8")))
    if t is not None:
        assert isinstance(t, NgffBaseTransformation)
        return t
    else:
        return t


def _adjust_transformation_axes(e: SpatialElement, t: NgffBaseTransformation) -> NgffBaseTransformation:
    return t
    # TODO: to be reimplmeented or deleted after the new transformations refactoring
    # element_cs = get_default_coordinate_system(get_dims(e))
    # # the unit for the new element is the default one, called "unit". If the transformation has units, let's copy them
    # for axis in element_cs._axes:
    #     if axis.unit == "unit":
    #         assert t.input_coordinate_system is not None
    #         if t.input_coordinate_system.has_axis(axis.name):
    #             axis.unit = t.input_coordinate_system.get_axis(axis.name).unit
    # adjusted = _adjust_transformation_between_mismatching_coordinate_systems(t, element_cs)
    # return adjusted


@singledispatch
def set_transform(e: SpatialElement, t: NgffBaseTransformation) -> SpatialElement:
    raise TypeError(f"Unsupported type: {type(e)}")


@set_transform.register(SpatialImage)
def _(e: SpatialImage, t: NgffBaseTransformation) -> SpatialImage:
    new_t = _adjust_transformation_axes(e, t)
    e.attrs[TRANSFORM_KEY] = new_t
    return e


@set_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, t: NgffBaseTransformation) -> MultiscaleSpatialImage:
    new_t = _adjust_transformation_axes(e, t)
    e.attrs[TRANSFORM_KEY] = new_t
    return e


@set_transform.register(GeoDataFrame)
def _(e: GeoDataFrame, t: NgffBaseTransformation) -> GeoDataFrame:
    new_t = _adjust_transformation_axes(e, t)
    e.attrs[TRANSFORM_KEY] = new_t
    return e


@set_transform.register(AnnData)
def _(e: AnnData, t: NgffBaseTransformation) -> AnnData:
    new_t = _adjust_transformation_axes(e, t)
    e.uns[TRANSFORM_KEY] = new_t
    return e


@set_transform.register(pa.Table)
def _(e: pa.Table, t: NgffBaseTransformation) -> pa.Table:
    # in theory this doesn't really copy the data in the table but is referncing to them
    new_t = _adjust_transformation_axes(e, t)
    new_e = e.replace_schema_metadata({TRANSFORM_KEY: json.dumps(new_t.to_dict()).encode("utf-8")})
    return new_e


# unit is a default placeholder value. This is not suported by NGFF so the user should replace it before saving
# TODO: when saving, give a warning if the user does not replace it
x_axis = NgffAxis(name=X, type="space", unit="unit")
y_axis = NgffAxis(name=Y, type="space", unit="unit")
z_axis = NgffAxis(name=Z, type="space", unit="unit")
c_axis = NgffAxis(name=C, type="channel")
x_cs = NgffCoordinateSystem(name="x", axes=[x_axis])
y_cs = NgffCoordinateSystem(name="y", axes=[y_axis])
z_cs = NgffCoordinateSystem(name="z", axes=[z_axis])
c_cs = NgffCoordinateSystem(name="c", axes=[c_axis])
xy_cs = NgffCoordinateSystem(name="xy", axes=[x_axis, y_axis])
xyz_cs = NgffCoordinateSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
yx_cs = NgffCoordinateSystem(name="yx", axes=[y_axis, x_axis])
zyx_cs = NgffCoordinateSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
cyx_cs = NgffCoordinateSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
czyx_cs = NgffCoordinateSystem(name="czyx", axes=[c_axis, z_axis, y_axis, x_axis])

_DEFAULT_COORDINATE_SYSTEM = {
    (X,): x_cs,
    (Y,): y_cs,
    (Z,): z_cs,
    (C,): c_cs,
    (X, Y): xy_cs,
    (X, Y, Z): xyz_cs,
    (Y, X): yx_cs,
    (Z, Y, X): zyx_cs,
    (C, Y, X): cyx_cs,
    (C, Z, Y, X): czyx_cs,
}

get_default_coordinate_system = lambda dims: copy.deepcopy(_DEFAULT_COORDINATE_SYSTEM[tuple(dims)])


@singledispatch
def get_dims(e: SpatialElement) -> tuple[str, ...]:
    """
    Get the dimensions of a spatial element

    Parameters
    ----------
    e
        Spatial element

    Returns
    -------
    dims
        Dimensions of the spatial element (e.g. ("z", "y", "x"))
    """
    raise TypeError(f"Unsupported type: {type(e)}")


@get_dims.register(SpatialImage)
def _(e: SpatialImage) -> tuple[str, ...]:
    dims = e.dims
    return dims  # type: ignore


@get_dims.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> tuple[str, ...]:
    variables = list(e[list(e.keys())[0]].variables)
    return e[list(e.keys())[0]][variables[0]].dims  # type: ignore


@get_dims.register(GeoDataFrame)
def _(e: GeoDataFrame) -> tuple[str, ...]:
    dims = (X, Y, Z)
    n = e.geometry.iloc[0]._ndim
    return dims[:n]


@get_dims.register(AnnData)
def _(e: AnnData) -> tuple[str, ...]:
    dims = (X, Y, Z)
    n = e.obsm["spatial"].shape[1]
    return dims[:n]


@get_dims.register(pa.Table)
def _(e: pa.Table) -> tuple[str, ...]:
    valid_dims = (X, Y, Z)
    dims = [c for c in valid_dims if c in e.column_names]
    return tuple(dims)
