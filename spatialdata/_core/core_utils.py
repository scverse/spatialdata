import copy
from functools import singledispatch
from typing import Tuple, Union

from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.coordinate_system import Axis, CoordinateSystem
from spatialdata._core.transformations import BaseTransformation

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, AnnData]

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


@singledispatch
def get_transform(e: SpatialElement) -> BaseTransformation:
    raise TypeError(f"Unsupported type: {type(e)}")


@get_transform.register(SpatialImage)
def _(e: SpatialImage) -> BaseTransformation:
    t = e.attrs.get(TRANSFORM_KEY)
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> BaseTransformation:
    t = e.attrs.get(TRANSFORM_KEY)
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(GeoDataFrame)
def _(e: GeoDataFrame) -> BaseTransformation:
    t = e.attrs.get(TRANSFORM_KEY)
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(AnnData)
def _(e: AnnData) -> BaseTransformation:
    t = e.uns[TRANSFORM_KEY]
    assert isinstance(t, BaseTransformation)
    return t


@singledispatch
def set_transform(e: SpatialElement, t: BaseTransformation) -> None:
    raise TypeError(f"Unsupported type: {type(e)}")


@set_transform.register(SpatialImage)
def _(e: SpatialImage, t: BaseTransformation) -> None:
    e.attrs[TRANSFORM_KEY] = t


@set_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, t: BaseTransformation) -> None:
    e.attrs[TRANSFORM_KEY] = t


@set_transform.register(GeoDataFrame)
def _(e: GeoDataFrame, t: BaseTransformation) -> None:
    e.attrs[TRANSFORM_KEY] = t


@set_transform.register(AnnData)
def _(e: AnnData, t: BaseTransformation) -> None:
    e.uns[TRANSFORM_KEY] = t


# unit is a default placeholder value. This is not suported by NGFF so the user should replace it before saving
# TODO: when saving, give a warning if the user does not replace it
x_axis = Axis(name=X, type="space", unit="unit")
y_axis = Axis(name=Y, type="space", unit="unit")
z_axis = Axis(name=Z, type="space", unit="unit")
c_axis = Axis(name=C, type="channel")
xy_cs = CoordinateSystem(name="xy", axes=[x_axis, y_axis])
xyz_cs = CoordinateSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
yx_cs = CoordinateSystem(name="yx", axes=[y_axis, x_axis])
zyx_cs = CoordinateSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
cyx_cs = CoordinateSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
czyx_cs = CoordinateSystem(name="czyx", axes=[c_axis, z_axis, y_axis, x_axis])

_DEFAULT_COORDINATE_SYSTEM = {
    (X, Y): xy_cs,
    (X, Y, Z): xyz_cs,
    (Y, X): yx_cs,
    (Z, Y, X): zyx_cs,
    (C, Y, X): cyx_cs,
    (C, Z, Y, X): czyx_cs,
}

get_default_coordinate_system = lambda dims: copy.deepcopy(_DEFAULT_COORDINATE_SYSTEM[tuple(dims)])


@singledispatch
def get_dims(e: SpatialElement) -> Tuple[str, ...]:
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
def _(e: SpatialImage) -> Tuple[str, ...]:
    dims = e.dims
    return dims  # type: ignore


@get_dims.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> Tuple[str, ...]:
    return e[list(e.keys())[0]].dims  # type: ignore


@get_dims.register(GeoDataFrame)
def _(e: GeoDataFrame) -> Tuple[str, ...]:
    dims = (X, Y, Z)
    n = e.geometry.iloc[0]._ndim
    return dims[:n]


@get_dims.register(AnnData)
def _(e: AnnData) -> Tuple[str, ...]:
    dims = (X, Y, Z)
    n = e.obsm["spatial"].shape[1]
    return dims[:n]
