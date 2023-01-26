import copy
import json
from functools import singledispatch
from typing import Optional, Union

import numpy as np
import pyarrow as pa
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis, NgffCoordinateSystem
from spatialdata._core.transformations import BaseTransformation
from spatialdata._types import ArrayLike

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, AnnData, pa.Table]

__all__ = [
    "SpatialElement",
    "TRANSFORM_KEY",
    "_get_transform",
    "_set_transform",
    "get_default_coordinate_system",
    "get_dims",
    "C",
    "Z",
    "Y",
    "X",
]


TRANSFORM_KEY = "transform"
C, Z, Y, X = "c", "z", "y", "x"
ValidAxis_t = str
# ValidAxis_t = Literal["c", "x", "y", "z"]


def validate_axis_name(axis: ValidAxis_t) -> None:
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def _get_transform_xarray(e: DataArray) -> Optional[BaseTransformation]:
    if TRANSFORM_KEY in e.attrs:
        t = e.attrs.get(TRANSFORM_KEY)
        # this double return is to make mypy happy
        if t is not None:
            assert isinstance(t, BaseTransformation)
            return t
        else:
            return t
    else:
        return None


@singledispatch
def _get_transform(e: SpatialElement) -> Optional[BaseTransformation]:
    raise TypeError(f"Unsupported type: {type(e)}")


@_get_transform.register(SpatialImage)
def _(e: SpatialImage) -> Optional[BaseTransformation]:
    return _get_transform_xarray(e)


@_get_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> Optional[BaseTransformation]:
    if TRANSFORM_KEY in e.attrs:
        raise ValueError(
            "A multiscale image must not contain a transformation in the outer level; the transformations need to be "
            "stored in the inner levels."
        )
    d = dict(e["scale0"])
    assert len(d) == 1
    xdata = d.values().__iter__().__next__()
    t = _get_transform_xarray(xdata)
    if t is not None:
        assert isinstance(t, BaseTransformation)
        return t
    else:
        return t


@_get_transform.register(GeoDataFrame)
def _(e: GeoDataFrame) -> Optional[BaseTransformation]:
    if TRANSFORM_KEY in e.attrs:
        t = e.attrs.get(TRANSFORM_KEY)
        if t is not None:
            assert isinstance(t, BaseTransformation)
            return t
        else:
            return t
    else:
        return None


@_get_transform.register(AnnData)
def _(e: AnnData) -> Optional[BaseTransformation]:
    if TRANSFORM_KEY in e.uns:
        t = e.uns[TRANSFORM_KEY]
        if t is not None:
            assert isinstance(t, BaseTransformation)
            return t
        else:
            return t
    else:
        return None


# we need the return type because pa.Table is immutable
@_get_transform.register(pa.Table)
def _(e: pa.Table) -> Optional[BaseTransformation]:
    raise NotImplementedError("waiting for the new points implementation")
    t_bytes = e.schema.metadata[TRANSFORM_KEY.encode("utf-8")]
    t = BaseTransformation.from_dict(json.loads(t_bytes.decode("utf-8")))
    if t is not None:
        assert isinstance(t, BaseTransformation)
        return t
    else:
        return t


def _set_transform_xarray(e: DataArray, t: BaseTransformation) -> None:
    e.attrs[TRANSFORM_KEY] = t


@singledispatch
def _set_transform(e: SpatialElement, t: BaseTransformation) -> None:
    """
    Set the transformation of a spatial element *only in memory*.
    Parameters
    ----------
    e
        spatial element
    t
        transformation

    Notes
    -----
    This function only replaces the transformation in memory and is meant of internal use only. The function
    SpatialData.set_transform() should be used instead, since it will update the transformation in memory and on disk
    (when the spatial element is backed).

    """
    raise TypeError(f"Unsupported type: {type(e)}")


@_set_transform.register(SpatialImage)
def _(e: SpatialImage, t: BaseTransformation) -> None:
    _set_transform_xarray(e, t)


@_set_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, t: BaseTransformation) -> None:
    # set the transformation at the highest level and concatenate with the appropriate scale at each level
    dims = get_dims(e)
    from spatialdata._core.transformations import Scale, Sequence

    i = 0
    old_shape: Optional[ArrayLike] = None
    for scale, node in dict(e).items():
        # this is to be sure that the pyramid levels are listed here in the correct order
        assert scale == f"scale{i}"
        assert len(dict(node)) == 1
        xdata = list(node.values())[0]
        new_shape = np.array(xdata.shape)
        if i > 0:
            assert old_shape is not None
            scale_factors = old_shape / new_shape
            filtered_scale_factors = [scale_factors[i] for i, ax in enumerate(dims) if ax != "c"]
            filtered_axes = [ax for ax in dims if ax != "c"]
            scale = Scale(scale=filtered_scale_factors, axes=tuple(filtered_axes))
            assert t is not None
            sequence = Sequence([scale, t])
            _set_transform_xarray(xdata, sequence)
        else:
            _set_transform_xarray(xdata, t)
            old_shape = new_shape
        i += 1


@_set_transform.register(GeoDataFrame)
def _(e: GeoDataFrame, t: BaseTransformation) -> None:
    e.attrs[TRANSFORM_KEY] = t


@_set_transform.register(AnnData)
def _(e: AnnData, t: BaseTransformation) -> None:
    e.uns[TRANSFORM_KEY] = t


@_set_transform.register(pa.Table)
def _(e: pa.Table, t: BaseTransformation) -> None:
    # in theory this doesn't really copy the data in the table but is referncing to them
    raise NotImplementedError("waiting for the new points implementation")
    # new_e = e.replace_schema_metadata({TRANSFORM_KEY: json.dumps(t.to_dict()).encode("utf-8")})
    # return new_e


# unit is a default placeholder value. This is not suported by NGFF so the user should replace it before saving
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
    # luca: I prefer this first method
    d = dict(e["scale0"])
    assert len(d) == 1
    dims0 = d.values().__iter__().__next__().dims
    assert isinstance(dims0, tuple)
    # still, let's do a runtime check against the other method
    variables = list(e[list(e.keys())[0]].variables)
    dims1 = e[list(e.keys())[0]][variables[0]].dims
    assert dims0 == dims1
    return dims0


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
