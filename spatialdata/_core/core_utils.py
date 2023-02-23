import copy
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis, NgffCoordinateSystem
from spatialdata._core.transformations import BaseTransformation
from spatialdata._types import ArrayLike

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame]

__all__ = [
    "SpatialElement",
    "TRANSFORM_KEY",
    "_get_transformations",
    "_set_transformations",
    "get_default_coordinate_system",
    "get_dims",
    "C",
    "Z",
    "Y",
    "X",
]


TRANSFORM_KEY = "transform"
DEFAULT_COORDINATE_SYSTEM = "global"
C, Z, Y, X = "c", "z", "y", "x"
ValidAxis_t = str
# ValidAxis_t = Literal["c", "x", "y", "z"]
# maybe later we will want this, but let's keep it simple for now
# MappingToCoordinateSystem_t = dict[NgffCoordinateSystem, BaseTransformation]
MappingToCoordinateSystem_t = dict[str, BaseTransformation]

# added this code as part of a refactoring to catch errors earlier


# mypy says that we can't do isinstance(something, SpatialElement), even if the code works fine in my machine. Since the solution described here don't work: https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6, I am just using the function below
def has_type_spatial_element(e: Any) -> bool:
    return isinstance(e, (SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame))


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
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
    for ax in axes:
        validate_axis_name(ax)
    if len(axes) != len(set(axes)):
        raise ValueError("Axes must be unique.")


def get_spatial_axes(axes: tuple[ValidAxis_t, ...]) -> tuple[ValidAxis_t, ...]:
    validate_axes(axes)
    return tuple(ax for ax in axes if ax in [X, Y, Z])


def _get_transformations_from_dict_container(dict_container: Any) -> Optional[MappingToCoordinateSystem_t]:
    if TRANSFORM_KEY in dict_container:
        d = dict_container[TRANSFORM_KEY]
        return d  # type: ignore[no-any-return]
    else:
        return None


def _get_transformations_xarray(e: DataArray) -> Optional[MappingToCoordinateSystem_t]:
    return _get_transformations_from_dict_container(e.attrs)


@singledispatch
def _get_transformations(e: SpatialElement) -> Optional[MappingToCoordinateSystem_t]:
    raise TypeError(f"Unsupported type: {type(e)}")


@_get_transformations.register(SpatialImage)
def _(e: SpatialImage) -> Optional[MappingToCoordinateSystem_t]:
    return _get_transformations_xarray(e)


@_get_transformations.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> Optional[MappingToCoordinateSystem_t]:
    if TRANSFORM_KEY in e.attrs:
        raise ValueError(
            "A multiscale image must not contain a transformation in the outer level; the transformations need to be "
            "stored in the inner levels."
        )
    d = dict(e["scale0"])
    assert len(d) == 1
    xdata = d.values().__iter__().__next__()
    return _get_transformations_xarray(xdata)


@_get_transformations.register(GeoDataFrame)
@_get_transformations.register(DaskDataFrame)
def _(e: Union[GeoDataFrame, DaskDataFrame]) -> Optional[MappingToCoordinateSystem_t]:
    return _get_transformations_from_dict_container(e.attrs)


@_get_transformations.register(AnnData)
def _(e: AnnData) -> Optional[MappingToCoordinateSystem_t]:
    return _get_transformations_from_dict_container(e.uns)


def _set_transformations_to_dict_container(dict_container: Any, transformations: MappingToCoordinateSystem_t) -> None:
    if TRANSFORM_KEY not in dict_container:
        dict_container[TRANSFORM_KEY] = {}
    dict_container[TRANSFORM_KEY] = transformations


def _set_transformations_xarray(e: DataArray, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_dict_container(e.attrs, transformations)


@singledispatch
def _set_transformations(e: SpatialElement, transformations: MappingToCoordinateSystem_t) -> None:
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


@_set_transformations.register(SpatialImage)
def _(e: SpatialImage, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_xarray(e, transformations)


@_set_transformations.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, transformations: MappingToCoordinateSystem_t) -> None:
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
            assert transformations is not None
            new_transformations = {}
            for k, v in transformations.items():
                sequence: BaseTransformation = Sequence([scale, v])
                new_transformations[k] = sequence
            _set_transformations_xarray(xdata, new_transformations)
        else:
            _set_transformations_xarray(xdata, transformations)
            old_shape = new_shape
        i += 1


@_set_transformations.register(GeoDataFrame)
@_set_transformations.register(DaskDataFrame)
def _(e: Union[GeoDataFrame, GeoDataFrame], transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_dict_container(e.attrs, transformations)


@_set_transformations.register(AnnData)
def _(e: AnnData, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_dict_container(e.uns, transformations)


def _(e: DaskDataFrame, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_dict_container(e.attrs, transformations)


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

# get_default_coordinate_system = lambda dims: copy.deepcopy(_DEFAULT_COORDINATE_SYSTEM[tuple(dims)])


def get_default_coordinate_system(dims: tuple[str, ...]) -> NgffCoordinateSystem:
    axes = []
    for c in dims:
        if c == X:
            axes.append(copy.deepcopy(x_axis))
        elif c == Y:
            axes.append(copy.deepcopy(y_axis))
        elif c == Z:
            axes.append(copy.deepcopy(z_axis))
        elif c == C:
            axes.append(copy.deepcopy(c_axis))
        else:
            raise ValueError(f"Invalid dimension: {c}")
    return NgffCoordinateSystem(name="".join(dims), axes=axes)


@singledispatch
def get_dims(e: SpatialElement) -> tuple[str, ...]:
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


@get_dims.register(SpatialImage)
def _(e: SpatialImage) -> tuple[str, ...]:
    dims = e.dims
    return dims  # type: ignore


@get_dims.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> tuple[str, ...]:
    if "scale0" in e:
        return tuple(i for i in e["scale0"].dims.keys())
    else:
        raise ValueError("MultiscaleSpatialImage does not contain the scale0 key")
        # return tuple(i for i in e.dims.keys())


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


@get_dims.register(DaskDataFrame)
def _(e: AnnData) -> tuple[str, ...]:
    valid_dims = (X, Y, Z)
    dims = [c for c in valid_dims if c in e.columns]
    return tuple(dims)


@singledispatch
def compute_coordinates(
    data: Union[SpatialImage, MultiscaleSpatialImage]
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    """
    Computes and assign coordinates to a (Multiscale)SpatialImage.

    Parameters
    ----------
    data
        :class:`SpatialImage` or :class:`MultiscaleSpatialImage`.

    Returns
    -------
    :class:`SpatialImage` or :class:`MultiscaleSpatialImage` with coordinates assigned.
    """
    raise TypeError(f"Unsupported type: {type(data)}")


@compute_coordinates.register(SpatialImage)
def _(data: SpatialImage) -> SpatialImage:
    coords: dict[str, ArrayLike] = {
        d: np.arange(data.sizes[d], dtype=np.float_) for d in data.sizes.keys() if d in ["x", "y", "z"]
    }
    return data.assign_coords(coords)


@compute_coordinates.register(MultiscaleSpatialImage)
def _(data: MultiscaleSpatialImage) -> MultiscaleSpatialImage:
    def _get_scale(transforms: dict[str, Any]) -> Optional[ArrayLike]:
        for t in transforms["global"].transformations:
            if hasattr(t, "scale"):
                if TYPE_CHECKING:
                    assert isinstance(t.scale, np.ndarray)
                return t.scale

    def _compute_coords(max_: int, scale_f: Union[int, float]) -> ArrayLike:
        return (  # type: ignore[no-any-return]
            DataArray(np.linspace(0, max_, max_, endpoint=False, dtype=np.float_))
            .coarsen(dim_0=scale_f, boundary="trim", side="right")
            .mean()
            .values
        )

    max_scale0 = {d: s for d, s in data["scale0"].sizes.items() if d in ["x", "y", "z"]}
    img_name = list(data["scale0"].data_vars.keys())[0]
    out = {}

    for name, dt in data.items():
        max_scale = {d: s for d, s in data["scale0"].sizes.items() if d in ["x", "y", "z"]}
        if name == "scale0":
            coords: dict[str, ArrayLike] = {d: np.arange(max_scale[d], dtype=np.float_) for d in max_scale.keys()}
            out[name] = dt[img_name].assign_coords(coords)
        else:
            scalef = _get_scale(dt[img_name].attrs["transform"])
            assert len(max_scale.keys()) == len(scalef), "Mismatch between coordinates and scales."  # type: ignore[arg-type]
            out[name] = dt[img_name].assign_coords(
                {k: _compute_coords(max_scale0[k], round(s)) for k, s in zip(max_scale.keys(), scalef)}  # type: ignore[arg-type]
            )
    return MultiscaleSpatialImage.from_dict(d=out)


@singledispatch
def get_channels(data: Any) -> list[Any]:
    """Get channels from data.

    Parameters
    ----------
    data
        data to get channels from

    Returns
    -------
    List of channels
    """
    raise ValueError(f"Cannot get channels from {type(data)}")


@get_channels.register
def _(data: SpatialImage) -> list[Any]:
    return data.coords["c"].values.tolist()  # type: ignore[no-any-return]


@get_channels.register
def _(data: MultiscaleSpatialImage) -> list[Any]:
    name = list({list(data[i].data_vars.keys())[0] for i in data.keys()})[0]
    channels = {tuple(data[i][name].coords["c"].values) for i in data.keys()}
    if len(channels) > 1:
        raise ValueError("TODO")
    return list(next(iter(channels)))
