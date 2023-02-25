from __future__ import annotations

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
from spatialdata._core.transformations import BaseTransformation, Sequence
from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata._core.transformations import Scale

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
    """Check if the object is an SpatialElement
    
    Parameters
    ----------
    e : Any
        The input object
    
    Returns
    -------
    bool
        Whether the object is an SpatialElement (i.e in Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame])
    """
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
    """Check if the axis name is valid
    
    Parameters
    ----------
    axis : ValidAxis_t
        The axis name
    
    Raises
    ------
    TypeError
        If not in ["c", "x", "y", "z"]
    """
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
    """Check if the axes' names are valid
    
    Parameters
    ----------
    axis : ValidAxis_t
        List of the axes' names
    
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
    """Get the spatial axes of interest
    
    Parameters
    ----------
    axes : tuple[ValidAxis_t, ...]
        Should be a subset of ['x', 'y', 'z']
    
    Returns
    -------
    tuple[ValidAxis_t, ...]
        The spatial axes
    """
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
            if not np.isfinite(filtered_scale_factors).all():
                raise ValueError("Scale factors must be finite.")
            scale_transformation = Scale(scale=filtered_scale_factors, axes=tuple(filtered_axes))
            assert transformations is not None
            new_transformations = {}
            for k, v in transformations.items():
                sequence: BaseTransformation = Sequence([scale_transformation, v])
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
    """Get the default coordinate system
    
    Parameters
    ----------
    dims : tuple[str, ...]
        The dimension names to get the corresponding axes of the defeault coordinate system for. 
        Names should be in ['x', 'y', 'z', 'c'].

    """
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


def _validate_dims(dims: tuple[str, ...]) -> None:
    for c in dims:
        if c not in (X, Y, Z, C):
            raise ValueError(f"Invalid dimension: {c}")
    if dims not in [(X,), (Y,), (Z,), (C,), (X, Y), (X, Y, Z), (Y, X), (Z, Y, X), (C, Y, X), (C, Z, Y, X)]:
        raise ValueError(f"Invalid dimensions: {dims}")


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
    # dims_sizes = tuple(list(e.sizes.keys()))
    # # we check that the following values are the same otherwise we could incur in subtle bugs downstreams
    # if dims != dims_sizes:
    #     raise ValueError(f"SpatialImage has inconsistent dimensions: {dims}, {dims_sizes}")
    _validate_dims(dims)
    return dims  # type: ignore


@get_dims.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> tuple[str, ...]:
    if "scale0" in e:
        # dims_coordinates = tuple(i for i in e["scale0"].dims.keys())

        assert len(e["scale0"].values()) == 1
        xdata = e["scale0"].values().__iter__().__next__()
        dims_data = xdata.dims
        assert isinstance(dims_data, tuple)

        # dims_sizes = tuple(list(xdata.sizes.keys()))

        # # we check that all the following values are the same otherwise we could incur in subtle bugs downstreams
        # if dims_coordinates != dims_data or dims_coordinates != dims_sizes:
        #     raise ValueError(
        #         f"MultiscaleSpatialImage has inconsistent dimensions: {dims_coordinates}, {dims_data}, {dims_sizes}"
        #     )
        _validate_dims(dims_data)
        return dims_data
    else:
        raise ValueError("MultiscaleSpatialImage does not contain the scale0 key")
        # return tuple(i for i in e.dims.keys())


@get_dims.register(GeoDataFrame)
def _(e: GeoDataFrame) -> tuple[str, ...]:
    all_dims = (X, Y, Z)
    n = e.geometry.iloc[0]._ndim
    dims = all_dims[:n]
    _validate_dims(dims)
    return dims


@get_dims.register(DaskDataFrame)
def _(e: AnnData) -> tuple[str, ...]:
    valid_dims = (X, Y, Z)
    dims = tuple([c for c in valid_dims if c in e.columns])
    _validate_dims(dims)
    return dims


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
        d: np.arange(data.sizes[d], dtype=np.float_) + 0.5 for d in data.sizes.keys() if d in ["x", "y", "z"]
    }
    return data.assign_coords(coords)


def _get_scale(transforms: dict[str, Any]) -> Scale:
    from spatialdata._core.transformations import Scale
    all_scale_vectors = []
    all_scale_axes = []
    for transformation in transforms.values():
        assert isinstance(transformation, Sequence)
        # the first transformation is the scale
        t = transformation.transformations[0]
        if hasattr(t, "scale"):
            if TYPE_CHECKING:
                assert isinstance(t.scale, np.ndarray)
            all_scale_vectors.append(tuple(t.scale.tolist()))
            assert isinstance(t, Scale)
            all_scale_axes.append(tuple(t.axes))
        else:
            raise ValueError(f"Unsupported transformation: {t}")
    # all the scales should be the same since they all refer to the mapping of the level of the multiscale to the
    # base level, with respect to the intrinstic coordinate system
    assert len(set(all_scale_vectors)) == 1
    assert len(set(all_scale_axes)) == 1
    scalef = np.array(all_scale_vectors[0])
    if not np.isfinite(scalef).all():
        raise ValueError(f"Invalid scale factor: {scalef}")
    scale_axes = all_scale_axes[0]
    scale = Scale(scalef, axes=scale_axes)
    return scale


@compute_coordinates.register(MultiscaleSpatialImage)
def _(data: MultiscaleSpatialImage) -> MultiscaleSpatialImage:
    def _compute_coords(n0: int, scale_f: float, n: int) -> ArrayLike:
        scaled_max = n0 / scale_f
        if n > 1:
            offset = scaled_max / (2.0 * (n - 1))
        else:
            offset = 0
        return np.linspace(0, scaled_max, n) + offset

    spatial_coords = [ax for ax in get_dims(data) if ax in ["x", "y", "z"]]
    img_name = list(data["scale0"].data_vars.keys())[0]
    out = {}

    for name, dt in data.items():
        if name == "scale0":
            coords: dict[str, ArrayLike] = {
                d: np.arange(data[name].sizes[d], dtype=np.float_) + 0.5 for d in spatial_coords
            }
            out[name] = dt[img_name].assign_coords(coords)
        else:
            scale = _get_scale(dt[img_name].attrs["transform"])
            scalef = scale.scale
            assert len(spatial_coords) == len(scalef), "Mismatch between coordinates and scales."  # type: ignore[arg-type]
            new_coords = {}
            for ax, s in zip(spatial_coords, scalef):
                new_coords[ax] = _compute_coords(
                    n0=data["scale0"].sizes[ax],
                    scale_f=s,
                    n=data[name].sizes[ax],
                )
            out[name] = dt[img_name].assign_coords(new_coords)
    msi = MultiscaleSpatialImage.from_dict(d=out)
    # this is to trigger the validation of the dims
    _ = get_dims(msi)
    return msi


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
