from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata.models import SpatialElement
    from spatialdata.models._utils import MappingToCoordinateSystem_t
    from spatialdata.transformations.transformations import BaseTransformation, Scale


def _get_transformations_from_dict_container(dict_container: Any) -> Optional[MappingToCoordinateSystem_t]:
    from spatialdata.models._utils import TRANSFORM_KEY

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


def _set_transformations_to_dict_container(dict_container: Any, transformations: MappingToCoordinateSystem_t) -> None:
    from spatialdata.models._utils import TRANSFORM_KEY

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


@_get_transformations.register(SpatialImage)
def _(e: SpatialImage) -> Optional[MappingToCoordinateSystem_t]:
    return _get_transformations_xarray(e)


@_get_transformations.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> Optional[MappingToCoordinateSystem_t]:
    from spatialdata.models._utils import TRANSFORM_KEY

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


@_set_transformations.register(SpatialImage)
def _(e: SpatialImage, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_xarray(e, transformations)


@_set_transformations.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, transformations: MappingToCoordinateSystem_t) -> None:
    from spatialdata.models import get_axes_names

    # set the transformation at the highest level and concatenate with the appropriate scale at each level
    dims = get_axes_names(e)
    from spatialdata.transformations.transformations import Scale, Sequence

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


def _get_scale(transforms: dict[str, Any]) -> Scale:
    from spatialdata.transformations.transformations import Scale, Sequence

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


@compute_coordinates.register(SpatialImage)
def _(data: SpatialImage) -> SpatialImage:
    coords: dict[str, ArrayLike] = {
        d: np.arange(data.sizes[d], dtype=np.float_) + 0.5 for d in data.sizes if d in ["x", "y", "z"]
    }
    return data.assign_coords(coords)


@compute_coordinates.register(MultiscaleSpatialImage)
def _(data: MultiscaleSpatialImage) -> MultiscaleSpatialImage:
    from spatialdata.models import get_axes_names

    spatial_coords = [ax for ax in get_axes_names(data) if ax in ["x", "y", "z"]]
    img_name = list(data["scale0"].data_vars.keys())[0]
    out = {}
    for name, dt in data.items():
        new_coords = {}
        for ax in spatial_coords:
            max_dim = data["scale0"].sizes[ax]
            n = dt.sizes[ax]
            offset = max_dim / n / 2
            coords = np.linspace(0, max_dim, n + 1)[:-1] + offset
            new_coords[ax] = coords
        out[name] = dt[img_name].assign_coords(new_coords)
    msi = MultiscaleSpatialImage.from_dict(d=out)
    # this is to trigger the validation of the dims
    _ = get_axes_names(msi)
    return msi
