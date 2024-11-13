from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, Dataset, DataTree

from spatialdata._logging import logger
from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData
    from spatialdata.models import SpatialElement
    from spatialdata.models._utils import MappingToCoordinateSystem_t
    from spatialdata.transformations.transformations import Affine, BaseTransformation, Scale


def _get_transformations_from_dict_container(dict_container: Any) -> MappingToCoordinateSystem_t | None:
    from spatialdata.models._utils import TRANSFORM_KEY

    if TRANSFORM_KEY in dict_container:
        d = dict_container[TRANSFORM_KEY]
        return d  # type: ignore[no-any-return]
    else:
        return None


def _get_transformations_xarray(e: DataArray) -> MappingToCoordinateSystem_t | None:
    return _get_transformations_from_dict_container(e.attrs)


@singledispatch
def _get_transformations(e: SpatialElement) -> MappingToCoordinateSystem_t | None:
    raise TypeError(f"Unsupported type: {type(e)}")


def _set_transformations_to_dict_container(dict_container: Any, transformations: MappingToCoordinateSystem_t) -> None:
    from spatialdata.models._utils import TRANSFORM_KEY

    if TRANSFORM_KEY not in dict_container:
        dict_container[TRANSFORM_KEY] = {}
    # this modifies the dict in place without triggering a setter in the element class. Probably we want to stop using
    # _set_transformations_to_dict_container and use _set_transformations_to_element instead
    dict_container[TRANSFORM_KEY] = transformations


def _set_transformations_to_element(element: Any, transformations: MappingToCoordinateSystem_t) -> None:
    from spatialdata.models._utils import TRANSFORM_KEY

    attrs = element.attrs
    if TRANSFORM_KEY not in attrs:
        attrs[TRANSFORM_KEY] = {}
    attrs[TRANSFORM_KEY] = transformations
    # this calls an eventual setter in the element class; modifying the attrs directly would not trigger the setter
    element.attrs = attrs


@singledispatch
def _set_transformations(e: SpatialElement, transformations: MappingToCoordinateSystem_t) -> None:
    """
    Set the transformation of a SpatialElement *only in memory*.
    Parameters
    ----------
    e
        SpatialElement
    t
        transformation

    Notes
    -----
    This function only replaces the transformation in memory and is meant of internal use only. The function
    SpatialData.set_transform() should be used instead, since it will update the transformation in memory and on disk
    (when the SpatialElement is backed).

    """
    raise TypeError(f"Unsupported type: {type(e)}")


@_set_transformations.register(DataArray)
def _(e: DataArray, transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_dict_container(e.attrs, transformations)


@_set_transformations.register(DataTree)
def _(e: DataTree, transformations: MappingToCoordinateSystem_t) -> None:
    from spatialdata.models import get_axes_names

    # set the transformation at the highest level and concatenate with the appropriate scale at each level
    dims = get_axes_names(e)
    from spatialdata.transformations.transformations import Scale, Sequence

    old_shape: Optional[ArrayLike] = None
    for i, (scale, node) in enumerate(dict(e).items()):
        # this is to be sure that the pyramid levels are listed here in the correct order
        if scale != f"scale{i}":
            pass
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
            _set_transformations(xdata, new_transformations)
        else:
            _set_transformations(xdata, transformations)
            old_shape = new_shape


@_set_transformations.register(GeoDataFrame)
@_set_transformations.register(DaskDataFrame)
def _(e: Union[GeoDataFrame, GeoDataFrame], transformations: MappingToCoordinateSystem_t) -> None:
    _set_transformations_to_element(e, transformations)
    # _set_transformations_to_dict_container(e.attrs, transformations)


@_get_transformations.register(DataArray)
def _(e: DataArray) -> MappingToCoordinateSystem_t | None:
    return _get_transformations_xarray(e)


@_get_transformations.register(DataTree)
def _(e: DataTree) -> MappingToCoordinateSystem_t | None:
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
def _(e: Union[GeoDataFrame, DaskDataFrame]) -> MappingToCoordinateSystem_t | None:
    return _get_transformations_from_dict_container(e.attrs)


@singledispatch
def compute_coordinates(data: DataArray | DataTree) -> DataArray | DataTree:
    """
    Computes and assign coordinates to a spatialdata supported DataArray or DataTree.

    Parameters
    ----------
    data
        :class:`DataArray` or :class:`DataTree`.

    Returns
    -------
    :class:`DataArray` or :class:`DataTree` with coordinates assigned.
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


@compute_coordinates.register(DataArray)
def _(data: DataArray) -> DataArray:
    coords: dict[str, ArrayLike] = {
        d: np.arange(data.sizes[d], dtype=np.float64) + 0.5 for d in data.sizes if d in ["x", "y", "z"]
    }
    return data.assign_coords(coords)


@compute_coordinates.register(DataTree)
def _(data: DataTree) -> DataTree:
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

        # Xarray now only accepts Dataset as dictionary values for DataTree.from_dict.
        out[name] = Dataset({img_name: dt[img_name].assign_coords(new_coords)})
    datatree = DataTree.from_dict(out)
    # this is to trigger the validation of the dims
    _ = get_axes_names(datatree)
    return datatree


def scale_radii(radii: ArrayLike, affine: Affine, axes: tuple[str, ...]) -> ArrayLike:
    """
    Scale the radii (of a list of points) by the average of the modules of the eigenvalues of an affine transformation.

    Parameters
    ----------
    radii
        radii of the points
    affine
        affine transformation
    axes
        axes of the points, e.g. ("x", "y") or ("x", "y", "z")

    Returns
    -------
    scaled radii
    """
    matrix = affine.to_affine_matrix(input_axes=(axes), output_axes=(axes))
    eigenvalues = np.linalg.eigvals(matrix[:-1, :-1])
    modules = np.absolute(eigenvalues)
    if not np.allclose(modules, modules[0]):
        scale_factor = np.mean(modules)
        logger.warning(
            "The vector part of the transformation matrix is not isotropic, the radius will be scaled by the average "
            f"of the modules of eigenvalues of the affine transformation matrix.\nmatrix={matrix}\n"
            f"eigenvalues={eigenvalues}\nscale_factor={scale_factor}"
        )
    else:
        scale_factor = modules[0]
    new_radii = radii * scale_factor
    assert isinstance(new_radii, np.ndarray)
    return new_radii


def convert_transformations_to_affine(sdata: SpatialData, coordinate_system: str) -> None:
    """
    Convert all transformations to the given coordinate system to affine transformations.

    Parameters
    ----------
    coordinate_system
        The coordinate system to convert to.

    Notes
    -----
    The new transformations are modified only in-memory. If you want to save the changes to disk please call
    `SpatialData.write_transformations()`.
    """
    from spatialdata.transformations.operations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import Affine, _get_affine_for_element

    for _, _, element in sdata.gen_spatial_elements():
        transformations = get_transformation(element, get_all=True)
        assert isinstance(transformations, dict)
        if coordinate_system in transformations:
            t = transformations[coordinate_system]
            if not isinstance(t, Affine):
                affine = _get_affine_for_element(element, t)
                set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
