from __future__ import annotations

import itertools
import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import dask.array as da
import dask_image.ndinterp
import numpy as np
from dask.array.core import Array as DaskArray
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import Point
from xarray import DataArray, Dataset, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import SpatialElement, get_axes_names, get_model
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM, get_channels
from spatialdata.transformations._utils import _get_scale, compute_coordinates, scale_radii

if TYPE_CHECKING:
    from spatialdata.transformations.transformations import (
        BaseTransformation,
        Translation,
    )

DEBUG_WITH_PLOTS = False
ERROR_MSG_AFTER_0_0_15 = """\
Starting after v0.0.15, `transform()` requires to specify `to_coordinate_system` instead of `transformation`, to avoid
ambiguity when multiple transformations are available for an element. If the transformation is not present in the
element, please add it with `set_transformation()`.
"""


def _transform_raster(
    data: DaskArray, axes: tuple[str, ...], transformation: BaseTransformation, **kwargs: Any
) -> tuple[DaskArray, Translation]:
    # dims = {ch: axes.index(ch) for ch in axes}
    from spatialdata.transformations.transformations import Sequence, Translation

    n_spatial_dims = transformation._get_n_spatial_dims(axes)
    binary: ArrayLike = np.array(list(itertools.product([0, 1], repeat=n_spatial_dims)))
    spatial_shape = data.shape[len(data.shape) - n_spatial_dims :]
    binary *= np.array(spatial_shape)
    c_channel = [np.zeros(len(binary)).reshape((-1, 1))] if "c" in axes else []
    v: ArrayLike = np.hstack(c_channel + [binary, np.ones(len(binary)).reshape((-1, 1))])
    matrix = transformation.to_affine_matrix(input_axes=axes, output_axes=axes)
    inverse_matrix = transformation.inverse().to_affine_matrix(input_axes=axes, output_axes=axes)
    new_v: ArrayLike = (matrix @ v.T).T
    c_shape: tuple[int, ...]
    c_shape = (data.shape[0],) if "c" in axes else ()
    new_spatial_shape = tuple(
        int(np.max(new_v[:, i]) - np.min(new_v[:, i])) for i in range(len(c_shape), n_spatial_dims + len(c_shape))  # type: ignore[operator]
    )
    output_shape = c_shape + new_spatial_shape
    translation_vector = np.min(new_v[:, :-1], axis=0)
    translation = Translation(translation_vector, axes=axes)
    inverse_matrix_adjusted = Sequence(
        [
            translation,
            transformation.inverse(),
        ]
    ).to_affine_matrix(input_axes=axes, output_axes=axes)

    # fix chunk shape, it should be possible for the user to specify them,
    # and by default we could reuse the chunk shape of the input
    transformed_dask = dask_image.ndinterp.affine_transform(
        data,
        matrix=inverse_matrix_adjusted,
        output_shape=output_shape,
        **kwargs,
        # , output_chunks=output_chunks
    )
    assert isinstance(transformed_dask, DaskArray)

    if DEBUG_WITH_PLOTS:
        if n_spatial_dims == 2:
            import matplotlib.pyplot as plt

            plt.figure()
            im = data
            new_v_inverse = (inverse_matrix @ v.T).T
            # min_x_inverse = np.min(new_v_inverse[:, 2])
            # min_y_inverse = np.min(new_v_inverse[:, 1])

            if "c" in axes:
                plt.imshow(da.moveaxis(transformed_dask, 0, 2), origin="lower", alpha=0.5)  # type: ignore[attr-defined]
                plt.imshow(da.moveaxis(im, 0, 2), origin="lower", alpha=0.5)  # type: ignore[attr-defined]
            else:
                plt.imshow(transformed_dask, origin="lower", alpha=0.5)
                plt.imshow(im, origin="lower", alpha=0.5)
            start_index = 1 if "c" in axes else 0
            plt.scatter(v[:, start_index:-1][:, 1] - 0.5, v[:, start_index:-1][:, 0] - 0.5, c="r")
            plt.scatter(new_v[:, start_index:-1][:, 1] - 0.5, new_v[:, start_index:-1][:, 0] - 0.5, c="g")
            plt.scatter(
                new_v_inverse[:, start_index:-1][:, 1] - 0.5, new_v_inverse[:, start_index:-1][:, 0] - 0.5, c="k"
            )
            plt.show()
        else:
            assert n_spatial_dims == 3
            # raise NotImplementedError()
    return transformed_dask, translation


def _set_transformation_for_transformed_elements(
    element: SpatialElement,
    old_transformations: dict[str, BaseTransformation],
    transformation: BaseTransformation,
    raster_translation: Translation | None,
    maintain_positioning: bool,
    to_coordinate_system: str | None = None,
) -> None:
    """Transformed elements don't have transformations to the new coordinate system, so we need to set them.

     By default, we add a identity transformation to the new coordinate system of the transformed element (if
     maintain positioning is `False`), or we add all the transformations of the original element to the new
     coordinate system of the transformed element, prepending the inverse of the transformation that was used to
     transform the data (if maintain positioning is `True`).

    In addition to this, when the transformed data is of raster type, we need to add a translation that takes into
    account for several factors: new origin, padding (at the corners) due to rotation, and the fact that the (0, 0)
    coordinate is the center of the (0, 0) pixel (this leads to small offsets that need to be compensated when scaling
    or rotating the data).
    All these factors are already contained in the `raster_translation` parameter, so we just need to prepend it to the
    data.

    Parameters
    ----------
    element
        The SpatialElement to which the transformation should be prepended.
    old_transformations
        The transformations that were present in the element before the data was transformed.
    transformation
        The transformation that was used to transform the data.
    raster_translation
        If the data is non-raster this parameter must be None. If the data is raster, this translation is the one
        that would make the old data and the transformed data aligned. Note that if the transformation that was used
        to transform the data contained a rotation-like component, then the transformed data contains some padding on
        the corners. This parameter takes into account also for that (i.e. if prepended to the transformation,
        it will make the transformed data aligned with the old data).
    maintain_positioning
        If True, the inverse of the transformation is prepended to the existing transformations of the element (after
        the eventual raster_translation). This is useful when the user wants to transform the actual data,
        but maintain the positioning of the element in the various coordinate systems.
    to_coordinate_system
        The coordinate system to which the data is to be transformed. This value must be None if maintain_positioning
        is True.
    """  # noqa: D401
    from spatialdata.transformations import (
        BaseTransformation,
        get_transformation,
        remove_transformation,
        set_transformation,
    )
    from spatialdata.transformations.transformations import Identity, Sequence

    if maintain_positioning:
        assert to_coordinate_system is None

    to_prepend: BaseTransformation | None
    if isinstance(element, DataArray | DataTree):
        if maintain_positioning:
            assert raster_translation is not None
            to_prepend = Sequence([raster_translation, transformation.inverse()])
        else:
            to_prepend = raster_translation
    elif isinstance(element, GeoDataFrame | DaskDataFrame):
        assert raster_translation is None
        to_prepend = transformation.inverse() if maintain_positioning else Identity()
    else:
        raise TypeError(f"Unsupported type {type(element)}")
    assert isinstance(to_prepend, BaseTransformation)

    d = get_transformation(element, get_all=True)
    assert isinstance(d, dict)
    if DEFAULT_COORDINATE_SYSTEM not in d:
        raise RuntimeError(f"Coordinate system {DEFAULT_COORDINATE_SYSTEM} not found in element")
    assert isinstance(d, dict)
    assert len(d) == 1
    assert isinstance(d[DEFAULT_COORDINATE_SYSTEM], Identity)
    remove_transformation(element, remove_all=True)

    if maintain_positioning:
        for cs, t in old_transformations.items():
            new_t: BaseTransformation
            new_t = Sequence([to_prepend, t])
            set_transformation(element, new_t, to_coordinate_system=cs)
    else:
        set_transformation(element, to_prepend, to_coordinate_system=to_coordinate_system)


def _validate_target_coordinate_systems(
    data: SpatialElement,
    transformation: BaseTransformation | None,
    maintain_positioning: bool,
    to_coordinate_system: str | None,
) -> tuple[BaseTransformation, str | None]:
    from spatialdata.transformations import BaseTransformation, get_transformation

    if not maintain_positioning:
        d = get_transformation(data, get_all=True)
        assert isinstance(d, dict)
        if transformation is None and to_coordinate_system is not None:
            assert to_coordinate_system in d, f"Coordinate system {to_coordinate_system} not found in element"
            return d[to_coordinate_system], to_coordinate_system
        if transformation is not None and to_coordinate_system is None and len(d) == 1:
            k = list(d.keys())[0]
            if transformation == d[k]:
                warnings.warn(ERROR_MSG_AFTER_0_0_15, stacklevel=2)
                return transformation, k
        raise RuntimeError(ERROR_MSG_AFTER_0_0_15)
    s = "When maintain_positioning is True, only one of transformation and to_coordinate_system can be None"
    assert bool(transformation is None) != bool(to_coordinate_system is None), s
    if transformation is not None:
        return transformation, to_coordinate_system
    t = get_transformation(data, to_coordinate_system=to_coordinate_system)
    assert isinstance(t, BaseTransformation)
    return t, None


@singledispatch
def transform(
    data: Any,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> Any:
    """
    Transform a SpatialElement using the transformation to a coordinate system, and returns the transformed element.

    Parameters
    ----------
    data
        SpatialElement to transform.
    transformation
        The transformation to apply to the element. This parameter can be used only when `maintain_positioning=True`,
        otherwise `to_coordinate_system` must be used.
    maintain_positioning
        The default and recommended behavior is to leave this parameter to False.

        - If True, in the transformed element, each transformation that was present in the original element will be
            prepended with the inverse of the transformation used to transform the data (i.e. the current
            transformation for which .transform() is called). In this way the data is transformed but the
            positioning (for each coordinate system) is maintained. A use case is changing the
            orientation/scale/etc. of the data but keeping the alignment of the data within each coordinate system.

        - If False, the data is transformed and the positioning changes; only the coordinate system in which the
            data is transformed to is kept. For raster data, the translation part of the transformation is assigned
            to the element (see Notes below for more details). Furthermore, for raster data, the returned object
            will have a translation to take into account for the pixel (0, 0) position. Also, rotated raster data
            will be padded in the corners with a black color, such padding will be reflected into the rotation.
            Please see notes for more details of how this parameter interact with xarray.DataArray for raster data.

    to_coordinate_system
        The coordinate system to which the data should be transformed. The coordinate system must be present in the
        element.

    Returns
    -------
    SpatialElement: Transformed SpatialElement.

    Notes
    -----
    An affine transformation contains a linear transformation and a translation. For raster types,
    only the linear transformation is applied to the data (e.g. the data is rotated or resized), but not the
    translation part.
    This means that calling Translation(...).transform(raster_element) will have the same effect as pre-pending the
    translation to each transformation of the raster element (if maintain_positioning=True), or assigning this
    translation to the element in the new coordinate system (if maintain_positioning=False). Analougous considerations
    apply to the black corner padding due to the rotation part of the transformation.
    We are considering to change this behavior by letting translations modify the coordinates stored with
    xarray.DataArray; this is tracked here: https://github.com/scverse/spatialdata/issues/308
    """
    raise RuntimeError(f"Cannot transform {type(data)}")


@transform.register(SpatialData)
def _(
    data: SpatialData,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> SpatialData:
    if not maintain_positioning:
        if transformation is None and to_coordinate_system is not None:
            return data.transform_to_coordinate_system(target_coordinate_system=to_coordinate_system)
        raise RuntimeError(ERROR_MSG_AFTER_0_0_15)
    assert bool(transformation is None) != bool(
        to_coordinate_system is None
    ), "When maintain_positioning is True, only one of transformation and to_coordinate_system can be None"
    new_elements: dict[str, dict[str, Any]] = {}
    for element_type in ["images", "labels", "points", "shapes"]:
        d = getattr(data, element_type)
        if len(d) > 0:
            new_elements[element_type] = {}
        for k, v in d.items():
            new_elements[element_type][k] = transform(
                v, transformation, to_coordinate_system=to_coordinate_system, maintain_positioning=maintain_positioning
            )
    return SpatialData(**new_elements)


@transform.register(DataArray)
def _(
    data: DataArray,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> DataArray:
    transformation, to_coordinate_system = _validate_target_coordinate_systems(
        data, transformation, maintain_positioning, to_coordinate_system
    )
    schema = get_model(data)
    from spatialdata.transformations import get_transformation

    kwargs = {"prefilter": False, "order": 0}
    axes = get_axes_names(data)
    transformed_dask, raster_translation = _transform_raster(
        data=data.data, axes=axes, transformation=transformation, **kwargs
    )
    c_coords = data.indexes["c"].values if "c" in data.indexes else None
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes, c_coords=c_coords)  # type: ignore[call-arg,arg-type]
    assert isinstance(transformed_data, DataArray)
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _set_transformation_for_transformed_elements(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
        to_coordinate_system=to_coordinate_system,
    )
    transformed_data = compute_coordinates(transformed_data)
    schema().validate(transformed_data)
    return transformed_data


@transform.register(DataTree)
def _(
    data: DataTree,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> DataTree:
    transformation, to_coordinate_system = _validate_target_coordinate_systems(
        data, transformation, maintain_positioning, to_coordinate_system
    )
    schema = get_model(data)
    from spatialdata.models import (
        Image2DModel,
        Image3DModel,
        Labels2DModel,
        Labels3DModel,
    )
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import Identity, Sequence

    # labels need to be preserved after the resizing of the image
    if schema in (Labels2DModel, Labels3DModel):
        # TODO: this should work, test better
        kwargs = {"prefilter": False}
        channel_names = None
    elif schema in (Image2DModel, Image3DModel):
        kwargs = {}
        channel_names = get_channels(data)
    else:
        raise ValueError(f"DataTree with schema {schema} not supported")

    get_axes_names(data)
    transformed_dict = {}
    raster_translation: Translation | None = None
    for k, v in data.items():
        assert len(v) == 1
        xdata = v.values().__iter__().__next__()

        composed: BaseTransformation
        if k == "scale0":
            composed = transformation
        else:
            scale = _get_scale(xdata.attrs["transform"])
            composed = Sequence([scale, transformation, scale.inverse()])

        transformed_dask, raster_translation_single_scale = _transform_raster(
            data=xdata.data, axes=xdata.dims, transformation=composed, **kwargs
        )
        if raster_translation is None:
            raster_translation = raster_translation_single_scale
        # we set a dummy empty dict for the transformation that will be replaced with the correct transformation for
        # each scale later in this function, when calling set_transformation()
        transformed_dict[k] = Dataset(
            {"image": DataArray(transformed_dask, dims=xdata.dims, name=xdata.name, attrs={TRANSFORM_KEY: {}})}
        )
        if channel_names is not None:
            # This expression returns a dataset now.
            transformed_dict[k] = transformed_dict[k].assign_coords(c=channel_names)

    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = DataTree.from_dict(transformed_dict)
    set_transformation(transformed_data, Identity(), to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _set_transformation_for_transformed_elements(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
        to_coordinate_system=to_coordinate_system,
    )
    transformed_data = compute_coordinates(transformed_data)
    schema().validate(transformed_data)
    return transformed_data


@transform.register(DaskDataFrame)
def _(
    data: DaskDataFrame,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> DaskDataFrame:
    from spatialdata.models import PointsModel
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations import Identity, get_transformation

    transformation, to_coordinate_system = _validate_target_coordinate_systems(
        data, transformation, maintain_positioning, to_coordinate_system
    )
    axes = get_axes_names(data)
    arrays = []
    for ax in axes:
        arrays.append(data[ax].to_dask_array(lengths=True).reshape(-1, 1))
    xdata = DataArray(da.concatenate(arrays, axis=1), coords={"points": range(len(data)), "dim": list(axes)})
    xtransformed = transformation._transform_coordinates(xdata)
    transformed = data.drop(columns=list(axes)).copy()
    # dummy transformation that will be replaced by _adjust_transformation()
    transformed.attrs[TRANSFORM_KEY] = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    # TODO: the following line, used in place of the line before, leads to an incorrect aggregation result. Look into
    #  this! Reported here: ...
    # transformed.attrs = {TRANSFORM_KEY: {DEFAULT_COORDINATE_SYSTEM: Identity()}}
    assert isinstance(transformed, DaskDataFrame)
    for ax in axes:
        indices = xtransformed["dim"] == ax
        new_ax = xtransformed[:, indices]
        transformed[ax] = new_ax.data.flatten()  # type: ignore[attr-defined]

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _set_transformation_for_transformed_elements(
        transformed,
        old_transformations,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
        to_coordinate_system=to_coordinate_system,
    )
    PointsModel.validate(transformed)
    return transformed


@transform.register(GeoDataFrame)
def _(
    data: GeoDataFrame,
    transformation: BaseTransformation | None = None,
    maintain_positioning: bool = False,
    to_coordinate_system: str | None = None,
) -> GeoDataFrame:
    from spatialdata.models import ShapesModel
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations import Identity, get_transformation

    axes = get_axes_names(data)
    transformation, to_coordinate_system = _validate_target_coordinate_systems(
        data, transformation, maintain_positioning, to_coordinate_system
    )
    # TODO: nitpick, mypy expects a listof literals and here we have a list of strings.
    # I ignored but we may want to fix this
    affine = transformation.to_affine(axes, axes)  # type: ignore[arg-type]
    matrix = affine.matrix
    shapely_notation = matrix[:-1, :-1].ravel().tolist() + matrix[:-1, -1].tolist()
    transformed_geometry = data.geometry.affine_transform(shapely_notation)
    transformed_data = data.copy(deep=True)
    transformed_data.attrs[TRANSFORM_KEY] = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    transformed_data.geometry = transformed_geometry

    if isinstance(transformed_geometry.iloc[0], Point) and "radius" in transformed_data.columns:
        old_radii = transformed_data["radius"].to_numpy()
        new_radii = scale_radii(radii=old_radii, affine=affine, axes=axes)
        transformed_data["radius"] = new_radii

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _set_transformation_for_transformed_elements(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
        to_coordinate_system=to_coordinate_system,
    )
    ShapesModel.validate(transformed_data)
    return transformed_data
