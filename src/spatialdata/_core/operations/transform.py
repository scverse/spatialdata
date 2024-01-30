from __future__ import annotations

import itertools
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import dask.array as da
import dask_image.ndinterp
import numpy as np
from dask.array.core import Array as DaskArray
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import Point
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata.models import SpatialElement, get_axes_names, get_model
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from spatialdata.transformations._utils import _get_scale, compute_coordinates

if TYPE_CHECKING:
    from spatialdata.transformations.transformations import (
        BaseTransformation,
        Translation,
    )

DEBUG_WITH_PLOTS = False


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
    new_v = (matrix @ v.T).T
    c_shape: tuple[int, ...]
    c_shape = (data.shape[0],) if "c" in axes else ()
    new_spatial_shape = tuple(
        int(np.max(new_v[:, i]) - np.min(new_v[:, i])) for i in range(len(c_shape), n_spatial_dims + len(c_shape))
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


def _adjust_transformations(
    element: SpatialElement,
    old_transformations: dict[str, BaseTransformation],
    transformation: BaseTransformation,
    raster_translation: Translation | None,
    maintain_positioning: bool,
) -> None:
    """Adjust the transformations of an element after it has been transformed.

    Parameters
    ----------
    element
        The SpatialElement to which the transformation should be prepended
    old_transformations
        The transformations that were present in the element before the data was transformed
    transformation
        The transformation that was used to transform the data
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
    """
    from spatialdata.transformations import (
        BaseTransformation,
        get_transformation,
        remove_transformation,
        set_transformation,
    )
    from spatialdata.transformations.transformations import Identity, Sequence

    to_prepend: BaseTransformation | None
    if isinstance(element, (SpatialImage, MultiscaleSpatialImage)):
        if maintain_positioning:
            assert raster_translation is not None
            to_prepend = Sequence([raster_translation, transformation.inverse()])
        else:
            to_prepend = raster_translation
    elif isinstance(element, (GeoDataFrame, DaskDataFrame)):
        assert raster_translation is None
        to_prepend = transformation.inverse() if maintain_positioning else Identity()
    else:
        raise TypeError(f"Unsupported type {type(element)}")
    assert isinstance(to_prepend, BaseTransformation)

    d = get_transformation(element, get_all=True)
    assert isinstance(d, dict)
    assert len(d) == 1
    assert isinstance(d[DEFAULT_COORDINATE_SYSTEM], Identity)
    remove_transformation(element, remove_all=True)

    # if maintain_positioning:
    if True:
        for cs, t in old_transformations.items():
            new_t: BaseTransformation
            new_t = Sequence([to_prepend, t])
            set_transformation(element, new_t, to_coordinate_system=cs)
    # else:
    #     set_transformation(element, to_prepend, to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)


@singledispatch
def transform(data: Any, transformation: BaseTransformation, maintain_positioning: bool = False) -> Any:
    """
    Transform a SpatialElement using this transformation and returns the transformed element.

    Parameters
    ----------
    element
        SpatialElement to transform.
    maintain_positioning
        If True, in the transformed element, each transformation that was present in the original element will be
        prepended with the inverse of the transformation used to transform the data (i.e. the current
        transformation for which .transform() is called). In this way the data is transformed but the
        positioning (for each coordinate system) is maintained. A use case is changing the orientation/scale/etc. of
        the data but keeping the alignment of the data within each coordinate system.
        If False, the data is simply transformed and the positioning changes; only the coordinate system in which the
        data is transformed to is kept. For raster data, the translation part of the transformation is assigned to the
        element (see Notes below for more details). Furthermore, again in the case of raster data, if the transformation
        being applied has a rotation-like component, then the translation will take into account for the fact that the
        rotated data will have some paddings on each corner, and so it's origin must be shifted accordingly.
        Please see notes for more details of how this parameter interact with xarray.DataArray for raster data.

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
    translation to the element in the new coordinate system (if maintain_positioning=False).
    We are considering to change this behavior by letting translations modify the coordinates stored with
    xarray.DataArray; this is tracked here: https://github.com/scverse/spatialdata/issues/308
    """
    raise RuntimeError(f"Cannot transform {type(data)}")


@transform.register(SpatialData)
def _(data: SpatialData, transformation: BaseTransformation, maintain_positioning: bool = False) -> SpatialData:
    new_elements: dict[str, dict[str, Any]] = {}
    for element_type in ["images", "labels", "points", "shapes"]:
        d = getattr(data, element_type)
        if len(d) > 0:
            new_elements[element_type] = {}
        for k, v in d.items():
            new_elements[element_type][k] = transform(v, transformation, maintain_positioning=maintain_positioning)
    return SpatialData(**new_elements)


@transform.register(SpatialImage)
def _(data: SpatialImage, transformation: BaseTransformation, maintain_positioning: bool = False) -> SpatialImage:
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
    assert isinstance(transformed_data, SpatialImage)
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _adjust_transformations(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
    )
    transformed_data = compute_coordinates(transformed_data)
    schema().validate(transformed_data)
    return transformed_data


@transform.register(MultiscaleSpatialImage)
def _(
    data: MultiscaleSpatialImage, transformation: BaseTransformation, maintain_positioning: bool = False
) -> MultiscaleSpatialImage:
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
    elif schema in (Image2DModel, Image3DModel):
        kwargs = {}
    else:
        raise ValueError(f"MultiscaleSpatialImage with schema {schema} not supported")

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
        transformed_dict[k] = SpatialImage(
            transformed_dask, dims=xdata.dims, name=xdata.name, attrs={TRANSFORM_KEY: {}}
        )

    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = MultiscaleSpatialImage.from_dict(transformed_dict)
    set_transformation(transformed_data, Identity(), to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _adjust_transformations(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
    )
    transformed_data = compute_coordinates(transformed_data)
    schema().validate(transformed_data)
    return transformed_data


@transform.register(DaskDataFrame)
def _(data: DaskDataFrame, transformation: BaseTransformation, maintain_positioning: bool = False) -> DaskDataFrame:
    from spatialdata.models import PointsModel
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations import Identity, get_transformation

    axes = get_axes_names(data)
    arrays = []
    for ax in axes:
        arrays.append(data[ax].to_dask_array(lengths=True).reshape(-1, 1))
    xdata = DataArray(da.concatenate(arrays, axis=1), coords={"points": range(len(data)), "dim": list(axes)})
    xtransformed = transformation._transform_coordinates(xdata)
    transformed = data.drop(columns=list(axes)).copy()
    # dummy transformation that will be replaced by _adjust_transformation()
    transformed.attrs[TRANSFORM_KEY] = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    assert isinstance(transformed, DaskDataFrame)
    for ax in axes:
        indices = xtransformed["dim"] == ax
        new_ax = xtransformed[:, indices]
        transformed[ax] = new_ax.data.flatten()  # type: ignore[attr-defined]

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _adjust_transformations(
        transformed,
        old_transformations,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    PointsModel.validate(transformed)
    return transformed


@transform.register(GeoDataFrame)
def _(data: GeoDataFrame, transformation: BaseTransformation, maintain_positioning: bool = False) -> GeoDataFrame:
    from spatialdata.models import ShapesModel
    from spatialdata.models._utils import TRANSFORM_KEY
    from spatialdata.transformations import Identity, get_transformation

    ndim = len(get_axes_names(data))
    # TODO: nitpick, mypy expects a listof literals and here we have a list of strings.
    # I ignored but we may want to fix this
    matrix = transformation.to_affine_matrix(["x", "y", "z"][:ndim], ["x", "y", "z"][:ndim])  # type: ignore[arg-type]
    shapely_notation = matrix[:-1, :-1].ravel().tolist() + matrix[:-1, -1].tolist()
    transformed_geometry = data.geometry.affine_transform(shapely_notation)
    transformed_data = data.copy(deep=True)
    transformed_data.attrs[TRANSFORM_KEY] = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    transformed_data.geometry = transformed_geometry

    if isinstance(transformed_geometry.iloc[0], Point) and "radius" in transformed_data.columns:
        old_radius = transformed_data["radius"]
        eigenvalues = np.linalg.eigvals(matrix[:-1, :-1])
        modules = np.absolute(eigenvalues)
        if not np.allclose(modules, modules[0]):
            logger.warning(
                "The transformation matrix is not isotropic, the radius will be scaled by the average of the "
                "eigenvalues of the affine transformation matrix"
            )
            scale_factor = np.mean(modules)
        else:
            scale_factor = modules[0]
        new_radius = old_radius * scale_factor
        transformed_data["radius"] = new_radius

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _adjust_transformations(
        transformed_data,
        old_transformations,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    ShapesModel.validate(transformed_data)
    return transformed_data
