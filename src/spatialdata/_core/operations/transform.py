from __future__ import annotations

import itertools
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional

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
from spatialdata.transformations.operations import set_transformation

if TYPE_CHECKING:
    from spatialdata.transformations.transformations import (
        BaseTransformation,
        Translation,
    )

# from spatialdata._core.ngff.ngff_coordinate_system import NgffCoordinateSystem

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


def _prepend_transformation(
    element: SpatialElement,
    transformation: BaseTransformation,
    raster_translation: Optional[Translation],
    maintain_positioning: bool,
) -> None:
    """Prepend a transformation to an element.

    After an element has been transformed, this method is called to eventually prepend a particular transformation to
    the existing transformations of the element. The transformation to prepend depends on the type of the element (
    raster vs non-raster) and on the maintain_positioning flag.

    Parameters
    ----------
    element
        The spatial element to which the transformation should be prepended
    transformation
        The transformation to prepend
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
    from spatialdata.transformations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import Identity, Sequence

    to_prepend: Optional[BaseTransformation] = None
    if isinstance(element, (SpatialImage, MultiscaleSpatialImage)):
        if maintain_positioning:
            assert raster_translation is not None
            to_prepend = Sequence([raster_translation, transformation.inverse()])
        else:
            to_prepend = raster_translation

    elif isinstance(element, (GeoDataFrame, DaskDataFrame)):
        assert raster_translation is None
        if maintain_positioning:
            to_prepend = transformation.inverse()
    else:
        raise TypeError(f"Unsupported type {type(element)}")

    d = get_transformation(element, get_all=True)
    assert isinstance(d, dict)
    if len(d) == 0:
        logger.info(
            f"No transformations found in the element,"
            f"adding a default identity transformation to the coordinate system "
            f"{DEFAULT_COORDINATE_SYSTEM}"
        )
        d = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    for cs, t in d.items():
        new_t: BaseTransformation
        new_t = Sequence([to_prepend, t]) if to_prepend is not None else t
        set_transformation(element, new_t, to_coordinate_system=cs)


@singledispatch
def transform(data: Any, transformation: BaseTransformation, maintain_positioning: bool = False) -> Any:
    """
    Transform a spatial element using this transformation and returns the transformed element.

    Parameters
    ----------
    element
        Spatial element to transform.
    maintain_positioning
        If True, in the transformed element, each transformation that was present in the original element will be
        prepended with the inverse of the transformation used to transform the data (i.e. the current
        transformation for which .transform() is called). In this way the data is transformed but the
        positioning (for each coordinate system) is maintained. A use case is changing the orientation/scale/etc. of
        the data but keeping the alignment of the data within each coordinate system.
        If False, the data is simply transformed and the positioning (for each coordinate system) changes. For
        raster data, the translation part of the transformation is prepended to any tranformation already present in
        the element (see Notes below for more details). Furthermore, again in the case of raster data,
        if the transformation being applied has a rotation-like component, then the translation that is prepended
        also takes into account for the fact that the rotated data will have some paddings on each corner, and so
        it's origin must be shifted accordingly.
        Please see notes for more details of how this parameter interact with xarray.DataArray for raster data.

    Returns
    -------
    SpatialElement: Transformed spatial element.

    Notes
    -----
    An affine transformation contains a linear transformation and a translation. For raster types,
    only the linear transformation is applied to the data (e.g. the data is rotated or resized), but not the
    translation part.
    This means that calling Translation(...).transform(raster_element) will have the same effect as pre-pending the
    translation to each transformation of the raster element.
    Similarly, `Translation(...).transform(raster_element, maintain_positioning=True)` will not modify the raster
    element. We are considering to change this behavior by letting translations modify the coordinates stored with
    xarray.DataArray. If you are interested in this use case please get in touch by opening a GitHub Issue.
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
    from spatialdata.models import (
        Image2DModel,
        Image3DModel,
        Labels2DModel,
        Labels3DModel,
    )
    from spatialdata.transformations import get_transformation, set_transformation

    # labels need to be preserved after the resizing of the image
    if schema in (Labels2DModel, Labels3DModel):
        kwargs = {"prefilter": False, "order": 0}
    elif schema in (Image2DModel, Image3DModel):
        kwargs = {}
    else:
        raise ValueError(f"Unsupported schema {schema}")

    axes = get_axes_names(data)
    transformed_dask, raster_translation = _transform_raster(
        data=data.data, axes=axes, transformation=transformation, **kwargs
    )
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes)  # type: ignore[call-arg,arg-type]
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    set_transformation(transformed_data, old_transformations.copy(), set_all=True)
    _prepend_transformation(
        transformed_data,
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
    from spatialdata.transformations import get_transformation, set_transformation
    from spatialdata.transformations.transformations import BaseTransformation, Sequence

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
    for k, v in data.items():
        assert len(v) == 1
        xdata = v.values().__iter__().__next__()

        composed: BaseTransformation
        if k == "scale0":
            composed = transformation
        else:
            scale = _get_scale(xdata.attrs["transform"])
            composed = Sequence([scale, transformation, scale.inverse()])

        transformed_dask, raster_translation = _transform_raster(
            data=xdata.data, axes=xdata.dims, transformation=composed, **kwargs
        )
        transformed_dict[k] = SpatialImage(transformed_dask, dims=xdata.dims, name=xdata.name)

    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = MultiscaleSpatialImage.from_dict(transformed_dict)
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    set_transformation(transformed_data, old_transformations.copy(), set_all=True)
    _prepend_transformation(
        transformed_data,
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
    from spatialdata.transformations import get_transformation, set_transformation

    axes = get_axes_names(data)
    arrays = []
    for ax in axes:
        arrays.append(data[ax].to_dask_array(lengths=True).reshape(-1, 1))
    xdata = DataArray(da.concatenate(arrays, axis=1), coords={"points": range(len(data)), "dim": list(axes)})
    xtransformed = transformation._transform_coordinates(xdata)
    transformed = data.drop(columns=list(axes))
    assert isinstance(transformed, DaskDataFrame)
    for ax in axes:
        indices = xtransformed["dim"] == ax
        new_ax = xtransformed[:, indices]
        transformed[ax] = new_ax.data.flatten()  # type: ignore[attr-defined]

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    set_transformation(transformed, old_transformations.copy(), set_all=True)
    _prepend_transformation(
        transformed,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    PointsModel.validate(transformed)
    return transformed


@transform.register(GeoDataFrame)
def _(data: GeoDataFrame, transformation: BaseTransformation, maintain_positioning: bool = False) -> GeoDataFrame:
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import get_transformation

    ndim = len(get_axes_names(data))
    # TODO: nitpick, mypy expects a listof literals and here we have a list of strings.
    # I ignored but we may want to fix this
    matrix = transformation.to_affine_matrix(["x", "y", "z"][:ndim], ["x", "y", "z"][:ndim])  # type: ignore[arg-type]
    shapely_notation = matrix[:-1, :-1].ravel().tolist() + matrix[:-1, -1].tolist()
    transformed_geometry = data.geometry.affine_transform(shapely_notation)
    transformed_data = data.copy(deep=True)
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
    set_transformation(transformed_data, old_transformations.copy(), set_all=True)
    _prepend_transformation(
        transformed_data,
        transformation,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    ShapesModel.validate(transformed_data)
    return transformed_data
