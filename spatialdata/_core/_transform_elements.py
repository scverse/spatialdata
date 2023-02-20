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
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import SpatialData
from spatialdata._core.core_utils import get_dims
from spatialdata._core.models import get_schema
from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata._core.transformations import BaseTransformation

# from spatialdata._core.ngff.ngff_coordinate_system import NgffCoordinateSystem

DEBUG_WITH_PLOTS = False


def _transform_raster(
    data: DaskArray, axes: tuple[str, ...], transformation: BaseTransformation, **kwargs: Any
) -> DaskArray:
    # dims = {ch: axes.index(ch) for ch in axes}
    from spatialdata._core.transformations import Sequence, Translation

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
    if "c" in axes:
        c_shape = (data.shape[0],)
    else:
        c_shape = ()
    new_spatial_shape = tuple(
        int(np.max(new_v[:, i]) - np.min(new_v[:, i])) for i in range(len(c_shape), n_spatial_dims + len(c_shape))
    )
    output_shape = c_shape + new_spatial_shape
    ##
    translation_vector = np.min(new_v[:, :-1], axis=0)
    inverse_matrix_adjusted = Sequence(
        [
            Translation(translation_vector, axes=axes),
            transformation.inverse(),
        ]
    ).to_affine_matrix(input_axes=axes, output_axes=axes)

    # fix chunk shape, it should be possible for the user to specify them, and by default we could reuse the chunk shape of the input
    # output_chunks = data.chunks
    ##
    transformed_dask = dask_image.ndinterp.affine_transform(
        data,
        matrix=inverse_matrix_adjusted,
        output_shape=output_shape,
        **kwargs,
        # , output_chunks=output_chunks
    )
    assert isinstance(transformed_dask, DaskArray)
    ##

    if DEBUG_WITH_PLOTS:
        if n_spatial_dims == 2:
            ##
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
            ##
        else:
            assert n_spatial_dims == 3
            # raise NotImplementedError()
    return transformed_dask


@singledispatch
def _transform(data: Any, transformation: BaseTransformation) -> Any:
    raise NotImplementedError()


@_transform.register(SpatialData)
def _(data: SpatialData, transformation: BaseTransformation) -> SpatialData:
    new_elements: dict[str, dict[str, Any]] = {}
    for element_type in ["images", "labels", "points", "shapes"]:
        d = getattr(data, element_type)
        if len(d) > 0:
            new_elements[element_type] = {}
        for k, v in d.items():
            new_elements[element_type][k] = transformation.transform(v)

    new_sdata = SpatialData(**new_elements)
    return new_sdata


@_transform.register(SpatialImage)
def _(data: SpatialImage, transformation: BaseTransformation) -> SpatialImage:
    schema = get_schema(data)
    from spatialdata._core.models import Labels2DModel, Labels3DModel

    # labels need to be preserved after the resizing of the image
    if schema == Labels2DModel or schema == Labels3DModel:
        # TODO: this should work, test better
        kwargs = {"prefilter": False}
    else:
        kwargs = {}

    axes = get_dims(data)
    transformed_dask = _transform_raster(data=data.data, axes=axes, transformation=transformation, **kwargs)
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes)  # type: ignore[call-arg,arg-type]
    print(
        "TODO: compose the transformation!!!! we need to put the previous one concatenated with the translation showen above. The translation operates before the other transformation"
    )
    return transformed_data


@_transform.register(MultiscaleSpatialImage)
def _(data: MultiscaleSpatialImage, transformation: BaseTransformation) -> MultiscaleSpatialImage:
    schema = get_schema(data)
    from spatialdata._core.models import Labels2DModel, Labels3DModel

    # labels need to be preserved after the resizing of the image
    if schema == Labels2DModel or schema == Labels3DModel:
        # TODO: this should work, test better
        kwargs = {"prefilter": False}
    else:
        kwargs = {}

    axes = get_dims(data)
    scale0 = dict(data["scale0"])
    assert len(scale0) == 1
    scale0_data = scale0.values().__iter__().__next__()
    transformed_dask = _transform_raster(
        data=scale0_data.data, axes=scale0_data.dims, transformation=transformation, **kwargs
    )

    # this code is temporary and doens't work in all cases (in particular it breaks when the data is not similar
    # to a square but has sides of very different lengths). I would remove it an implement (inside the parser)
    # the logic described in https://github.com/scverse/spatialdata/issues/108)
    shapes = []
    for level in range(len(data)):
        dims = data[f"scale{level}"].dims.values()
        shape = np.array([dict(dims._mapping)[k] for k in axes if k != "c"])
        shapes.append(shape)
    multiscale_factors = []
    shape0 = shapes[0]
    for shape in shapes[1:]:
        factors = shape0 / shape
        factors - min(factors)
        # assert np.allclose(almost_zero, np.zeros_like(almost_zero), rtol=2.)
        try:
            multiscale_factors.append(round(factors[0]))
        except ValueError as e:
            raise e
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes, scale_factors=multiscale_factors)  # type: ignore[call-arg,arg-type]
    print(
        "TODO: compose the transformation!!!! we need to put the previous one concatenated with the translation showen above. The translation operates before the other transformation"
    )
    return transformed_data


@_transform.register(DaskDataFrame)
def _(data: DaskDataFrame, transformation: BaseTransformation) -> DaskDataFrame:
    axes = get_dims(data)
    arrays = []
    for ax in axes:
        arrays.append(data[ax].to_dask_array())
    xdata = DataArray(np.array(arrays).T, coords={"points": range(len(data)), "dim": list(axes)})
    xtransformed = transformation._transform_coordinates(xdata)
    transformed = data.drop(columns=list(axes))
    assert isinstance(transformed, DaskDataFrame)
    for ax in axes:
        indices = xtransformed["dim"] == ax
        new_ax = xtransformed[:, indices].data.flatten()
        # mypy says that from_array is not a method of DaskDataFrame, but it is
        transformed[ax] = da.from_array(np.array(new_ax))  # type: ignore[attr-defined]

    # to avoid cyclic import
    from spatialdata._core.models import PointsModel

    PointsModel.validate(transformed)
    return transformed


@_transform.register(GeoDataFrame)
def _(data: GeoDataFrame, transformation: BaseTransformation) -> GeoDataFrame:
    ##
    ndim = len(get_dims(data))
    # TODO: nitpick, mypy expects a listof literals and here we have a list of strings. I ignored but we may want to fix this
    matrix = transformation.to_affine_matrix(["x", "y", "z"][:ndim], ["x", "y", "z"][:ndim])  # type: ignore[arg-type]
    shapely_notation = matrix[:-1, :-1].ravel().tolist() + matrix[:-1, -1].tolist()
    transformed_geometry = data.geometry.affine_transform(shapely_notation)
    transformed_data = data.copy(deep=True)
    transformed_data.geometry = transformed_geometry

    # to avoid cyclic import
    from spatialdata._core.models import ShapesModel

    ShapesModel.validate(transformed_data)
    return transformed_data
