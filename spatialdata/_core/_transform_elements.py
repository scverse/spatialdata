from __future__ import annotations

import itertools
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import dask_image.ndinterp
import numpy as np
from dask.array.core import Array as DaskArray
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from skimage.transform import estimate_transform
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import SpatialData
from spatialdata._core._spatialdata_ops import get_transformation, set_transformation
from spatialdata._core.core_utils import (
    DEFAULT_COORDINATE_SYSTEM,
    MappingToCoordinateSystem_t,
    SpatialElement,
    get_dims,
)
from spatialdata._core.models import get_schema
from spatialdata._logging import logger
from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata._core.transformations import (
        Affine,
        BaseTransformation,
        Translation,
    )

# from spatialdata._core.ngff.ngff_coordinate_system import NgffCoordinateSystem

DEBUG_WITH_PLOTS = False


def _transform_raster(
    data: DaskArray, axes: tuple[str, ...], transformation: BaseTransformation, **kwargs: Any
) -> tuple[DaskArray, Translation]:
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
    translation = Translation(translation_vector, axes=axes)
    inverse_matrix_adjusted = Sequence(
        [
            translation,
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
    return transformed_dask, translation


def _prepend_transformation(
    element: SpatialElement,
    transformation: BaseTransformation,
    old_transformations: MappingToCoordinateSystem_t,
    raster_translation: Optional[Translation],
    maintain_positioning: bool,
) -> None:
    ##
    from spatialdata._core._spatialdata_ops import set_transformation
    from spatialdata._core.transformations import Identity, Sequence

    to_prepend: Optional[BaseTransformation] = None
    if isinstance(element, SpatialImage) or isinstance(element, MultiscaleSpatialImage):
        if maintain_positioning:
            assert raster_translation is not None
            to_prepend = Sequence([raster_translation, transformation.inverse()])
        else:
            to_prepend = raster_translation

    elif isinstance(element, GeoDataFrame) or isinstance(element, DaskDataFrame):
        assert raster_translation is None
        if maintain_positioning:
            to_prepend = transformation.inverse()
    else:
        raise TypeError(f"Unsupported type {type(element)}")

    d = old_transformations
    if len(d) == 0:
        logger.info(
            f"No transformations found in the element, adding a default identity transformation to the coordinate system "
            f"{DEFAULT_COORDINATE_SYSTEM}"
        )
        d = {DEFAULT_COORDINATE_SYSTEM: Identity()}
    for cs, t in d.items():
        new_t: BaseTransformation
        if to_prepend is not None:
            new_t = Sequence([to_prepend, t])
        else:
            new_t = t
        set_transformation(element, new_t, to_coordinate_system=cs)


@singledispatch
def _transform(data: Any, transformation: BaseTransformation, maintain_positioning: bool) -> Any:
    """This function is documented in the docstring of BaseTransformation.transform()"""
    raise NotImplementedError()


@_transform.register(SpatialData)
def _(data: SpatialData, transformation: BaseTransformation, maintain_positioning: bool) -> SpatialData:
    new_elements: dict[str, dict[str, Any]] = {}
    for element_type in ["images", "labels", "points", "shapes"]:
        d = getattr(data, element_type)
        if len(d) > 0:
            new_elements[element_type] = {}
        for k, v in d.items():
            new_elements[element_type][k] = transformation.transform(v, maintain_positioning=maintain_positioning)
    new_sdata = SpatialData(**new_elements)
    return new_sdata


@_transform.register(SpatialImage)
def _(data: SpatialImage, transformation: BaseTransformation, maintain_positioning: bool) -> SpatialImage:
    schema = get_schema(data)
    from spatialdata._core._spatialdata_ops import get_transformation
    from spatialdata._core.models import Labels2DModel, Labels3DModel

    # labels need to be preserved after the resizing of the image
    if schema == Labels2DModel or schema == Labels3DModel:
        # TODO: this should work, test better
        kwargs = {"prefilter": False}
    else:
        kwargs = {}

    axes = get_dims(data)
    transformed_dask, raster_translation = _transform_raster(
        data=data.data, axes=axes, transformation=transformation, **kwargs
    )
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes)  # type: ignore[call-arg,arg-type]
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _prepend_transformation(
        transformed_data,
        transformation,
        old_transformations=old_transformations,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
    )
    return transformed_data


@_transform.register(MultiscaleSpatialImage)
def _(
    data: MultiscaleSpatialImage, transformation: BaseTransformation, maintain_positioning: bool
) -> MultiscaleSpatialImage:
    schema = get_schema(data)
    from spatialdata._core._spatialdata_ops import get_transformation
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
    transformed_dask, raster_translation = _transform_raster(
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
        except OverflowError as e:
            raise e
    # mypy thinks that schema could be ShapesModel, PointsModel, ...
    transformed_data = schema.parse(transformed_dask, dims=axes, multiscale_factors=multiscale_factors)  # type: ignore[call-arg,arg-type]
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _prepend_transformation(
        transformed_data,
        transformation,
        old_transformations=old_transformations,
        raster_translation=raster_translation,
        maintain_positioning=maintain_positioning,
    )
    return transformed_data


@_transform.register(DaskDataFrame)
def _(data: DaskDataFrame, transformation: BaseTransformation, maintain_positioning: bool) -> DaskDataFrame:
    from spatialdata._core._spatialdata_ops import get_transformation

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
    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    _prepend_transformation(
        transformed,
        transformation,
        old_transformations=old_transformations,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    return transformed


@_transform.register(GeoDataFrame)
def _(data: GeoDataFrame, transformation: BaseTransformation, maintain_positioning: bool) -> GeoDataFrame:
    from spatialdata._core._spatialdata_ops import get_transformation

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

    old_transformations = get_transformation(data, get_all=True)
    assert isinstance(old_transformations, dict)
    ShapesModel.validate(transformed_data)
    _prepend_transformation(
        transformed_data,
        transformation,
        old_transformations=old_transformations,
        raster_translation=None,
        maintain_positioning=maintain_positioning,
    )
    return transformed_data


def get_transformation_between_landmarks(
    references_coords: Union[GeoDataFrame, DaskDataFrame],
    moving_coords: Union[GeoDataFrame, DaskDataFrame],
) -> Affine:
    """
    Get a similarity transformation between two lists of (n >= 3) landmarks. Landmarks are assumed to be in the same space.

    Parameters
    ----------
    references_coords
        landmarks annotating the reference element. Must be a valid element describing points or circles.
    moving_coords
        landmarks annotating the moving element. Must be a valid element describing points or circles.

    Returns
    -------
    The Affine transformation that maps the moving element to the reference element.

    Examples
    --------
    If you save the landmark points using napari_spatialdata, they will be alredy saved as circles. Here is an
    example on how to call this function on two sets of numpy arrays describing x, y coordinates.
    >>> import numpy as np
    >>> from spatialdata.models import PointsModel
    >>> from spatialdata.transform import get_transformation_between_landmarks
    >>> points_moving = np.array([[0, 0], [1, 1], [2, 2]])
    >>> points_reference = np.array([[0, 0], [10, 10], [20, 20]])
    >>> moving_coords = PointsModel(points_moving)
    >>> references_coords = PointsModel(points_reference)
    >>> transformation = get_transformation_between_landmarks(references_coords, moving_coords)
    """
    from spatialdata._core.transformations import Affine, BaseTransformation, Sequence

    assert get_dims(references_coords) == ("x", "y")
    assert get_dims(moving_coords) == ("x", "y")

    if isinstance(references_coords, GeoDataFrame):
        references_xy = np.stack([references_coords.geometry.x, references_coords.geometry.y], axis=1)
        moving_xy = np.stack([moving_coords.geometry.x, moving_coords.geometry.y], axis=1)
    elif isinstance(references_coords, DaskDataFrame):
        references_xy = references_coords[["x", "y"]].to_dask_array().compute()
        moving_xy = moving_coords[["x", "y"]].to_dask_array().compute()
    else:
        raise TypeError("references_coords must be either an GeoDataFrame or a DaskDataFrame")

    model = estimate_transform("affine", src=moving_xy, dst=references_xy)
    transform_matrix = model.params
    a = transform_matrix[:2, :2]
    d = np.linalg.det(a)
    final: BaseTransformation
    if d < 0:
        m = (moving_xy[:, 0].max() - moving_xy[:, 0].min()) / 2
        flip = Affine(
            np.array(
                [
                    [-1, 0, 2 * m],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        flipped_moving = flip.transform(moving_coords)
        if isinstance(flipped_moving, GeoDataFrame):
            flipped_moving_xy = np.stack([flipped_moving.geometry.x, flipped_moving.geometry.y], axis=1)
        elif isinstance(flipped_moving, DaskDataFrame):
            flipped_moving_xy = flipped_moving[["x", "y"]].to_dask_array().compute()
        else:
            raise TypeError("flipped_moving must be either an GeoDataFrame or a DaskDataFrame")
        model = estimate_transform("similarity", src=flipped_moving_xy, dst=references_xy)
        final = Sequence([flip, Affine(model.params, input_axes=("x", "y"), output_axes=("x", "y"))])
    else:
        model = estimate_transform("similarity", src=moving_xy, dst=references_xy)
        final = Affine(model.params, input_axes=("x", "y"), output_axes=("x", "y"))

    affine = Affine(
        final.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    return affine


def align_elements_using_landmarks(
    references_coords: Union[GeoDataFrame | DaskDataFrame],
    moving_coords: Union[GeoDataFrame | DaskDataFrame],
    reference_element: SpatialElement,
    moving_element: SpatialElement,
    reference_coordinate_system: str = "global",
    moving_coordinate_system: str = "global",
    new_coordinate_system: Optional[str] = None,
    write_to_sdata: Optional[SpatialData] = None,
) -> BaseTransformation:
    """
    Maps a moving object into a reference object using two lists of (n >= 3) landmarks; returns the transformations that enable this
    mapping and optinally saves them, to map to a new shared coordinate system.

    Parameters
    ----------
    references_coords
        landmarks annotating the reference element. Must be a valid element describing points or circles.
    moving_coords
        landmarks annotating the moving element. Must be a valid element describing points or circles.
    reference_element
        the reference element.
    moving_element
        the moving element.
    reference_coordinate_system
        the coordinate system of the reference element that have been used to annotate the landmarks.
    moving_coordinate_system
        the coordinate system of the moving element that have been used to annotate the landmarks.
    new_coordinate_system
        If provided, both elements will be mapped to this new coordinate system with the new transformations just
        computed.
    write_to_sdata
        If provided, the transformations will be saved to disk in the specified SpatialData object. The SpatialData
        object must be backed and must contain both the reference and moving elements.

    Returns
    -------
    A similarity transformation that maps the moving element to the same coordinate of reference element in the
    coordinate system specified by reference_coordinate_system.
    """
    from spatialdata._core.transformations import BaseTransformation, Sequence

    affine = get_transformation_between_landmarks(references_coords, moving_coords)

    # get the old transformations of the visium and xenium data
    old_moving_transformation = get_transformation(moving_element, moving_coordinate_system)
    old_reference_transformation = get_transformation(reference_element, reference_coordinate_system)
    assert isinstance(old_moving_transformation, BaseTransformation)
    assert isinstance(old_reference_transformation, BaseTransformation)

    # compute the new transformations
    new_moving_transformation = Sequence([old_moving_transformation, affine])
    new_reference_transformation = old_reference_transformation

    if new_coordinate_system is not None:
        # this allows to work on singleton objects, not embedded in a SpatialData object
        set_transformation(
            moving_element, new_moving_transformation, new_coordinate_system, write_to_sdata=write_to_sdata
        )
        set_transformation(
            reference_element, new_reference_transformation, new_coordinate_system, write_to_sdata=write_to_sdata
        )
    return new_moving_transformation
