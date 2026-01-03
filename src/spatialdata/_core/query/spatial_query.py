import warnings
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import numpy as np
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata import to_polygons
from spatialdata._core.query._utils import _get_filtered_or_unfiltered_tables, get_bounding_box_corners
from spatialdata._core.spatialdata import SpatialData
from spatialdata._docs import docstring_parameter
from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array
from spatialdata.models import (
    PointsModel,
    ShapesModel,
    SpatialElement,
    get_axes_names,
    points_dask_dataframe_to_geopandas,
    points_geopandas_to_dask_dataframe,
)
from spatialdata.models._utils import ValidAxis_t, get_spatial_axes
from spatialdata.models.models import ATTRS_KEY
from spatialdata.transformations.operations import set_transformation
from spatialdata.transformations.transformations import Affine, BaseTransformation, _get_affine_for_element

MIN_COORDINATE_DOCS = """\
    The upper left hand corners of the bounding boxes (i.e., minimum coordinates along all dimensions).
        Shape: (n_boxes, n_axes) or (n_axes,) for a single box.
"""
MAX_COORDINATE_DOCS = """\
    The lower right hand corners of the bounding boxes (i.e., the maximum coordinates along all dimensions).
        Shape: (n_boxes, n_axes)
"""


@docstring_parameter(min_coordinate_docs=MIN_COORDINATE_DOCS, max_coordinate_docs=MAX_COORDINATE_DOCS)
def _get_bounding_box_corners_in_intrinsic_coordinates(
    element: SpatialElement,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> tuple[DataArray, tuple[str, ...]]:
    """Get all corners of a bounding box in the intrinsic coordinates of an element.

    Parameters
    ----------
    element
        The SpatialElement to get the intrinsic coordinate system from.
    axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
    {min_coordinate_docs}
    max_coordinate
    {max_coordinate_docs}
    target_coordinate_system
        The coordinate system the bounding box is defined in.

    Returns
    -------
    All the corners of the bounding box in the intrinsic coordinate system of the element. The shape
    is (2, 4) when axes has 2 spatial dimensions, and (2, 8) when axes has 3 spatial dimensions.

    The axes of the intrinsic coordinate system.
    """
    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # compute the output axes of the transformation, remove c from input and output axes, return the matrix without c
    # and then build an affine transformation from that
    m_without_c, input_axes_without_c, output_axes_without_c = _get_axes_of_tranformation(
        element, target_coordinate_system
    )
    spatial_transform = Affine(m_without_c, input_axes=input_axes_without_c, output_axes=output_axes_without_c)

    # we identified 5 cases (see the responsible function for details), cases 1 and 5 correspond to invertible
    # transformations; we focus on them. The following code triggers a validation that ensures we are in case 1 or 5.
    m_without_c_linear = m_without_c[:-1, :-1]
    _ = _get_case_of_bounding_box_query(m_without_c_linear, input_axes_without_c, output_axes_without_c)

    # adjust the bounding box to the real axes, dropping or adding eventually mismatching axes; the order of the axes is
    # not adjusted
    axes_adjusted, min_coordinate, max_coordinate = _adjust_bounding_box_to_real_axes(
        axes, min_coordinate, max_coordinate, output_axes_without_c
    )
    if set(axes_adjusted) != set(output_axes_without_c):
        raise ValueError("The axes of the bounding box must match the axes of the transformation.")

    # let's get the bounding box corners and inverse transform then to the intrinsic coordinate system; since we are
    # in case 1 or 5, the transformation is invertible
    spatial_transform_bb_axes = Affine(
        spatial_transform.to_affine_matrix(input_axes=input_axes_without_c, output_axes=axes_adjusted),
        input_axes=input_axes_without_c,
        output_axes=axes_adjusted,
    )

    bounding_box_corners = get_bounding_box_corners(
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes_adjusted,
    )

    inverse = spatial_transform_bb_axes.inverse()
    if not isinstance(inverse, Affine):
        raise RuntimeError("This should not happen")
    rotation_matrix = inverse.matrix[0:-1, 0:-1]
    translation = inverse.matrix[0:-1, -1]

    intrinsic_bounding_box_corners = bounding_box_corners.data @ rotation_matrix.T + translation

    if bounding_box_corners.ndim > 2:  # multiple boxes
        coords = {
            "box": range(len(bounding_box_corners)),
            "corner": range(bounding_box_corners.shape[1]),
            "axis": list(inverse.output_axes),
        }
    else:
        coords = {"corner": range(len(bounding_box_corners)), "axis": list(inverse.output_axes)}
    return (
        DataArray(
            intrinsic_bounding_box_corners,
            coords=coords,
        ),
        input_axes_without_c,
    )


def _get_polygon_in_intrinsic_coordinates(
    element: DaskDataFrame | GeoDataFrame, target_coordinate_system: str, polygon: Polygon | MultiPolygon
) -> GeoDataFrame:
    from spatialdata._core.operations.transform import transform

    polygon_gdf = ShapesModel.parse(GeoDataFrame(geometry=[polygon]))

    m_without_c, input_axes_without_c, output_axes_without_c = _get_axes_of_tranformation(
        element, target_coordinate_system
    )
    spatial_transform = Affine(m_without_c, input_axes=input_axes_without_c, output_axes=output_axes_without_c)

    m_without_c_linear = m_without_c[:-1, :-1]
    case = _get_case_of_bounding_box_query(m_without_c_linear, input_axes_without_c, output_axes_without_c)
    # as explained in https://github.com/scverse/spatialdata/pull/151#issuecomment-1444609101, this asserts that
    # the transformation between the intrinsic coordinate system and the query space, restricted to the domain
    # of the data, is invertible (either with dimension 2 or 3)
    assert case in [1, 5]

    # Since we asserted above that the transformation is invertible, then inverse image of the xy plane is a plane.
    # Here, to keep the implementation simple, we want to restrict to the case in which this inverse image plane is
    # parallel to the xy plane also in the intrinsic coordinate system.
    # If in the future there is a need to implement the general case we could proceed as follows.
    # 1. The data in the intrinsic coordinate system is necessarily points (because this function is not called for
    # raster data and 3D polygons/meshes are not implemented).
    # 2. We project the points to the inverse image plane.
    # 3. We query these new points in the inverse image plane.
    # Now, let's not handle this complexity and simply raise an error if, informally, the inverse transformation is
    # "mixing" the 'z' axis with the other axes, or formally, if the vector part of the affine transformation is not a
    # block diagonal matrix with one block for the z axis and one block for the x, y, c axes.
    sorted_input_axes_without_c = ("x", "y", "z")[: len(input_axes_without_c)]
    spatial_transform_bb_axes = Affine(
        spatial_transform.to_affine_matrix(input_axes=sorted_input_axes_without_c, output_axes=("x", "y")),
        input_axes=sorted_input_axes_without_c,
        output_axes=("x", "y"),
    )
    error_message = 'The transformation is mixing the "z" axis with the other axes. This case is not supported.'
    assert spatial_transform_bb_axes.matrix[2, 0] == 0, error_message
    assert spatial_transform_bb_axes.matrix[2, 1] == 0, error_message

    # now that we checked the above, we can restrict the transformation to the "x" and "y" axes; this will make it
    # invertible when the points to query are 3D
    spatial_transform_bb_axes = spatial_transform_bb_axes.to_affine(input_axes=("x", "y"), output_axes=("x", "y"))

    inverse = spatial_transform_bb_axes.inverse()
    assert isinstance(inverse, Affine)
    set_transformation(polygon_gdf, inverse, "inverse")

    return transform(polygon_gdf, to_coordinate_system="inverse")


def _get_axes_of_tranformation(
    element: SpatialElement, target_coordinate_system: str
) -> tuple[ArrayLike, tuple[str, ...], tuple[str, ...]]:
    """
    Get the transformation matrix and the transformation's axes (ignoring `c`).

    The transformation is the one from the element's intrinsic coordinate system to the query coordinate space.
    Note that the axes which specify the query shape are not necessarily the same as the axes that are output of the
    transformation

    Parameters
    ----------
    element
        SpatialData element to be transformed.
    target_coordinate_system
        The target coordinate system for the transformation.

    Returns
    -------
    m_without_c
        The transformation from the element's intrinsic coordinate system to the query coordinate space, without the
        "c" axis.
    input_axes_without_c
        The axes of the element's intrinsic coordinate system, without the "c" axis.
    output_axes_without_c
        The axes of the query coordinate system, without the "c" axis.

    """
    from spatialdata.transformations import get_transformation

    transform_to_query_space = get_transformation(element, to_coordinate_system=target_coordinate_system)
    assert isinstance(transform_to_query_space, BaseTransformation)
    m = _get_affine_for_element(element, transform_to_query_space)
    input_axes_without_c = tuple(ax for ax in m.input_axes if ax != "c")
    output_axes_without_c = tuple(ax for ax in m.output_axes if ax != "c")
    m_without_c = m.to_affine_matrix(input_axes=input_axes_without_c, output_axes=output_axes_without_c)
    return m_without_c, input_axes_without_c, output_axes_without_c


def _adjust_bounding_box_to_real_axes(
    axes_bb: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    axes_out_without_c: tuple[str, ...],
) -> tuple[tuple[str, ...], ArrayLike, ArrayLike]:
    """
    Adjust the bounding box to the real axes of the transformation.

    The bounding box is defined by the user and its axes may not coincide with the axes of the transformation.
    """
    # the following variable `axis` is the index of the axis in the variable min_coordinates that corresponds to the
    # named axes ('x', 'y', ...). We need it to know at which index to remove/add new named axes
    axis = min_coordinate.ndim - 1
    if set(axes_bb) != set(axes_out_without_c):
        axes_only_in_bb = set(axes_bb) - set(axes_out_without_c)
        axes_only_in_output = set(axes_out_without_c) - set(axes_bb)

        # let's remove from the bounding box whose axes that are not in the output axes (e.g. querying 2D points with a
        # 3D bounding box)
        indices_to_remove_from_bb = [axes_bb.index(ax) for ax in axes_only_in_bb]
        axes_bb = tuple(ax for ax in axes_bb if ax not in axes_only_in_bb)
        min_coordinate = np.delete(min_coordinate, indices_to_remove_from_bb, axis=axis)
        max_coordinate = np.delete(max_coordinate, indices_to_remove_from_bb, axis=axis)

        # if there are axes in the output axes that are not in the bounding box, we need to add them to the bounding box
        # with a range that includes everything (e.g. querying 3D points with a 2D bounding box)
        M = np.finfo(np.float32).max - 1
        for ax in axes_only_in_output:
            axes_bb = axes_bb + (ax,)
            min_coordinate = np.insert(min_coordinate, min_coordinate.shape[axis], -M, axis=axis)
            max_coordinate = np.insert(max_coordinate, max_coordinate.shape[axis], M, axis=axis)
    else:
        indices = [axes_bb.index(ax) for ax in axes_out_without_c]
        min_coordinate = np.take(min_coordinate, indices, axis=axis)
        max_coordinate = np.take(max_coordinate, indices, axis=axis)
        axes_bb = axes_out_without_c
    return axes_bb, min_coordinate, max_coordinate


def _get_case_of_bounding_box_query(
    m_without_c_linear: ArrayLike,
    input_axes_without_c: tuple[str, ...],
    output_axes_without_c: tuple[str, ...],
) -> int:
    """
    The bounding box query is handled in different ways depending on the "case" we are in, which we identify here.

    See https://github.com/scverse/spatialdata/pull/151#issuecomment-1444609101 for a detailed overview of the logic of
    this code, or see the comments below for an overview of the cases we consider.
    """  # noqa: D401
    transform_dimension = np.linalg.matrix_rank(m_without_c_linear)
    transform_coordinate_length = len(output_axes_without_c)
    data_dim = len(input_axes_without_c)

    assert data_dim in [2, 3]
    assert transform_dimension in [2, 3]
    assert transform_coordinate_length in [2, 3]
    assert not (data_dim == 2 and transform_dimension == 3)
    assert not (transform_dimension == 3 and transform_coordinate_length == 2)
    # TL;DR of the GitHub comment linked in the docstring.
    # the combinations of values for `data_dim`, `transform_dimension`, and `transform_coordinate_length` # lead to 8
    # cases, but of these, 3 are not possible:
    # a. transform_dimension == 3 and transform_coordinate_length == 2 (these are 2 cases)
    # b. data_dim == 2 and transform_dimension == 3 (these are 2 cases, but one already covered by the previous point)
    # what remains are the 5 cases of the if-elif-else below
    # for simplicity we will start by implementing only the cases 1 and 5, which are the ones that correspond to
    # invertible transformations, the other cases are not handled and will raise an error, and can be implemented in the
    # future if needed. The GitHub discussion also contains a detailed explanation of the logic behind the cases.
    if data_dim == 2 and transform_dimension == 2 and transform_coordinate_length == 2:
        case = 1
    elif data_dim == 2 and transform_dimension == 2 and transform_coordinate_length == 3:
        # currently not handled
        case = 2
    elif data_dim == 3 and transform_dimension == 2 and transform_coordinate_length == 2:
        # currently not handled
        case = 3
    elif data_dim == 3 and transform_dimension == 2 and transform_coordinate_length == 3:
        # currently not handled
        case = 4
    elif data_dim == 3 and transform_dimension == 3 and transform_coordinate_length == 3:
        case = 5
    else:
        raise RuntimeError("This should not happen")

    if case in [2, 3, 4]:
        # to implement case 2: we need to intersect the plane in the extrinsic coordinate system with the 3D bounding
        # box. The vertices of this polygon needs to be transformed to the intrinsic coordinate system
        error_message = (
            f"This case is not supported (data with dimension {data_dim} but transformation with rank "
            f"{transform_dimension}. Please open a GitHub issue if you want to discuss a use case."
        )
        raise ValueError(error_message)
    return case


@dataclass(frozen=True)
class BaseSpatialRequest:
    """Base class for spatial queries."""

    target_coordinate_system: str

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass


@docstring_parameter(min_coordinate_docs=MIN_COORDINATE_DOCS, max_coordinate_docs=MAX_COORDINATE_DOCS)
@dataclass(frozen=True)
class BoundingBoxRequest(BaseSpatialRequest):
    """Query with an axis-aligned bounding box.

    Attributes
    ----------
    axes
        The axes the coordinates are expressed in.
    min_coordinate
        {min_coordinate_docs}
    max_coordinate
        {max_coordinate_docs}
    """

    min_coordinate: ArrayLike
    max_coordinate: ArrayLike
    axes: tuple[ValidAxis_t, ...]

    def __post_init__(self) -> None:
        # validate the axes
        spatial_axes = get_spatial_axes(self.axes)
        non_spatial_axes = set(self.axes) - set(spatial_axes)
        if len(non_spatial_axes) > 0:
            raise ValueError(f"Non-spatial axes specified: {non_spatial_axes}")

        # validate the axes
        if self.min_coordinate.shape != self.max_coordinate.shape:
            raise ValueError("The `min_coordinate` and `max_coordinate` must have the same shape.")

        n_axes_coordinate = len(self.min_coordinate) if self.min_coordinate.ndim == 1 else self.min_coordinate.shape[1]

        if len(self.axes) != n_axes_coordinate:
            raise ValueError("The number of axes must match the number of coordinates.")

        # validate the coordinates
        if np.any(self.min_coordinate > self.max_coordinate):
            raise ValueError("The minimum coordinate must be less than the maximum coordinate.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_coordinate_system": self.target_coordinate_system,
            "axes": self.axes,
            "min_coordinate": self.min_coordinate,
            "max_coordinate": self.max_coordinate,
        }


@docstring_parameter(min_coordinate_docs=MIN_COORDINATE_DOCS, max_coordinate_docs=MAX_COORDINATE_DOCS)
def _bounding_box_mask_points(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
) -> list[ArrayLike]:
    """Compute a mask that is true for the points inside axis-aligned bounding boxes.

    Parameters
    ----------
    points
        The points element to perform the query on.
    axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        PLACEHOLDER
        The upper left hand corners of the bounding boxes (i.e., minimum coordinates along all dimensions).
        Shape: (n_boxes, n_axes) or (n_axes,) for a single box.
    {min_coordinate_docs}
    max_coordinate
        The lower right hand corners of the bounding boxes (i.e., the maximum coordinates along all dimensions).
        Shape: (n_boxes, n_axes) or (n_axes,) for a single box.
    {max_coordinate_docs}

    Returns
    -------
    The masks for the points inside the bounding boxes.
    """
    element_axes = get_axes_names(points)

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # Ensure min_coordinate and max_coordinate are 2D arrays
    min_coordinate = min_coordinate[np.newaxis, :] if min_coordinate.ndim == 1 else min_coordinate
    max_coordinate = max_coordinate[np.newaxis, :] if max_coordinate.ndim == 1 else max_coordinate

    n_boxes = min_coordinate.shape[0]
    in_bounding_box_masks = []

    for box in range(n_boxes):
        box_masks = []
        for axis_index, axis_name in enumerate(axes):
            if axis_name not in element_axes:
                continue
            min_value = min_coordinate[box, axis_index]
            max_value = max_coordinate[box, axis_index]
            box_masks.append(points[axis_name].gt(min_value).compute() & points[axis_name].lt(max_value).compute())
        bounding_box_mask = np.stack(box_masks, axis=-1)
        in_bounding_box_masks.append(np.all(bounding_box_mask, axis=1))
    return in_bounding_box_masks


def _dict_query_dispatcher(
    elements: dict[str, SpatialElement], query_function: Callable[[SpatialElement], SpatialElement], **kwargs: Any
) -> dict[str, SpatialElement]:
    from spatialdata.transformations import get_transformation

    queried_elements = {}
    for key, element in elements.items():
        target_coordinate_system = kwargs["target_coordinate_system"]
        d = get_transformation(element, get_all=True)
        assert isinstance(d, dict)
        if target_coordinate_system in d:
            result = query_function(element, **kwargs)
            if result is not None:
                # query returns None if it is empty
                queried_elements[key] = result
    return queried_elements


@docstring_parameter(min_coordinate_docs=MIN_COORDINATE_DOCS, max_coordinate_docs=MAX_COORDINATE_DOCS)
@singledispatch
def bounding_box_query(
    element: SpatialElement | SpatialData,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    return_request_only: bool = False,
    filter_table: bool = True,
    **kwargs: Any,
) -> SpatialElement | SpatialData | None:
    """
    Query a SpatialData object or SpatialElement within a bounding box.

    This function can also be accessed as a method of a `SpatialData` object,
    via `sdata.query.bounding_box(...)`, without specifying `element`.

    Parameters
    ----------
    element
        The SpatialElement or SpatialData object to query.
    axes
        The axes `min_coordinate` and `max_coordinate` refer to.
    min_coordinate
        {min_coordinate_docs}
    max_coordinate
        {max_coordinate_docs}
    target_coordinate_system
        The coordinate system the bounding box is defined in.
    filter_table
        If `True`, the table is filtered to only contain rows that are annotating regions
        contained within the bounding box.
    return_request_only
        If `True`, the function returns the bounding box coordinates in the target coordinate system.
        Only valid with `DataArray` and `DataTree` elements.

    Returns
    -------
    The SpatialData object or SpatialElement containing the requested data.
    Eventual empty Elements are omitted by the SpatialData object.

    Notes
    -----
    If the object has `points` element, depending on the number of points, it MAY suffer from performance issues. Please
    consider filtering the object before calling this function by calling the `subset()` method of `SpatialData`.
    """
    raise RuntimeError("Unsupported type for bounding_box_query: " + str(type(element)) + ".")


@bounding_box_query.register(SpatialData)
def _(
    sdata: SpatialData,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    filter_table: bool = True,
) -> SpatialData:
    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    new_elements = {}
    if sdata.points:
        warnings.warn(
            (
                "The object has `points` element. Depending on the number of points, querying MAY suffer from "
                "performance issues. Please consider filtering the object before calling this function by calling the "
                "`subset()` method of `SpatialData`."
            ),
            UserWarning,
            stacklevel=2,
        )
    for element_type in ["points", "images", "labels", "shapes"]:
        elements = getattr(sdata, element_type)
        queried_elements = _dict_query_dispatcher(
            elements,
            bounding_box_query,
            axes=axes,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=target_coordinate_system,
        )
        new_elements[element_type] = queried_elements

    tables = _get_filtered_or_unfiltered_tables(filter_table, new_elements, sdata)

    return SpatialData(**new_elements, tables=tables, attrs=sdata.attrs)


@bounding_box_query.register(DataArray)
@bounding_box_query.register(DataTree)
def _(
    image: DataArray | DataTree,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    return_request_only: bool = False,
) -> DataArray | DataTree | Mapping[str, slice] | list[DataArray] | list[DataTree] | None:
    """Implement bounding box query for Spatialdata supported DataArray.

    Notes
    -----
    See https://github.com/scverse/spatialdata/pull/151 for a detailed overview of the logic of this code,
    and for the cases the comments refer to.
    """
    from spatialdata._core.query._utils import _create_slices_and_translation, _process_query_result

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    intrinsic_bounding_box_corners, axes = _get_bounding_box_corners_in_intrinsic_coordinates(
        image, axes, min_coordinate, max_coordinate, target_coordinate_system
    )
    if TYPE_CHECKING:
        assert isinstance(intrinsic_bounding_box_corners, DataArray)

    min_values = intrinsic_bounding_box_corners.min(dim="corner")
    max_values = intrinsic_bounding_box_corners.max(dim="corner")

    min_values_np = min_values.data
    max_values_np = max_values.data

    if min_values_np.ndim == 1:
        min_values_np = min_values_np[np.newaxis, :]
        max_values_np = max_values_np[np.newaxis, :]

    slices, translation_vectors = _create_slices_and_translation(min_values_np, max_values_np)

    if min_values.ndim == 2:  # Multiple boxes
        selection: list[dict[str, Any]] | dict[str, Any] = [
            {
                axis: slice(slices[box_idx, axis_idx, 0], slices[box_idx, axis_idx, 1])
                for axis_idx, axis in enumerate(axes)
            }
            for box_idx in range(len(min_values_np))
        ]
        translation_vectors = translation_vectors.tolist()
    else:  # Single box
        selection = {axis: slice(slices[0, axis_idx, 0], slices[0, axis_idx, 1]) for axis_idx, axis in enumerate(axes)}
        translation_vectors = translation_vectors[0].tolist()

    if return_request_only:
        return selection

    # query the data
    query_result: DataArray | DataTree | list[DataArray] | list[DataTree] | None = (
        image.sel(selection) if isinstance(selection, dict) else [image.sel(sel) for sel in selection]
    )

    if isinstance(query_result, list):
        processed_results = []
        for result, translation_vector in zip(query_result, translation_vectors, strict=True):
            processed_result = _process_query_result(result, translation_vector, axes)
            if processed_result is not None:
                processed_results.append(processed_result)
        query_result = processed_results if processed_results else None
    else:
        query_result = _process_query_result(query_result, translation_vectors, axes)
    return query_result


@bounding_box_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> DaskDataFrame | list[DaskDataFrame] | None:
    from spatialdata import transform
    from spatialdata.transformations import get_transformation

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # Ensure min_coordinate and max_coordinate are 2D arrays
    min_coordinate = min_coordinate[np.newaxis, :] if min_coordinate.ndim == 1 else min_coordinate
    max_coordinate = max_coordinate[np.newaxis, :] if max_coordinate.ndim == 1 else max_coordinate

    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    # get the four corners of the bounding box (2D case), or the 8 corners of the "3D bounding box" (3D case)
    (intrinsic_bounding_box_corners, intrinsic_axes) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=points,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
    )
    min_coordinate_intrinsic = intrinsic_bounding_box_corners.min(dim="corner")
    max_coordinate_intrinsic = intrinsic_bounding_box_corners.max(dim="corner")

    min_coordinate_intrinsic = min_coordinate_intrinsic.data
    max_coordinate_intrinsic = max_coordinate_intrinsic.data

    # get the points in the intrinsic coordinate bounding box
    in_intrinsic_bounding_box = _bounding_box_mask_points(
        points=points,
        axes=intrinsic_axes,
        min_coordinate=min_coordinate_intrinsic,
        max_coordinate=max_coordinate_intrinsic,
    )

    if not (len_df := len(in_intrinsic_bounding_box)) == (len_bb := len(min_coordinate)):
        raise ValueError(
            f"Length of list of dataframes `{len_df}` is not equal to the number of bounding boxes axes `{len_bb}`."
        )
    points_in_intrinsic_bounding_box: list[DaskDataFrame | None] = []
    points_pd = points.compute()
    attrs = points.attrs.copy()
    for mask_np in in_intrinsic_bounding_box:
        if mask_np.sum() == 0:
            points_in_intrinsic_bounding_box.append(None)
        else:
            # TODO there is a problem when mixing dask dataframe graph with dask array graph. Need to compute for now.
            # we can't compute either mask or points as when we calculate either one of them
            # test_query_points_multiple_partitions will fail as the mask will be used to index each partition.
            # However, if we compute and then create the dask array again we get the mixed dask graph problem.
            filtered_pd = points_pd[mask_np]
            points_filtered = dd.from_pandas(filtered_pd, npartitions=points.npartitions)
            points_filtered.attrs.update(attrs)
            points_in_intrinsic_bounding_box.append(points_filtered)
    if len(points_in_intrinsic_bounding_box) == 0:
        return None

    # assert that the number of queried points is correct
    assert len(points_in_intrinsic_bounding_box) == len(min_coordinate)

    # # we have to reset the index since we have subset
    # # https://stackoverflow.com/questions/61395351/how-to-reset-index-on-concatenated-dataframe-in-dask
    # points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.assign(idx=1)
    # points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.set_index(
    #     points_in_intrinsic_bounding_box.idx.cumsum() - 1
    # )
    # points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.map_partitions(
    #     lambda df: df.rename(index={"idx": None})
    # )
    # points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.drop(columns=["idx"])

    # transform the element to the query coordinate system
    output: list[DaskDataFrame | None] = []
    for p, min_c, max_c in zip(points_in_intrinsic_bounding_box, min_coordinate, max_coordinate, strict=True):
        if p is None:
            output.append(None)
        else:
            points_query_coordinate_system = transform(
                p, to_coordinate_system=target_coordinate_system, maintain_positioning=False
            )

            # get a mask for the points in the bounding box
            bounding_box_mask = _bounding_box_mask_points(
                points=points_query_coordinate_system,
                axes=axes,
                min_coordinate=min_c,  # type: ignore[arg-type]
                max_coordinate=max_c,  # type: ignore[arg-type]
            )
            if len(bounding_box_mask) != 1:
                raise ValueError(f"Expected a single mask, got {len(bounding_box_mask)} masks. Please report this bug.")
            bounding_box_indices = np.where(bounding_box_mask[0])[0]

            if len(bounding_box_indices) == 0:
                output.append(None)
            else:
                points_df = p.compute().iloc[bounding_box_indices]
                old_transformations = get_transformation(p, get_all=True)
                assert isinstance(old_transformations, dict)
                feature_key = p.attrs.get(ATTRS_KEY, {}).get(PointsModel.FEATURE_KEY)

                output.append(
                    PointsModel.parse(
                        dd.from_pandas(points_df, npartitions=1),
                        transformations=old_transformations.copy(),
                        feature_key=feature_key,
                    )
                )
    if len(output) == 0:
        return None
    if len(output) == 1:
        return output[0]
    return output


@bounding_box_query.register(GeoDataFrame)
def _(
    polygons: GeoDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> GeoDataFrame | list[GeoDataFrame] | None:
    from spatialdata.transformations import get_transformation

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    # get the four corners of the bounding box
    (intrinsic_bounding_box_corners, _) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=polygons,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
    )

    # Create a list of Polygons for each bounding box
    old_transformations = get_transformation(polygons, get_all=True)
    assert isinstance(old_transformations, dict)

    queried_polygons = []
    intrinsic_bounding_box_corners = (
        intrinsic_bounding_box_corners.expand_dims(dim="box")
        if "box" not in intrinsic_bounding_box_corners.dims
        else intrinsic_bounding_box_corners
    )
    for box_corners in intrinsic_bounding_box_corners:
        bounding_box_non_axes_aligned = Polygon(box_corners.data)
        indices = polygons.geometry.intersects(bounding_box_non_axes_aligned)
        queried = polygons[indices]
        if len(queried) == 0:
            queried_polygon = None
        else:
            del queried.attrs[ShapesModel.TRANSFORM_KEY]
            queried_polygon = ShapesModel.parse(queried, transformations=old_transformations.copy())
        queried_polygons.append(queried_polygon)
    if len(queried_polygons) == 0:
        return None
    if len(queried_polygons) == 1:
        return queried_polygons[0]
    return queried_polygons


@singledispatch
def polygon_query(
    element: SpatialElement | SpatialData,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    filter_table: bool = True,
    clip: bool = False,
) -> SpatialElement | SpatialData | None:
    """
    Query a SpatialData object or a SpatialElement by a polygon or multipolygon.

    This function can also be accessed as a method of a `SpatialData` object,
    via `sdata.query.polygon(...)`, without specifying `element`.

    Parameters
    ----------
    element
        The SpatialElement or SpatialData object to query.
    polygon
        The polygon/multipolygon to query by.
    target_coordinate_system
        The coordinate system of the polygon/multipolygon.
    filter_table
        Specifies whether to filter the tables to only include tables that annotate elements in the retrieved
        SpatialData object of the query.
    clip
        If `True`, the shapes are clipped to the polygon. This behavior is implemented only when querying
        polygons/multipolygons or circles, and it is ignored for other types of elements (images, labels, points).
        Importantly, when clipping is enabled, the circles will be converted to polygons before the clipping. This may
        affect downstream operations that rely on the circle radius or on performance, so it is recommended to disable
        clipping when querying circles or when querying a `SpatialData` object that contains circles.

    Returns
    -------
    The queried SpatialData object or SpatialElement containing the requested data.
    Eventual empty Elements are omitted by the SpatialData object.

    Examples
    --------
    Here is an example for multipolygon use case. If you have a sequence of polygons/multipolygons, in particular a
    GeoDataFrame, and you want to query the data that belongs to any one of these shapes, you can call this function
    to the multipolygon obtained by merging all the polygons. To merge you can use a unary union.
    """
    raise RuntimeError("Unsupported type for polygon_query: " + str(type(element)) + ".")


@polygon_query.register(SpatialData)
def _(
    sdata: SpatialData,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    filter_table: bool = True,
    clip: bool = False,
) -> SpatialData:
    new_elements = {}
    for element_type in ["points", "images", "labels", "shapes"]:
        elements = getattr(sdata, element_type)
        queried_elements = _dict_query_dispatcher(
            elements,
            polygon_query,
            polygon=polygon,
            target_coordinate_system=target_coordinate_system,
            clip=clip,
        )
        new_elements[element_type] = queried_elements

    tables = _get_filtered_or_unfiltered_tables(filter_table, new_elements, sdata)

    return SpatialData(**new_elements, tables=tables, attrs=sdata.attrs)


@polygon_query.register(DataArray)
@polygon_query.register(DataTree)
def _(
    image: DataArray | DataTree,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    return_request_only: bool = False,
    **kwargs: Any,
) -> DataArray | DataTree | None:
    gdf = GeoDataFrame(geometry=[polygon])
    min_x, min_y, max_x, max_y = gdf.bounds.values.flatten().tolist()
    return bounding_box_query(
        image,
        min_coordinate=[min_x, min_y],
        max_coordinate=[max_x, max_y],
        axes=("x", "y"),
        target_coordinate_system=target_coordinate_system,
        return_request_only=return_request_only,
    )


@polygon_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    **kwargs: Any,
) -> DaskDataFrame | None:
    from spatialdata.transformations import get_transformation, set_transformation

    polygon_gdf = _get_polygon_in_intrinsic_coordinates(points, target_coordinate_system, polygon)

    points_gdf = points_dask_dataframe_to_geopandas(points, suppress_z_warning=True)
    joined = polygon_gdf.sjoin(points_gdf)
    if len(joined) == 0:
        return None
    assert len(joined.index.unique()) == 1
    queried_points = points_gdf.loc[joined["index_right"]]
    ddf = points_geopandas_to_dask_dataframe(queried_points, suppress_z_warning=True)
    transformation = get_transformation(points, target_coordinate_system)
    feature_key = points.attrs.get(ATTRS_KEY, {}).get(PointsModel.FEATURE_KEY)
    if "z" in ddf.columns:
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y", "z": "z"}, feature_key=feature_key)
    else:
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y"}, feature_key=feature_key)
    set_transformation(ddf, transformation, target_coordinate_system)
    t = get_transformation(ddf, get_all=True)
    assert isinstance(t, dict)
    set_transformation(ddf, t.copy(), set_all=True)
    return ddf


@polygon_query.register(GeoDataFrame)
def _(
    element: GeoDataFrame,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    clip: bool = False,
    **kwargs: Any,
) -> GeoDataFrame | None:
    from spatialdata.transformations import get_transformation, set_transformation

    polygon_gdf = _get_polygon_in_intrinsic_coordinates(element, target_coordinate_system, polygon)
    polygon = polygon_gdf["geometry"].iloc[0]

    buffered = to_polygons(element) if ShapesModel.RADIUS_KEY in element.columns else element

    OLD_INDEX = "__old_index"
    if OLD_INDEX in buffered.columns:
        assert np.all(element[OLD_INDEX] == buffered.index)
    else:
        buffered[OLD_INDEX] = buffered.index
    indices = buffered.geometry.apply(lambda x: x.intersects(polygon))
    if np.sum(indices) == 0:
        return None
    queried_shapes = element[indices]
    queried_shapes.index = buffered[indices][OLD_INDEX]
    queried_shapes.index.name = None

    if clip:
        if isinstance(element.geometry.iloc[0], Point):
            queried_shapes = buffered[indices]
            queried_shapes.index = buffered[indices][OLD_INDEX]
            queried_shapes.index.name = None
        queried_shapes = queried_shapes.clip(polygon_gdf, keep_geom_type=True)

    del buffered[OLD_INDEX]
    if OLD_INDEX in queried_shapes.columns:
        del queried_shapes[OLD_INDEX]

    transformation = get_transformation(buffered, target_coordinate_system)
    queried_shapes = ShapesModel.parse(queried_shapes)
    set_transformation(queried_shapes, transformation, target_coordinate_system)
    t = get_transformation(queried_shapes, get_all=True)
    assert isinstance(t, dict)
    set_transformation(queried_shapes, t.copy(), set_all=True)
    return queried_shapes
