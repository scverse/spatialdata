from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable

import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from datatree import DataTree
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from shapely.geometry import MultiPolygon, Polygon
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.query._utils import (
    _get_filtered_or_unfiltered_tables,
    circles_to_polygons,
    get_bounding_box_corners,
)
from spatialdata._core.spatialdata import SpatialData
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
from spatialdata.transformations._utils import compute_coordinates
from spatialdata.transformations.operations import set_transformation
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Sequence,
    Translation,
    _get_affine_for_element,
)


def _get_bounding_box_corners_in_intrinsic_coordinates(
    element: SpatialElement,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Get all corners of a bounding box in the intrinsic coordinates of an element.

    Parameters
    ----------
    element
        The SpatialElement to get the intrinsic coordinate system from.
    axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions
    target_coordinate_system
        The coordinate system the bounding box is defined in.

    Returns
    -------
    All the corners of the bounding box in the intrinsic coordinate system of the element. The shape
    is (2, 4) when axes has 2 spatial dimensions, and (2, 8) when axes has 3 spatial dimensions.

    The axes of the intrinsic coordinate system.
    """
    from spatialdata.transformations import get_transformation

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    # get the transformation from the element's intrinsic coordinate system
    # to the query coordinate space
    transform_to_query_space = get_transformation(element, to_coordinate_system=target_coordinate_system)
    m_without_c, input_axes_without_c, output_axes_without_c = _get_axes_of_tranformation(
        element, target_coordinate_system
    )
    axes, min_coordinate, max_coordinate = _adjust_bounding_box_to_real_axes(
        axes, min_coordinate, max_coordinate, output_axes_without_c
    )

    # get the coordinates of the bounding box corners
    bounding_box_corners = get_bounding_box_corners(
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes,
    ).data

    # transform the coordinates to the intrinsic coordinate system
    intrinsic_axes = get_axes_names(element)
    transform_to_intrinsic = transform_to_query_space.inverse().to_affine_matrix(  # type: ignore[union-attr]
        input_axes=axes, output_axes=intrinsic_axes
    )
    rotation_matrix = transform_to_intrinsic[0:-1, 0:-1]
    translation = transform_to_intrinsic[0:-1, -1]

    intrinsic_bounding_box_corners = bounding_box_corners @ rotation_matrix.T + translation

    return intrinsic_bounding_box_corners, intrinsic_axes


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
    if set(axes_bb) != set(axes_out_without_c):
        axes_only_in_bb = set(axes_bb) - set(axes_out_without_c)
        axes_only_in_output = set(axes_out_without_c) - set(axes_bb)

        # let's remove from the bounding box whose axes that are not in the output axes (e.g. querying 2D points with a
        # 3D bounding box)
        indices_to_remove_from_bb = [axes_bb.index(ax) for ax in axes_only_in_bb]
        axes_bb = tuple(ax for ax in axes_bb if ax not in axes_only_in_bb)
        min_coordinate = np.delete(min_coordinate, indices_to_remove_from_bb)
        max_coordinate = np.delete(max_coordinate, indices_to_remove_from_bb)

        # if there are axes in the output axes that are not in the bounding box, we need to add them to the bounding box
        # with a range that includes everything (e.g. querying 3D points with a 2D bounding box)
        for ax in axes_only_in_output:
            axes_bb = axes_bb + (ax,)
            M = np.finfo(np.float32).max - 1
            min_coordinate = np.append(min_coordinate, -M)
            max_coordinate = np.append(max_coordinate, M)
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


@dataclass(frozen=True)
class BoundingBoxRequest(BaseSpatialRequest):
    """Query with an axis-aligned bounding box.

    Attributes
    ----------
    axes
        The axes the coordinates are expressed in.
    min_coordinate
        The coordinate of the lower left hand corner (i.e., minimum values)
        of the bounding box.
    max_coordinate
        The coordinate of the upper right hand corner (i.e., maximum values)
        of the bounding box
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
        if len(self.axes) != len(self.min_coordinate) or len(self.axes) != len(self.max_coordinate):
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


def _bounding_box_mask_points(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
) -> da.Array:
    """Compute a mask that is true for the points inside an axis-aligned bounding box.

    Parameters
    ----------
    points
        The points element to perform the query on.
    axes
        The axes that min_coordinate and max_coordinate refer to.
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates along all dimensions).

    Returns
    -------
    The mask for the points inside the bounding box.
    """
    element_axes = get_axes_names(points)
    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)
    in_bounding_box_masks = []
    for axis_index, axis_name in enumerate(axes):
        if axis_name not in element_axes:
            continue
        min_value = min_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].gt(min_value).to_dask_array(lengths=True))
    for axis_index, axis_name in enumerate(axes):
        if axis_name not in element_axes:
            continue
        max_value = max_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].lt(max_value).to_dask_array(lengths=True))
    in_bounding_box_masks = da.stack(in_bounding_box_masks, axis=-1)
    return da.all(in_bounding_box_masks, axis=1)


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


@singledispatch
def bounding_box_query(
    element: SpatialElement | SpatialData,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
    filter_table: bool = True,
    **kwargs: Any,
) -> SpatialElement | SpatialData | None:
    """
    Query a SpatialData object or SpatialElement within a bounding box.

    Parameters
    ----------
    axes
        The axes `min_coordinate` and `max_coordinate` refer to.
    min_coordinate
        The minimum coordinates of the bounding box.
    max_coordinate
        The maximum coordinates of the bounding box.
    target_coordinate_system
        The coordinate system the bounding box is defined in.
    filter_table
        If `True`, the table is filtered to only contain rows that are annotating regions
        contained within the bounding box.

    Returns
    -------
    The SpatialData object or SpatialElement containing the requested data.
    Eventual empty Elements are omitted by the SpatialData object.
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

    return SpatialData(**new_elements, tables=tables)


@bounding_box_query.register(SpatialImage)
@bounding_box_query.register(MultiscaleSpatialImage)
def _(
    image: SpatialImage | MultiscaleSpatialImage,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> SpatialImage | MultiscaleSpatialImage | None:
    """Implement bounding box query for SpatialImage.

    Notes
    -----
    See https://github.com/scverse/spatialdata/pull/151 for a detailed overview of the logic of this code,
    and for the cases the comments refer to.
    """
    from spatialdata.transformations import get_transformation, set_transformation

    min_coordinate = _parse_list_into_array(min_coordinate)
    max_coordinate = _parse_list_into_array(max_coordinate)

    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    # compute the output axes of the transformation, remove c from input and output axes, return the matrix without c
    # and then build an affine transformation from that
    m_without_c, input_axes_without_c, output_axes_without_c = _get_axes_of_tranformation(
        image, target_coordinate_system
    )
    spatial_transform = Affine(m_without_c, input_axes=input_axes_without_c, output_axes=output_axes_without_c)

    # we identified 5 cases (see the responsible function for details), cases 1 and 5 correspond to invertible
    # transformations; we focus on them
    m_without_c_linear = m_without_c[:-1, :-1]
    case = _get_case_of_bounding_box_query(m_without_c_linear, input_axes_without_c, output_axes_without_c)
    assert case in [1, 5]

    # adjust the bounding box to the real axes, dropping or adding eventually mismatching axes; the order of the axes is
    # not adjusted
    axes, min_coordinate, max_coordinate = _adjust_bounding_box_to_real_axes(
        axes, min_coordinate, max_coordinate, output_axes_without_c
    )
    assert set(axes) == set(output_axes_without_c)

    # since the order of the axes is arbitrary, let's adjust the affine transformation without c to match those axes
    spatial_transform_bb_axes = Affine(
        spatial_transform.to_affine_matrix(input_axes=input_axes_without_c, output_axes=axes),
        input_axes=input_axes_without_c,
        output_axes=axes,
    )

    # let's get the bounding box corners and inverse transform then to the intrinsic coordinate system; since we are
    # in case 1 or 5, the transformation is invertible
    bounding_box_corners = get_bounding_box_corners(
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes,
    )
    inverse = spatial_transform_bb_axes.inverse()
    assert isinstance(inverse, Affine)
    rotation_matrix = inverse.matrix[0:-1, 0:-1]
    translation = inverse.matrix[0:-1, -1]
    intrinsic_bounding_box_corners = DataArray(
        bounding_box_corners.data @ rotation_matrix.T + translation,
        coords={"corner": range(len(bounding_box_corners)), "axis": list(inverse.output_axes)},
    )

    # build the request: now that we have the bounding box corners in the intrinsic coordinate system, we can use them
    # to build the request to query the raster data using the xarray APIs
    selection = {}
    translation_vector = []
    for axis_name in axes:
        # get the min value along the axis
        min_value = intrinsic_bounding_box_corners.sel(axis=axis_name).min().item()

        # get max value, slices are open half interval
        max_value = intrinsic_bounding_box_corners.sel(axis=axis_name).max().item()

        # add the
        selection[axis_name] = slice(min_value, max_value)

        if min_value > 0:
            translation_vector.append(np.ceil(min_value).item())
        else:
            translation_vector.append(0)

    # query the data
    query_result = image.sel(selection)
    if isinstance(image, SpatialImage):
        if 0 in query_result.shape:
            return None
        assert isinstance(query_result, SpatialImage)
        # rechunk the data to avoid irregular chunks
        image = image.chunk("auto")
    else:
        assert isinstance(image, MultiscaleSpatialImage)
        assert isinstance(query_result, DataTree)
        # we need to convert query_result it to MultiscaleSpatialImage, dropping eventual collapses scales (or even
        # the whole object if the first scale is collapsed)
        d = {}
        for k, data_tree in query_result.items():
            v = data_tree.values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            if 0 in xdata.shape:
                if k == "scale0":
                    return None
            else:
                d[k] = xdata
        # the list of scales may not be contiguous when the data has small shape (for instance with yx = 22 and
        # rotations we may end up having scale0 and scale2 but not scale1. Practically this may occur in torch tiler if
        # the tiles are request to be too small).
        # Here we remove scales after we found a scale missing
        scales_to_keep = []
        for i, scale_name in enumerate(d.keys()):
            if scale_name == f"scale{i}":
                scales_to_keep.append(scale_name)
            else:
                break
        # case in which scale0 is not present but other scales are
        if len(scales_to_keep) == 0:
            return None
        d = {k: d[k] for k in scales_to_keep}

        query_result = MultiscaleSpatialImage.from_dict(d)
        # rechunk the data to avoid irregular chunks
        for scale in query_result:
            query_result[scale]["image"] = query_result[scale]["image"].chunk("auto")
    query_result = compute_coordinates(query_result)

    # the bounding box, mapped back to the intrinsic coordinate system is a set of points. The bounding box of these
    # points is likely starting away from the origin (this is described by translation_vector), so we need to prepend
    # this translation to every transformation in the new queries elements (unless the translation_vector is zero,
    # in that case the translation is not needed)
    if not np.allclose(np.array(translation_vector), 0):
        translation_transform = Translation(translation=translation_vector, axes=axes)

        transformations = get_transformation(query_result, get_all=True)
        assert isinstance(transformations, dict)

        new_transformations = {}
        for coordinate_system, initial_transform in transformations.items():
            new_transformation: BaseTransformation = Sequence(
                [translation_transform, initial_transform],
            )
            new_transformations[coordinate_system] = new_transformation
        set_transformation(query_result, new_transformations, set_all=True)
    # let's make a copy of the transformations so that we don't modify the original object
    t = get_transformation(query_result, get_all=True)
    assert isinstance(t, dict)
    set_transformation(query_result, t.copy(), set_all=True)
    return query_result


@bounding_box_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> DaskDataFrame | None:
    from spatialdata import transform
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

    # get the four corners of the bounding box (2D case), or the 8 corners of the "3D bounding box" (3D case)
    (intrinsic_bounding_box_corners, intrinsic_axes) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=points,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
    )
    min_coordinate_intrinsic = intrinsic_bounding_box_corners.min(axis=0)
    max_coordinate_intrinsic = intrinsic_bounding_box_corners.max(axis=0)

    # get the points in the intrinsic coordinate bounding box
    in_intrinsic_bounding_box = _bounding_box_mask_points(
        points=points,
        axes=intrinsic_axes,
        min_coordinate=min_coordinate_intrinsic,
        max_coordinate=max_coordinate_intrinsic,
    )
    # if there aren't any points, just return
    if in_intrinsic_bounding_box.sum() == 0:
        return None
    points_in_intrinsic_bounding_box = points.loc[in_intrinsic_bounding_box]

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
    points_query_coordinate_system = transform(
        points_in_intrinsic_bounding_box, to_coordinate_system=target_coordinate_system, maintain_positioning=False
    )  # type: ignore[union-attr]

    # get a mask for the points in the bounding box
    bounding_box_mask = _bounding_box_mask_points(
        points=points_query_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )
    bounding_box_indices = np.where(bounding_box_mask.compute())[0]
    if len(bounding_box_indices) == 0:
        return None
    points_df = points_in_intrinsic_bounding_box.compute().iloc[bounding_box_indices]
    old_transformations = get_transformation(points, get_all=True)
    assert isinstance(old_transformations, dict)
    # an alternative approach is to query for each partition in parallel
    return PointsModel.parse(dd.from_pandas(points_df, npartitions=1), transformations=old_transformations.copy())


@bounding_box_query.register(GeoDataFrame)
def _(
    polygons: GeoDataFrame,
    axes: tuple[str, ...],
    min_coordinate: list[Number] | ArrayLike,
    max_coordinate: list[Number] | ArrayLike,
    target_coordinate_system: str,
) -> GeoDataFrame | None:
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
    (intrinsic_bounding_box_corners, intrinsic_axes) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=polygons,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
    )

    bounding_box_non_axes_aligned = Polygon(intrinsic_bounding_box_corners)
    indices = polygons.geometry.intersects(bounding_box_non_axes_aligned)
    queried = polygons[indices]
    if len(queried) == 0:
        return None
    old_transformations = get_transformation(polygons, get_all=True)
    assert isinstance(old_transformations, dict)
    del queried.attrs[ShapesModel.TRANSFORM_KEY]
    return ShapesModel.parse(queried, transformations=old_transformations.copy())


# TODO: we can replace the manually triggered deprecation warning heres with the decorator from Wouter
def _check_deprecated_kwargs(kwargs: dict[str, Any]) -> None:
    deprecated_args = ["shapes", "points", "images", "labels"]
    for arg in deprecated_args:
        if arg in kwargs and kwargs[arg] is False:
            warnings.warn(
                f"The '{arg}' argument is deprecated and will be removed in one of the next following releases. Please "
                f"filter the SpatialData object before calling this function.",
                DeprecationWarning,
                stacklevel=2,
            )


@singledispatch
def polygon_query(
    element: SpatialElement | SpatialData,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    filter_table: bool = True,
    shapes: bool = True,
    points: bool = True,
    images: bool = True,
    labels: bool = True,
) -> SpatialElement | SpatialData | None:
    """
    Query a SpatialData object or a SpatialElement by a polygon or multipolygon.

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
    shapes [Deprecated]
        This argument is now ignored and will be removed. Please filter the SpatialData object before calling this
        function.
    points [Deprecated]
        This argument is now ignored and will be removed. Please filter the SpatialData object before calling this
        function.
    images [Deprecated]
        This argument is now ignored and will be removed. Please filter the SpatialData object before calling this
        function.
    labels [Deprecated]
        This argument is now ignored and will be removed. Please filter the SpatialData object before calling this
        function.

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
    shapes: bool = True,
    points: bool = True,
    images: bool = True,
    labels: bool = True,
) -> SpatialData:

    _check_deprecated_kwargs({"shapes": shapes, "points": points, "images": images, "labels": labels})
    new_elements = {}
    for element_type in ["points", "images", "labels", "shapes"]:
        elements = getattr(sdata, element_type)
        queried_elements = _dict_query_dispatcher(
            elements,
            polygon_query,
            polygon=polygon,
            target_coordinate_system=target_coordinate_system,
        )
        new_elements[element_type] = queried_elements

    tables = _get_filtered_or_unfiltered_tables(filter_table, new_elements, sdata)

    return SpatialData(**new_elements, tables=tables)


@polygon_query.register(SpatialImage)
@polygon_query.register(MultiscaleSpatialImage)
def _(
    image: SpatialImage | MultiscaleSpatialImage,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    **kwargs: Any,
) -> SpatialImage | MultiscaleSpatialImage | None:
    _check_deprecated_kwargs(kwargs)
    gdf = GeoDataFrame(geometry=[polygon])
    min_x, min_y, max_x, max_y = gdf.bounds.values.flatten().tolist()
    return bounding_box_query(
        image,
        min_coordinate=[min_x, min_y],
        max_coordinate=[max_x, max_y],
        axes=("x", "y"),
        target_coordinate_system=target_coordinate_system,
    )


@polygon_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    polygon: Polygon | MultiPolygon,
    target_coordinate_system: str,
    **kwargs: Any,
) -> DaskDataFrame | None:
    from spatialdata.transformations import get_transformation, set_transformation

    _check_deprecated_kwargs(kwargs)
    polygon_gdf = _get_polygon_in_intrinsic_coordinates(points, target_coordinate_system, polygon)

    points_gdf = points_dask_dataframe_to_geopandas(points, suppress_z_warning=True)
    joined = polygon_gdf.sjoin(points_gdf)
    if len(joined) == 0:
        return None
    assert len(joined.index.unique()) == 1
    queried_points = points_gdf.loc[joined["index_right"]]
    ddf = points_geopandas_to_dask_dataframe(queried_points, suppress_z_warning=True)
    transformation = get_transformation(points, target_coordinate_system)
    if "z" in ddf.columns:
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y", "z": "z"})
    else:
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y"})
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
    **kwargs: Any,
) -> GeoDataFrame | None:
    from spatialdata.transformations import get_transformation, set_transformation

    _check_deprecated_kwargs(kwargs)
    polygon_gdf = _get_polygon_in_intrinsic_coordinates(element, target_coordinate_system, polygon)
    polygon = polygon_gdf["geometry"].iloc[0]

    buffered = circles_to_polygons(element) if ShapesModel.RADIUS_KEY in element.columns else element

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
