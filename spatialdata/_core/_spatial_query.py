from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable, Optional, Union

import dask.array as da
import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from datatree import DataTree
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from shapely.geometry import Polygon
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import SpatialData, SpatialElement
from spatialdata._core.core_utils import (
    ValidAxis_t,
    compute_coordinates,
    get_dims,
    get_spatial_axes,
)
from spatialdata._core.transformations import (
    Affine,
    BaseTransformation,
    Sequence,
    Translation,
    _get_affine_for_element,
)
from spatialdata._logging import logger
from spatialdata._types import ArrayLike


def get_bounding_box_corners(min_coordinate: ArrayLike, max_coordinate: ArrayLike, axes: tuple[str, ...]) -> DataArray:
    """From the min and max coordinates of a bounding box, get the coordinates
    of all corners.

    Parameters
    ----------
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions

    Returns
    -------
    (N, D) array of coordinates of the corners. N = 4 for 2D and 8 for 3D.
    """
    if len(min_coordinate) == 2:
        # 2D bounding box
        assert len(axes) == 2
        return DataArray(
            [
                [min_coordinate[0], min_coordinate[1]],
                [min_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], min_coordinate[1]],
            ],
            coords={"corner": range(4), "axis": list(axes)},
        )

    elif len(min_coordinate) == 3:
        # 3D bounding cube
        assert len(axes) == 3
        return DataArray(
            [
                [min_coordinate[0], min_coordinate[1], min_coordinate[2]],
                [min_coordinate[0], min_coordinate[1], max_coordinate[2]],
                [min_coordinate[0], max_coordinate[1], max_coordinate[2]],
                [min_coordinate[0], max_coordinate[1], min_coordinate[2]],
                [max_coordinate[0], min_coordinate[1], min_coordinate[2]],
                [max_coordinate[0], min_coordinate[1], max_coordinate[2]],
                [max_coordinate[0], max_coordinate[1], max_coordinate[2]],
                [max_coordinate[0], max_coordinate[1], min_coordinate[2]],
            ],
            coords={"corner": range(8), "axis": list(axes)},
        )
    else:
        raise ValueError("bounding box must be 2D or 3D")


def _get_bounding_box_corners_in_intrinsic_coordinates(
    element: SpatialElement,
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    axes: tuple[str, ...],
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Get all corners of a bounding box in the intrinsic coordinates of an element.

    Parameters
    ----------
    element
        The SpatialElement to get the intrinsic coordinate system from.
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions
    target_coordinate_system
        The coordinate system the bounding box is defined in.
    axes
        The axes of the coordinate system the bounding box is defined in.

    Returns ------- All the corners of the bounding box in the intrinsic coordinate system of the element. The shape
    is (2, 4) when axes has 2 spatial dimensions, and (2, 8) when axes has 3 spatial dimensions.

    The axes of the intrinsic coordinate system.
    """
    from spatialdata._core._spatialdata_ops import get_transformation

    # get the transformation from the element's intrinsic coordinate system
    # to the query coordinate space
    transform_to_query_space = get_transformation(element, to_coordinate_system=target_coordinate_system)

    # get the coordinates of the bounding box corners
    bounding_box_corners = get_bounding_box_corners(
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes,
    ).data

    # transform the coordinates to the intrinsic coordinate system
    intrinsic_axes = get_dims(element)
    transform_to_intrinsic = transform_to_query_space.inverse().to_affine_matrix(  # type: ignore[union-attr]
        input_axes=axes, output_axes=intrinsic_axes
    )
    rotation_matrix = transform_to_intrinsic[0:-1, 0:-1]
    translation = transform_to_intrinsic[0:-1, -1]

    intrinsic_bounding_box_corners = bounding_box_corners @ rotation_matrix.T + translation

    return intrinsic_bounding_box_corners, intrinsic_axes


@dataclass(frozen=True)
class BaseSpatialRequest:
    """Base class for spatial queries."""

    target_coordinate_system: str
    axes: tuple[ValidAxis_t, ...]

    def __post_init__(self) -> None:
        # validate the axes
        spatial_axes = get_spatial_axes(self.axes)
        non_spatial_axes = set(self.axes) - set(spatial_axes)
        if len(non_spatial_axes) > 0:
            raise ValueError(f"Non-spatial axes specified: {non_spatial_axes}")

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

    min_coordinate: np.ndarray  # type: ignore[type-arg]
    max_coordinate: np.ndarray  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        super().__post_init__()

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
    points: DaskDataFrame, min_coordinate: ArrayLike, max_coordinate: ArrayLike, axes: tuple[str, ...]
) -> da.Array:
    """Compute a mask that is true for the points inside of an axis-aligned bounding box..

    Parameters
    ----------
    points
        The points element to perform the query on.
    min_coordinate
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions
    axes
        The axes for the min/max coordinates.

    Returns
    -------
    The mask for the points inside of the bounding box.
    """

    in_bounding_box_masks = []
    for axis_index, axis_name in enumerate(axes):
        min_value = min_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].gt(min_value).to_dask_array(lengths=True))
    for axis_index, axis_name in enumerate(axes):
        max_value = max_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].lt(max_value).to_dask_array(lengths=True))
    in_bounding_box_masks = da.stack(in_bounding_box_masks, axis=-1)
    return da.all(in_bounding_box_masks, axis=1)


def _dict_query_dispatcher(
    elements: dict[str, SpatialElement], query_function: Callable[[SpatialElement], SpatialElement], **kwargs: Any
) -> dict[str, SpatialElement]:
    queried_elements = {}
    for key, element in elements.items():
        result = query_function(element, **kwargs)
        if result is not None:
            # query returns None if it is empty
            queried_elements[key] = result
    return queried_elements


@singledispatch
def bounding_box_query(
    element: Union[SpatialElement, SpatialData],
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> Optional[Union[SpatialElement, SpatialData]]:
    raise NotImplementedError()


@bounding_box_query.register(SpatialData)
def _(
    sdata: SpatialData,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> SpatialData:
    from spatialdata import SpatialData

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
    return SpatialData(**new_elements, table=sdata.table)


@bounding_box_query.register(SpatialImage)
@bounding_box_query.register(MultiscaleSpatialImage)
def _(
    image: SpatialImage,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> Optional[Union[SpatialImage, MultiscaleSpatialImage]]:
    """

    Parameters
    ----------
    image
    axes
    min_coordinate
    max_coordinate
    target_coordinate_system

    Returns
    -------

    Notes
    _____
    See https://github.com/scverse/spatialdata/pull/151 for a detailed overview of the logic of this code,
    and for the cases the comments refer to.

    """
    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )
    from spatialdata._core._spatialdata_ops import (
        get_transformation,
        set_transformation,
    )

    # get the transformation from the element's intrinsic coordinate system to the query coordinate space
    transform_to_query_space = get_transformation(image, to_coordinate_system=target_coordinate_system)
    assert isinstance(transform_to_query_space, BaseTransformation)
    m = _get_affine_for_element(image, transform_to_query_space)
    input_axes_without_c = tuple([ax for ax in m.input_axes if ax != "c"])
    output_axes_without_c = tuple([ax for ax in m.output_axes if ax != "c"])
    m_without_c = m.to_affine_matrix(input_axes=input_axes_without_c, output_axes=output_axes_without_c)
    m_without_c_linear = m_without_c[:-1, :-1]

    transform_dimension = np.linalg.matrix_rank(m_without_c_linear)
    transform_coordinate_length = len(output_axes_without_c)
    data_dim = len(input_axes_without_c)

    assert data_dim in [2, 3]
    assert transform_dimension in [2, 3]
    assert transform_coordinate_length in [2, 3]
    assert not (data_dim == 2 and transform_dimension == 3)
    assert not (transform_dimension == 3 and transform_coordinate_length == 2)
    # see explanation in https://github.com/scverse/spatialdata/pull/151
    if data_dim == 2 and transform_dimension == 2 and transform_coordinate_length == 2:
        case = 1
    elif data_dim == 2 and transform_dimension == 2 and transform_coordinate_length == 3:
        case = 2
    elif data_dim == 3 and transform_dimension == 2 and transform_coordinate_length == 2:
        case = 3
    elif data_dim == 3 and transform_dimension == 2 and transform_coordinate_length == 3:
        case = 4
    elif data_dim == 3 and transform_dimension == 3 and transform_coordinate_length == 3:
        case = 5
    else:
        raise RuntimeError("This should not happen")

    if case in [3, 4]:
        raise ValueError(
            f"This case is not supported (data with dimension {data_dim} but transformation with rank {transform_dimension}. Please open a GitHub issue if you want to discuss a case."
        )

    if set(axes) != set(output_axes_without_c):
        if set(axes).issubset(output_axes_without_c):
            logger.warning(
                f"The element has axes {output_axes_without_c}, but the query has axes {axes}. Excluding the element "
                f"from the query result. In the future we can add support for this case. If you are interested, "
                f"please open a GitHub issue."
            )
            return None
        else:
            raise ValueError(
                f"Invalid case. The bounding box axes are {axes}, the spatial axes in {target_coordinate_system} are {output_axes_without_c}"
            )
    spatial_transform = Affine(m_without_c, input_axes=input_axes_without_c, output_axes=output_axes_without_c)
    spatial_transform_bb_axes = Affine(
        spatial_transform.to_affine_matrix(input_axes=input_axes_without_c, output_axes=axes),
        input_axes=input_axes_without_c,
        output_axes=axes,
    )
    assert case in [1, 2, 5]
    if case in [1, 5]:
        bounding_box_corners = get_bounding_box_corners(
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            axes=axes,
        )
    else:
        assert case == 2
        # TODO: we need to intersect the plane in the extrinsic coordiante system with the 3D bounding box. The
        #  vertices of this polygons needs to be transformed to the intrinsic coordinate system
        raise NotImplementedError(
            "Case 2 (the transformation is embedding 2D data in the 3D space, is not "
            "implemented yet. Please open a Github issue about this and we will prioritize the "
            "development."
        )
    inverse = spatial_transform_bb_axes.inverse()
    assert isinstance(inverse, Affine)
    rotation_matrix = inverse.matrix[0:-1, 0:-1]
    translation = inverse.matrix[0:-1, -1]

    intrinsic_bounding_box_corners = DataArray(
        bounding_box_corners.data @ rotation_matrix.T + translation,
        coords={"corner": range(len(bounding_box_corners)), "axis": list(inverse.output_axes)},
    )

    # build the request
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

    query_result = image.sel(selection)
    if isinstance(image, SpatialImage):
        if 0 in query_result.shape:
            return None
        assert isinstance(query_result, SpatialImage)
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
        query_result = MultiscaleSpatialImage.from_dict(d)
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
    return query_result


@bounding_box_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> Optional[DaskDataFrame]:
    # for triggering validation
    _ = BoundingBoxRequest(
        target_coordinate_system=target_coordinate_system,
        axes=axes,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )
    from spatialdata._core._spatialdata_ops import get_transformation

    # get the four corners of the bounding box (2D case), or the 8 corners of the "3D bounding box" (3D case)
    (intrinsic_bounding_box_corners, intrinsic_axes) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=points,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
        axes=axes,
    )
    min_coordinate_intrinsic = intrinsic_bounding_box_corners.min(axis=0)
    max_coordinate_intrinsic = intrinsic_bounding_box_corners.max(axis=0)

    # get the points in the intrinsic coordinate bounding box
    in_intrinsic_bounding_box = _bounding_box_mask_points(
        points=points,
        min_coordinate=min_coordinate_intrinsic,
        max_coordinate=max_coordinate_intrinsic,
        axes=intrinsic_axes,
    )
    points_in_intrinsic_bounding_box = points.loc[in_intrinsic_bounding_box]

    if in_intrinsic_bounding_box.sum() == 0:
        # if there aren't any points, just return
        return None

    # we have to reset the index since we have subset
    # https://stackoverflow.com/questions/61395351/how-to-reset-index-on-concatenated-dataframe-in-dask
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.assign(idx=1)
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.set_index(
        points_in_intrinsic_bounding_box.idx.cumsum() - 1
    )
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.map_partitions(
        lambda df: df.rename(index={"idx": None})
    )
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.drop(columns=["idx"])

    # transform the element to the query coordinate system
    transform_to_query_space = get_transformation(points, to_coordinate_system=target_coordinate_system)
    points_query_coordinate_system = transform_to_query_space.transform(points_in_intrinsic_bounding_box)  # type: ignore[union-attr]

    # get a mask for the points in the bounding box
    bounding_box_mask = _bounding_box_mask_points(
        points=points_query_coordinate_system,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes,
    )
    if bounding_box_mask.sum() == 0:
        return None
    else:
        return points_in_intrinsic_bounding_box.loc[bounding_box_mask]


@bounding_box_query.register(GeoDataFrame)
def _(
    polygons: GeoDataFrame,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> Optional[GeoDataFrame]:
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
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
        axes=axes,
    )

    bounding_box_non_axes_aligned = Polygon(intrinsic_bounding_box_corners)
    queried = polygons[polygons.geometry.within(bounding_box_non_axes_aligned)]
    if len(queried) == 0:
        return None
    return queried
