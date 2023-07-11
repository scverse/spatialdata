from __future__ import annotations

from collections import defaultdict
from functools import singledispatch

import numpy as np
import pandas as pd
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage

from spatialdata._core.operations.transform import transform
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import PointsModel
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import (
    BaseTransformation,
)

BoundingBoxDescription = tuple[ArrayLike, ArrayLike, tuple[str, ...]]


# def _get_coordinate_system_mapping(sdata: SpatialData) -> dict[str, list[str]]:
#     coordsys_keys = sdata.coordinate_systems
#     image_keys = [] if sdata.images is None else sdata.images.keys()
#     label_keys = [] if sdata.labels is None else sdata.labels.keys()
#     shape_keys = [] if sdata.shapes is None else sdata.shapes.keys()
#     point_keys = [] if sdata.points is None else sdata.points.keys()
#
#     mapping: dict[str, list[str]] = {}
#
#     if len(coordsys_keys) < 1:
#         raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")
#
#     for key in coordsys_keys:
#         mapping[key] = []
#
#         for image_key in image_keys:
#             transformations = get_transformation(sdata.images[image_key], get_all=True)
#
#             if key in list(transformations.keys()):
#                 mapping[key].append(image_key)
#
#         for label_key in label_keys:
#             transformations = get_transformation(sdata.labels[label_key], get_all=True)
#
#             if key in list(transformations.keys()):
#                 mapping[key].append(label_key)
#
#         for shape_key in shape_keys:
#             transformations = get_transformation(sdata.shapes[shape_key], get_all=True)
#
#             if key in list(transformations.keys()):
#                 mapping[key].append(shape_key)
#
#         for point_key in point_keys:
#             transformations = get_transformation(sdata.points[point_key], get_all=True)
#
#             if key in list(transformations.keys()):
#                 mapping[key].append(point_key)
#
#     return mapping
#
#
# def _flatten_transformation_sequence(
#     transformation_sequence: list[Sequence],
# ) -> list[Sequence]:
#     if isinstance(transformation_sequence, Sequence):
#         transformations = list(transformation_sequence.transformations)
#         found_bottom_of_tree = False
#         while not found_bottom_of_tree:
#             if all(not isinstance(t, Sequence) for t in transformations):
#                 found_bottom_of_tree = True
#             else:
#                 for idx, t in enumerate(transformations):
#                     if isinstance(t, Sequence):
#                         transformations.pop(idx)
#                         transformations += t.transformations
#
#         return transformations
#
#     if isinstance(transformation_sequence, BaseTransformation):
#         return [transformation_sequence]
#
#     raise TypeError("Parameter 'transformation_sequence' must be a Sequence.")
#
#
# def _get_cs_contents(sdata: SpatialData) -> pd.DataFrame:
#     """Check which coordinate systems contain which elements and return that info."""
#     cs_mapping = _get_coordinate_system_mapping(sdata)
#     content_flags = ["has_images", "has_labels", "has_points", "has_shapes"]
#     cs_contents = pd.DataFrame(columns=["cs"] + content_flags)
#
#     for cs_name, element_ids in cs_mapping.items():
#         # determine if coordinate system has the respective elements
#         cs_has_images = bool(any((e in sdata.images) for e in element_ids))
#         cs_has_labels = bool(any((e in sdata.labels) for e in element_ids))
#         cs_has_points = bool(any((e in sdata.points) for e in element_ids))
#         cs_has_shapes = bool(any((e in sdata.shapes) for e in element_ids))
#
#         cs_contents = pd.concat(
#             [
#                 cs_contents,
#                 pd.DataFrame(
#                     {
#                         "cs": cs_name,
#                         "has_images": [cs_has_images],
#                         "has_labels": [cs_has_labels],
#                         "has_points": [cs_has_points],
#                         "has_shapes": [cs_has_shapes],
#                     }
#                 ),
#             ]
#         )
#
#         cs_contents["has_images"] = cs_contents["has_images"].astype("bool")
#         cs_contents["has_labels"] = cs_contents["has_labels"].astype("bool")
#         cs_contents["has_points"] = cs_contents["has_points"].astype("bool")
#         cs_contents["has_shapes"] = cs_contents["has_shapes"].astype("bool")
#
#     return cs_contents
#
#
# def _get_extent(
#     sdata: SpatialData,
#     coordinate_systems: Sequence[str] | str | None = None,
#     has_images: bool = True,
#     has_labels: bool = True,
#     has_points: bool = True,
#     has_shapes: bool = True,
#     elements: Iterable[Any] | None = None,
#     share_extent: bool = False,
# ) -> dict[str, tuple[int, int, int, int]]:
#     """Return the extent of all elements in their respective coordinate systems.
#
#     Parameters
#     ----------
#     sdata
#         The sd.SpatialData object to retrieve the extent from
#     has_images
#         Flag indicating whether to consider images when calculating the extent
#     has_labels
#         Flag indicating whether to consider labels when calculating the extent
#     has_points
#         Flag indicating whether to consider points when calculating the extent
#     has_shapes
#         Flag indicating whether to consider shapes when calculating the extent
#     elements
#         Optional list of element names to be considered. When None, all are used.
#     share_extent
#         Flag indicating whether to use the same extent for all coordinate systems
#
#     Returns
#     -------
#     A dict of tuples with the shape (xmin, xmax, ymin, ymax). The keys of the
#         dict are the coordinate_system keys.
#
#     """
#     extent: dict[str, dict[str, Sequence[int]]] = {}
#     cs_mapping = _get_coordinate_system_mapping(sdata)
#     cs_contents = _get_cs_contents(sdata)
#
#     if elements is None:  # to shut up ruff
#         elements = []
#
#     if not isinstance(elements, list):
#         raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")
#
#     if coordinate_systems is not None:
#         if isinstance(coordinate_systems, str):
#             coordinate_systems = [coordinate_systems]
#         cs_contents = cs_contents[cs_contents["cs"].isin(coordinate_systems)]
#         cs_mapping = {k: v for k, v in cs_mapping.items() if k in coordinate_systems}
#
#     for cs_name, element_ids in cs_mapping.items():
#         extent[cs_name] = {}
#         if len(elements) > 0:
#             element_ids = [e for e in element_ids if e in elements]
#
#         def _get_extent_after_transformations(element: Any, cs_name: str) -> Sequence[int]:
#             tmp = element.copy()
#             if len(tmp.shape) == 3:
#                 x_idx = 2
#                 y_idx = 1
#             elif len(tmp.shape) == 2:
#                 x_idx = 1
#                 y_idx = 0
#
#             transformations = get_transformation(tmp, to_coordinate_system=cs_name)
#             transformations = _flatten_transformation_sequence(transformations)
#
#             if len(transformations) == 1 and isinstance(transformations[0], Identity):
#                 result = (0, tmp.shape[x_idx], 0, tmp.shape[y_idx])
#
#             else:
#                 origin = {
#                     "x": 0,
#                     "y": 0,
#                 }
#                 for t in transformations:
#                     if isinstance(t, Translation):
#                         # TODO: remove, in get_extent no data operation should be performed
#                         tmp = _translate_image(image=tmp, translation=t)
#
#                         for idx, ax in enumerate(t.axes):
#                             origin["x"] += t.translation[idx] if ax == "x" else 0
#                             origin["y"] += t.translation[idx] if ax == "y" else 0
#
#                     else:
#                         # TODO: remove, in get_extent no data operation should be performed
#                         tmp = transform(tmp, t)
#
#                         if isinstance(t, Scale):
#                             for idx, ax in enumerate(t.axes):
#                                 origin["x"] *= t.scale[idx] if ax == "x" else 1
#                                 origin["y"] *= t.scale[idx] if ax == "y" else 1
#
#                         elif isinstance(t, Affine):
#                             pass
#
#                 result = (origin["x"], tmp.shape[x_idx], origin["y"], tmp.shape[y_idx])
#
#             del tmp
#             return result
#
#         if has_images and cs_contents.query(f"cs == '{cs_name}'")["has_images"][0]:
#             for images_key in sdata.images:
#                 for e_id in element_ids:
#                     if images_key == e_id:
#                         if not isinstance(sdata.images[e_id], MultiscaleSpatialImage):
#                             extent[cs_name][e_id] = _get_extent_after_transformations(sdata.images[e_id], cs_name)
#                         else:
#                             pass
#
#         if has_labels and cs_contents.query(f"cs == '{cs_name}'")["has_labels"][0]:
#             for labels_key in sdata.labels:
#                 for e_id in element_ids:
#                     if labels_key == e_id:
#                         if not isinstance(sdata.labels[e_id], MultiscaleSpatialImage):
#                             extent[cs_name][e_id] = _get_extent_after_transformations(sdata.labels[e_id], cs_name)
#                         else:
#                             pass
#
#         if has_shapes and cs_contents.query(f"cs == '{cs_name}'")["has_shapes"][0]:
#             for shapes_key in sdata.shapes:
#                 for e_id in element_ids:
#                     if shapes_key == e_id:
#
#                         def get_point_bb(
#                             point: Point, radius: int, method: Literal["topleft", "bottomright"], buffer: int = 1
#                         ) -> Point:
#                             x, y = point.coords[0]
#                             if method == "topleft":
#                                 point_bb = Point(x - radius - buffer, y - radius - buffer)
#                             else:
#                                 point_bb = Point(x + radius + buffer, y + radius + buffer)
#
#                             return point_bb
#
#                         y_dims = []
#                         x_dims = []
#
#                         # Split by Point and Polygon:
#                         tmp_points = sdata.shapes[e_id][
#                             sdata.shapes[e_id]["geometry"].apply(lambda geom: geom.geom_type == "Point")
#                         ]
#                         tmp_polygons = sdata.shapes[e_id][
#                             sdata.shapes[e_id]["geometry"].apply(
#                                 lambda geom: geom.geom_type in ["Polygon", "MultiPolygon"]
#                             )
#                         ]
#
#                         if not tmp_points.empty:
#                             tmp_points["point_topleft"] = tmp_points.apply(
#                                 lambda row: get_point_bb(row["geometry"], row["radius"], "topleft"),
#                                 axis=1,
#                             )
#                             tmp_points["point_bottomright"] = tmp_points.apply(
#                                 lambda row: get_point_bb(row["geometry"], row["radius"], "bottomright"),
#                                 axis=1,
#                             )
#                             xmin_tl, ymin_tl, xmax_tl, ymax_tl = tmp_points["point_topleft"].total_bounds
#                             xmin_br, ymin_br, xmax_br, ymax_br = tmp_points["point_bottomright"].total_bounds
#                             y_dims += [min(ymin_tl, ymin_br), max(ymax_tl, ymax_br)]
#                             x_dims += [min(xmin_tl, xmin_br), max(xmax_tl, xmax_br)]
#
#                         if not tmp_polygons.empty:
#                             xmin, ymin, xmax, ymax = tmp_polygons.total_bounds
#                             y_dims += [ymin, ymax]
#                             x_dims += [xmin, xmax]
#
#                         del tmp_points
#                         del tmp_polygons
#
#                         extent[cs_name][e_id] = x_dims + y_dims
#
#                         transformations = get_transformation(sdata.shapes[e_id], to_coordinate_system=cs_name)
#                         transformations = _flatten_transformation_sequence(transformations)
#
#                         for t in transformations:
#                             if isinstance(t, Translation):
#                                 for idx, ax in enumerate(t.axes):
#                                     extent[cs_name][e_id][0] += t.translation[idx] if ax == "x" else 0
#                                     extent[cs_name][e_id][1] += t.translation[idx] if ax == "x" else 0
#                                     extent[cs_name][e_id][2] += t.translation[idx] if ax == "y" else 0
#                                     extent[cs_name][e_id][3] += t.translation[idx] if ax == "y" else 0
#
#                             else:
#                                 if isinstance(t, Scale):
#                                     for idx, ax in enumerate(t.axes):
#                                         extent[cs_name][e_id][1] *= t.scale[idx] if ax == "x" else 1
#                                         extent[cs_name][e_id][3] *= t.scale[idx] if ax == "y" else 1
#
#                                 elif isinstance(t, Affine):
#                                     pass
#
#         if has_points and cs_contents.query(f"cs == '{cs_name}'")["has_points"][0]:
#             for points_key in sdata.points:
#                 for e_id in element_ids:
#                     if points_key == e_id:
#                         tmp = sdata.points[points_key]
#                         xmin = tmp["x"].min().compute()
#                         xmax = tmp["x"].max().compute()
#                         ymin = tmp["y"].min().compute()
#                         ymax = tmp["y"].max().compute()
#                         extent[cs_name][e_id] = [xmin, xmax, ymin, ymax]
#
#     cswise_extent = {}
#     for cs_name, cs_contents in extent.items():
#         if len(cs_contents) > 0:
#             xmin = min([v[0] for v in cs_contents.values()])
#             xmax = max([v[1] for v in cs_contents.values()])
#             ymin = min([v[2] for v in cs_contents.values()])
#             ymax = max([v[3] for v in cs_contents.values()])
#             cswise_extent[cs_name] = (xmin, xmax, ymin, ymax)
#
#     if share_extent:
#         global_extent = {}
#         if len(cs_contents) > 0:
#             xmin = min([v[0] for v in cswise_extent.values()])
#             xmax = max([v[1] for v in cswise_extent.values()])
#             ymin = min([v[2] for v in cswise_extent.values()])
#             ymax = max([v[3] for v in cswise_extent.values()])
#             for cs_name in cswise_extent:
#                 global_extent[cs_name] = (xmin, xmax, ymin, ymax)
#         return global_extent
#
#     return cswise_extent


def _get_extent_of_circles(circles: GeoDataFrame) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of circles.

    Parameters
    ----------
    circles

    Returns
    -------
    The bounding box description.
    """
    assert isinstance(circles.geometry.iloc[0], Point)
    assert "radius" in circles.columns, "Circles must have a 'radius' column."
    axes = get_axes_names(circles)

    centroids = []
    for dim_name in axes:
        centroids.append(getattr(circles["geometry"], dim_name).to_numpy())
    centroids_array = np.column_stack(centroids)
    radius = np.expand_dims(circles["radius"].to_numpy(), axis=1)

    min_coordinates = np.min(centroids_array - radius, axis=0)
    max_coordinates = np.max(centroids_array + radius, axis=0)

    return min_coordinates, max_coordinates, axes


def _get_extent_of_polygons_multipolygons(shapes: GeoDataFrame) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of polygons and/or multipolygons.

    Parameters
    ----------
    shapes

    Returns
    -------
    The bounding box description.
    """
    assert isinstance(shapes.geometry.iloc[0], (Polygon, MultiPolygon))
    axes = get_axes_names(shapes)
    bounds = shapes["geometry"].bounds
    min_coordinates = np.array((bounds["minx"].min(), bounds["miny"].min()))
    max_coordinates = np.array((bounds["maxx"].max(), bounds["maxy"].max()))
    return min_coordinates, max_coordinates, axes


@singledispatch
def get_extent(object: SpatialData | SpatialElement, coordinate_system: str = "global") -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object or a SpatialElement.

    Returns
    -------
    min_coordinate
        The minimum coordinate of the bounding box.
    max_coordinate
        The maximum coordinate of the bounding box.
    axes
        The names of the dimensions of the bounding box
    """
    raise ValueError("The object type is not supported.")


@get_extent.register
def _(e: SpatialData, coordinate_system: str = "global") -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object: the extent of the union of the extents of all its elements.

    Parameters
    ----------
    e
        The SpatialData object.

    Returns
    -------
    The bounding box description.
    """
    new_min_coordinates_dict = defaultdict(list)
    new_max_coordinates_dict = defaultdict(list)
    for element in e._gen_elements_values():
        transformations = get_transformation(element, get_all=True)
        assert isinstance(transformations, dict)
        coordinate_systems = list(transformations.keys())
        if coordinate_system in coordinate_systems:
            min_coordinates, max_coordinates, axes = get_extent(element, coordinate_system=coordinate_system)
            for i, ax in enumerate(axes):
                new_min_coordinates_dict[ax].append(min_coordinates[i])
                new_max_coordinates_dict[ax].append(max_coordinates[i])
    axes = tuple(new_min_coordinates_dict.keys())
    if len(axes) == 0:
        raise ValueError(
            f"The SpatialData object does not contain any element in the coordinate system {coordinate_system!r}, "
            f"please pass a different coordinate system wiht the argument 'coordinate_system'."
        )
    new_min_coordinates = np.array([min(new_min_coordinates_dict[ax]) for ax in axes])
    new_max_coordinates = np.array([max(new_max_coordinates_dict[ax]) for ax in axes])
    return new_min_coordinates, new_max_coordinates, axes


@get_extent.register
def _(e: GeoDataFrame, coordinate_system: str = "global") -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of shapes.

    Parameters
    ----------
    shapes

    Returns
    -------
    The bounding box description.
    """
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    if isinstance(e.geometry.iloc[0], Point):
        assert "radius" in e.columns, "Shapes must have a 'radius' column."
        min_coordinates, max_coordinates, axes = _get_extent_of_circles(e)
    else:
        assert isinstance(e.geometry.iloc[0], (Polygon, MultiPolygon)), "Shapes must be polygons or multipolygons."
        min_coordinates, max_coordinates, axes = _get_extent_of_polygons_multipolygons(e)

    return _compute_extent_in_coordinate_system(
        element=e,
        coordinate_system=coordinate_system,
        min_coordinates=min_coordinates,
        max_coordinates=max_coordinates,
        axes=axes,
    )


@get_extent.register
def _(e: DaskDataFrame, coordinate_system: str = "global") -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    axes = get_axes_names(e)
    min_coordinates = np.array([e[ax].min().compute() for ax in axes])
    max_coordinates = np.array([e[ax].max().compute() for ax in axes])
    return _compute_extent_in_coordinate_system(
        element=e,
        coordinate_system=coordinate_system,
        min_coordinates=min_coordinates,
        max_coordinates=max_coordinates,
        axes=axes,
    )


@get_extent.register
def _(e: SpatialImage, coordinate_system: str = "global") -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    raise NotImplementedError()
    # return _compute_extent_in_coordinate_system(
    #     element=e,
    #     coordinate_system=coordinate_system,
    #     min_coordinates=min_coordinates,
    #     max_coordinates=max_coordinates,
    #     axes=axes,
    # )


@get_extent.register
def _(e: MultiscaleSpatialImage, coordinate_system: str = "global") -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    raise NotImplementedError()
    # return _compute_extent_in_coordinate_system(
    #     element=e,
    #     coordinate_system=coordinate_system,
    #     min_coordinates=min_coordinates,
    #     max_coordinates=max_coordinates,
    #     axes=axes,
    # )


def _check_element_has_coordinate_system(element: SpatialElement, coordinate_system: str) -> None:
    transformations = get_transformation(element, get_all=True)
    assert isinstance(transformations, dict)
    coordinate_systems = list(transformations.keys())
    if coordinate_system not in coordinate_systems:
        raise ValueError(
            f"The element does not contain any coordinate system named {coordinate_system!r}, "
            f"please pass a different coordinate system wiht the argument 'coordinate_system'."
        )


def _compute_extent_in_coordinate_system(
    element: SpatialElement,
    coordinate_system: str,
    min_coordinates: ArrayLike,
    max_coordinates: ArrayLike,
    axes: tuple[str, ...],
) -> BoundingBoxDescription:
    """
    Transform the extent from the intrinsic coordinates of the element to the given coordinate system.

    Parameters
    ----------
    element
        The SpatialElement.
    coordinate_system
        The coordinate system to transform the extent to.
    min_coordinates
        Min coordinates of the extent in the intrinsic coordinates of the element.
    max_coordinates
        Max coordinates of the extent in the intrinsic coordinates of the element.
    axes
        The min and max coordinates refer to.

    Returns
    -------
    The bounding box description in the specified coordinate system.
    """
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    assert isinstance(transformation, BaseTransformation)
    from spatialdata._core.query._utils import get_bounding_box_corners

    corners = get_bounding_box_corners(axes=axes, min_coordinate=min_coordinates, max_coordinate=max_coordinates)
    df = pd.DataFrame(corners.data, columns=corners.axis.data.tolist())
    points = PointsModel.parse(df, coordinates={k: k for k in axes})
    transformed_corners = transform(points, transformation).compute()
    min_coordinates = transformed_corners.min().to_numpy()
    max_coordinates = transformed_corners.max().to_numpy()
    return min_coordinates, max_coordinates, axes
