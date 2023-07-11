"""Functions to compute the bounding box describing the extent of a SpatialElement or region."""
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import Point

from spatialdata._core.operations.transform import transform
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import get_axes_names
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Scale,
    Sequence,
    Translation,
)


def _get_extent_of_circles(circles: GeoDataFrame) -> tuple[ArrayLike, ArrayLike, tuple[str, ...]]:
    """Get the coordinates for the corners of the bounding box of that encompasses a set of circles.

    This computes the bounding box around all circles in the element.

    Returns
    -------
    min_coordinate
        The minimum coordinate of the bounding box.
    max_coordinate
        The maximum coordinate of the bounding box.
    """
    assert isinstance(circles.geometry.iloc[0], Point)
    assert "radius" in circles.columns, "Circles must have a 'radius' column."
    circle_dims = get_axes_names(circles)

    centroids = []
    for dim_name in circle_dims:
        centroids.append(getattr(circles["geometry"], dim_name).to_numpy())
    centroids_array = np.column_stack(centroids)
    radius = np.expand_dims(circles["radius"].to_numpy(), axis=1)

    min_coordinates = (centroids_array - radius).astype(float)
    max_coordinates = (centroids_array + radius).astype(float)

    return min_coordinates, max_coordinates, circle_dims


def _get_coordinate_system_mapping(sdata: SpatialData) -> dict[str, list[str]]:
    coordsys_keys = sdata.coordinate_systems
    image_keys = [] if sdata.images is None else sdata.images.keys()
    label_keys = [] if sdata.labels is None else sdata.labels.keys()
    shape_keys = [] if sdata.shapes is None else sdata.shapes.keys()
    point_keys = [] if sdata.points is None else sdata.points.keys()

    mapping: dict[str, list[str]] = {}

    if len(coordsys_keys) < 1:
        raise ValueError("SpatialData object must have at least one coordinate system to generate a mapping.")

    for key in coordsys_keys:
        mapping[key] = []

        for image_key in image_keys:
            transformations = get_transformation(sdata.images[image_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(image_key)

        for label_key in label_keys:
            transformations = get_transformation(sdata.labels[label_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(label_key)

        for shape_key in shape_keys:
            transformations = get_transformation(sdata.shapes[shape_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(shape_key)

        for point_key in point_keys:
            transformations = get_transformation(sdata.points[point_key], get_all=True)

            if key in list(transformations.keys()):
                mapping[key].append(point_key)

    return mapping


def _flatten_transformation_sequence(
    transformation_sequence: list[Sequence],
) -> list[Sequence]:
    if isinstance(transformation_sequence, Sequence):
        transformations = list(transformation_sequence.transformations)
        found_bottom_of_tree = False
        while not found_bottom_of_tree:
            if all(not isinstance(t, Sequence) for t in transformations):
                found_bottom_of_tree = True
            else:
                for idx, t in enumerate(transformations):
                    if isinstance(t, Sequence):
                        transformations.pop(idx)
                        transformations += t.transformations

        return transformations

    if isinstance(transformation_sequence, BaseTransformation):
        return [transformation_sequence]

    raise TypeError("Parameter 'transformation_sequence' must be a Sequence.")


def _get_cs_contents(sdata: SpatialData) -> pd.DataFrame:
    """Check which coordinate systems contain which elements and return that info."""
    cs_mapping = _get_coordinate_system_mapping(sdata)
    content_flags = ["has_images", "has_labels", "has_points", "has_shapes"]
    cs_contents = pd.DataFrame(columns=["cs"] + content_flags)

    for cs_name, element_ids in cs_mapping.items():
        # determine if coordinate system has the respective elements
        cs_has_images = bool(any((e in sdata.images) for e in element_ids))
        cs_has_labels = bool(any((e in sdata.labels) for e in element_ids))
        cs_has_points = bool(any((e in sdata.points) for e in element_ids))
        cs_has_shapes = bool(any((e in sdata.shapes) for e in element_ids))

        cs_contents = pd.concat(
            [
                cs_contents,
                pd.DataFrame(
                    {
                        "cs": cs_name,
                        "has_images": [cs_has_images],
                        "has_labels": [cs_has_labels],
                        "has_points": [cs_has_points],
                        "has_shapes": [cs_has_shapes],
                    }
                ),
            ]
        )

        cs_contents["has_images"] = cs_contents["has_images"].astype("bool")
        cs_contents["has_labels"] = cs_contents["has_labels"].astype("bool")
        cs_contents["has_points"] = cs_contents["has_points"].astype("bool")
        cs_contents["has_shapes"] = cs_contents["has_shapes"].astype("bool")

    return cs_contents


def _get_extent(
    sdata: SpatialData,
    coordinate_systems: Sequence[str] | str | None = None,
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: Iterable[Any] | None = None,
    share_extent: bool = False,
) -> dict[str, tuple[int, int, int, int]]:
    """Return the extent of all elements in their respective coordinate systems.

    Parameters
    ----------
    sdata
        The sd.SpatialData object to retrieve the extent from
    has_images
        Flag indicating whether to consider images when calculating the extent
    has_labels
        Flag indicating whether to consider labels when calculating the extent
    has_points
        Flag indicating whether to consider points when calculating the extent
    has_shapes
        Flag indicating whether to consider shapes when calculating the extent
    elements
        Optional list of element names to be considered. When None, all are used.
    share_extent
        Flag indicating whether to use the same extent for all coordinate systems

    Returns
    -------
    A dict of tuples with the shape (xmin, xmax, ymin, ymax). The keys of the
        dict are the coordinate_system keys.

    """
    extent: dict[str, dict[str, Sequence[int]]] = {}
    cs_mapping = _get_coordinate_system_mapping(sdata)
    cs_contents = _get_cs_contents(sdata)

    if elements is None:  # to shut up ruff
        elements = []

    if not isinstance(elements, list):
        raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")

    if coordinate_systems is not None:
        if isinstance(coordinate_systems, str):
            coordinate_systems = [coordinate_systems]
        cs_contents = cs_contents[cs_contents["cs"].isin(coordinate_systems)]
        cs_mapping = {k: v for k, v in cs_mapping.items() if k in coordinate_systems}

    for cs_name, element_ids in cs_mapping.items():
        extent[cs_name] = {}
        if len(elements) > 0:
            element_ids = [e for e in element_ids if e in elements]

        def _get_extent_after_transformations(element: Any, cs_name: str) -> Sequence[int]:
            tmp = element.copy()
            if len(tmp.shape) == 3:
                x_idx = 2
                y_idx = 1
            elif len(tmp.shape) == 2:
                x_idx = 1
                y_idx = 0

            transformations = get_transformation(tmp, to_coordinate_system=cs_name)
            transformations = _flatten_transformation_sequence(transformations)

            if len(transformations) == 1 and isinstance(transformations[0], Identity):
                result = (0, tmp.shape[x_idx], 0, tmp.shape[y_idx])

            else:
                origin = {
                    "x": 0,
                    "y": 0,
                }
                for t in transformations:
                    if isinstance(t, Translation):
                        # TODO: remove, in get_extent no data operation should be performed
                        tmp = _translate_image(image=tmp, translation=t)

                        for idx, ax in enumerate(t.axes):
                            origin["x"] += t.translation[idx] if ax == "x" else 0
                            origin["y"] += t.translation[idx] if ax == "y" else 0

                    else:
                        # TODO: remove, in get_extent no data operation should be performed
                        tmp = transform(tmp, t)

                        if isinstance(t, Scale):
                            for idx, ax in enumerate(t.axes):
                                origin["x"] *= t.scale[idx] if ax == "x" else 1
                                origin["y"] *= t.scale[idx] if ax == "y" else 1

                        elif isinstance(t, Affine):
                            pass

                result = (origin["x"], tmp.shape[x_idx], origin["y"], tmp.shape[y_idx])

            del tmp
            return result

        if has_images and cs_contents.query(f"cs == '{cs_name}'")["has_images"][0]:
            for images_key in sdata.images:
                for e_id in element_ids:
                    if images_key == e_id:
                        if not isinstance(sdata.images[e_id], MultiscaleSpatialImage):
                            extent[cs_name][e_id] = _get_extent_after_transformations(sdata.images[e_id], cs_name)
                        else:
                            pass

        if has_labels and cs_contents.query(f"cs == '{cs_name}'")["has_labels"][0]:
            for labels_key in sdata.labels:
                for e_id in element_ids:
                    if labels_key == e_id:
                        if not isinstance(sdata.labels[e_id], MultiscaleSpatialImage):
                            extent[cs_name][e_id] = _get_extent_after_transformations(sdata.labels[e_id], cs_name)
                        else:
                            pass

        if has_shapes and cs_contents.query(f"cs == '{cs_name}'")["has_shapes"][0]:
            for shapes_key in sdata.shapes:
                for e_id in element_ids:
                    if shapes_key == e_id:

                        def get_point_bb(
                            point: Point, radius: int, method: Literal["topleft", "bottomright"], buffer: int = 1
                        ) -> Point:
                            x, y = point.coords[0]
                            if method == "topleft":
                                point_bb = Point(x - radius - buffer, y - radius - buffer)
                            else:
                                point_bb = Point(x + radius + buffer, y + radius + buffer)

                            return point_bb

                        y_dims = []
                        x_dims = []

                        # Split by Point and Polygon:
                        tmp_points = sdata.shapes[e_id][
                            sdata.shapes[e_id]["geometry"].apply(lambda geom: geom.geom_type == "Point")
                        ]
                        tmp_polygons = sdata.shapes[e_id][
                            sdata.shapes[e_id]["geometry"].apply(
                                lambda geom: geom.geom_type in ["Polygon", "MultiPolygon"]
                            )
                        ]

                        if not tmp_points.empty:
                            tmp_points["point_topleft"] = tmp_points.apply(
                                lambda row: get_point_bb(row["geometry"], row["radius"], "topleft"),
                                axis=1,
                            )
                            tmp_points["point_bottomright"] = tmp_points.apply(
                                lambda row: get_point_bb(row["geometry"], row["radius"], "bottomright"),
                                axis=1,
                            )
                            xmin_tl, ymin_tl, xmax_tl, ymax_tl = tmp_points["point_topleft"].total_bounds
                            xmin_br, ymin_br, xmax_br, ymax_br = tmp_points["point_bottomright"].total_bounds
                            y_dims += [min(ymin_tl, ymin_br), max(ymax_tl, ymax_br)]
                            x_dims += [min(xmin_tl, xmin_br), max(xmax_tl, xmax_br)]

                        if not tmp_polygons.empty:
                            xmin, ymin, xmax, ymax = tmp_polygons.total_bounds
                            y_dims += [ymin, ymax]
                            x_dims += [xmin, xmax]

                        del tmp_points
                        del tmp_polygons

                        extent[cs_name][e_id] = x_dims + y_dims

                        transformations = get_transformation(sdata.shapes[e_id], to_coordinate_system=cs_name)
                        transformations = _flatten_transformation_sequence(transformations)

                        for t in transformations:
                            if isinstance(t, Translation):
                                for idx, ax in enumerate(t.axes):
                                    extent[cs_name][e_id][0] += t.translation[idx] if ax == "x" else 0  # type: ignore
                                    extent[cs_name][e_id][1] += t.translation[idx] if ax == "x" else 0  # type: ignore
                                    extent[cs_name][e_id][2] += t.translation[idx] if ax == "y" else 0  # type: ignore
                                    extent[cs_name][e_id][3] += t.translation[idx] if ax == "y" else 0  # type: ignore

                            else:
                                if isinstance(t, Scale):
                                    for idx, ax in enumerate(t.axes):
                                        extent[cs_name][e_id][1] *= t.scale[idx] if ax == "x" else 1  # type: ignore
                                        extent[cs_name][e_id][3] *= t.scale[idx] if ax == "y" else 1  # type: ignore

                                elif isinstance(t, Affine):
                                    pass

        if has_points and cs_contents.query(f"cs == '{cs_name}'")["has_points"][0]:
            for points_key in sdata.points:
                for e_id in element_ids:
                    if points_key == e_id:
                        tmp = sdata.points[points_key]
                        xmin = tmp["x"].min().compute()
                        xmax = tmp["x"].max().compute()
                        ymin = tmp["y"].min().compute()
                        ymax = tmp["y"].max().compute()
                        extent[cs_name][e_id] = [xmin, xmax, ymin, ymax]

    cswise_extent = {}
    for cs_name, cs_contents in extent.items():
        if len(cs_contents) > 0:
            xmin = min([v[0] for v in cs_contents.values()])
            xmax = max([v[1] for v in cs_contents.values()])
            ymin = min([v[2] for v in cs_contents.values()])
            ymax = max([v[3] for v in cs_contents.values()])
            cswise_extent[cs_name] = (xmin, xmax, ymin, ymax)

    if share_extent:
        global_extent = {}
        if len(cs_contents) > 0:
            xmin = min([v[0] for v in cswise_extent.values()])
            xmax = max([v[1] for v in cswise_extent.values()])
            ymin = min([v[2] for v in cswise_extent.values()])
            ymax = max([v[3] for v in cswise_extent.values()])
            for cs_name in cswise_extent:
                global_extent[cs_name] = (xmin, xmax, ymin, ymax)
        return global_extent

    return cswise_extent
