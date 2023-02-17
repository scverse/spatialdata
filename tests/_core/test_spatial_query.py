from dataclasses import FrozenInstanceError

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import linearrings, polygons

from spatialdata import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
)
from spatialdata._core._spatial_query import (
    BaseSpatialRequest,
    BoundingBoxRequest,
    _bounding_box_query_image,
    _bounding_box_query_points,
    _bounding_box_query_shapes,
)


def _make_points_element():
    """Helper function to make a Points element."""
    coordinates = np.array([[10, 10], [20, 20], [20, 30]], dtype=float)
    return PointsModel.parse(
        coordinates, annotation=pd.DataFrame({"genes": np.repeat("a", len(coordinates))}), feature_key="genes"
    )


def test_bounding_box_request_immutable():
    """Test that the bounding box request is immutable."""
    request = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([0, 0]), max_coordinate=np.array([10, 10]))
    isinstance(request, BaseSpatialRequest)

    # fields should be immutable
    with pytest.raises(FrozenInstanceError):
        request.axes = ("c", "y", "x")
    with pytest.raises(FrozenInstanceError):
        request.axes = ("z", "y", "x")
    with pytest.raises(FrozenInstanceError):
        request.min_coordinate = np.array([5, 5, 5])
    with pytest.raises(FrozenInstanceError):
        request.max_coordinate = np.array([5, 5, 5])


def test_bounding_box_request_only_spatial_axes():
    """Requests with axes that are not spatial should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(axes=("c", "x"), min_coordinate=np.array([0, 0]), max_coordinate=np.array([10, 10]))


def test_bounding_box_request_wrong_number_of_coordinates():
    """Requests which specify coordinates not consistent with the axes should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([0, 0, 0]), max_coordinate=np.array([10, 10]))

    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([0, 0]), max_coordinate=np.array([10, 10, 10]))

    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(
            axes=("y", "x"), min_coordinate=np.array([0, 0, 0]), max_coordinate=np.array([10, 10, 10])
        )


def test_bounding_box_request_wrong_coordinate_order():
    """Requests where the min coordinate is greater than the max coordinate should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([0, 10]), max_coordinate=np.array([10, 0]))


def test_bounding_box_points():
    """test the points bounding box_query"""
    points_element = _make_points_element()
    original_x = np.array(points_element["x"])
    original_y = np.array(points_element["y"])

    request = BoundingBoxRequest(axes=("x", "y"), min_coordinate=np.array([18, 25]), max_coordinate=np.array([22, 35]))
    points_result = _bounding_box_query_points(points_element, request)
    np.testing.assert_allclose(points_result["x"], [20])
    np.testing.assert_allclose(points_result["y"], [30])

    # result should be valid points element
    PointsModel.validate(points_result)

    # original element should be unchanged
    np.testing.assert_allclose(points_element["x"], original_x)
    np.testing.assert_allclose(points_element["y"], original_y)


def test_bounding_box_points_no_points():
    """Points bounding box query with no points in range should
    return a points element with length 0.
    """
    points_element = _make_points_element()
    request = BoundingBoxRequest(axes=("x", "y"), min_coordinate=np.array([40, 50]), max_coordinate=np.array([45, 55]))
    points_result = _bounding_box_query_points(points_element, request)
    assert len(points_result) == 0

    # result should be valid points element
    PointsModel.validate(points_result)


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_bounding_box_image_2d(n_channels):
    """Apply a bounding box to a 2D image"""
    image = np.zeros((n_channels, 10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[:, 5::, 0:5] = 1
    image_element = Image2DModel.parse(image)

    # bounding box: y: [5, 9], x: [0, 4]
    request = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([5, 0]), max_coordinate=np.array([9, 4]))

    image_result = _bounding_box_query_image(image_element, request)
    expected_image = np.ones((n_channels, 5, 5))  # c dimension is preserved
    np.testing.assert_allclose(image_result, expected_image)


@pytest.mark.skip(reason="Image3D parser not working")
@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_bounding_box_image_3d(n_channels):
    """Apply a bounding box to a 3D image"""
    image = np.zeros((n_channels, 10, 10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[:, 5::, 0:5, 2:7] = 1
    image_element = Image3DModel.parse(image)

    # bounding box: z: [5, 9], y: [5, 9], x: [0, 4]
    request = BoundingBoxRequest(
        axes=("z", "y", "x"), min_coordinate=np.array([5, 0, 2]), max_coordinate=np.array([9, 4, 6])
    )

    image_result = _bounding_box_query_image(image_element, request)
    expected_image = np.ones((n_channels, 5, 5, 5))  # c dimension is preserved
    np.testing.assert_allclose(image_result, expected_image)


def test_bounding_box_labels_2d():
    """Apply a bounding box to a 2D label image"""
    image = np.zeros((10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[5::, 0:5] = 1
    labels_element = Labels2DModel.parse(image)

    # bounding box: y: [5, 9], x: [0, 4]
    request = BoundingBoxRequest(axes=("y", "x"), min_coordinate=np.array([5, 0]), max_coordinate=np.array([9, 4]))

    labels_result = _bounding_box_query_image(labels_element, request)
    expected_image = np.ones((5, 5))
    np.testing.assert_allclose(labels_result, expected_image)


def test_bounding_box_labels_3d():
    """Apply a bounding box to a 3D label image"""
    image = np.zeros((10, 10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[5::, 0:5, 2:7] = 1
    labels_element = Labels3DModel.parse(image)

    # bounding box: z: [5, 9], y: [5, 9], x: [0, 4]
    request = BoundingBoxRequest(
        axes=("z", "y", "x"), min_coordinate=np.array([5, 0, 2]), max_coordinate=np.array([9, 4, 6])
    )

    image_result = _bounding_box_query_image(labels_element, request)
    expected_image = np.ones((5, 5, 5))
    np.testing.assert_allclose(image_result, expected_image)


def _make_squares(centroid_coordinates: np.ndarray, half_width: float) -> polygons:
    linear_rings = []
    for centroid in centroid_coordinates:
        min_coords = centroid - half_width
        max_coords = centroid + half_width

        linear_rings.append(
            linearrings(
                [
                    [min_coords[0], min_coords[1]],
                    [min_coords[0], max_coords[1]],
                    [max_coords[0], max_coords[1]],
                    [max_coords[0], min_coords[1]],
                ]
            )
        )
    return polygons(linear_rings)


def test_bounding_box_polygons():
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])
    cell_outline_polygons = _make_squares(centroid_coordinates=centroids, half_width=6)

    polygon_series = gpd.GeoSeries(cell_outline_polygons)
    cell_polygon_table = gpd.GeoDataFrame(geometry=polygon_series)
    sd_polygons = ShapesModel.parse(cell_polygon_table)

    request = BoundingBoxRequest(
        axes=("y", "x"), min_coordinate=np.array([40, 40]), max_coordinate=np.array([100, 100])
    )
    polygons_result = _bounding_box_query_shapes(sd_polygons, request)

    assert len(polygons_result) == 1
    assert polygons_result.index[0] == 3


def test_bounding_box_circles():
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])

    sd_circles = ShapesModel.parse(centroids, geometry=0, radius=10)

    request = BoundingBoxRequest(
        axes=("y", "x"), min_coordinate=np.array([40, 40]), max_coordinate=np.array([100, 100])
    )
    circles_result = _bounding_box_query_shapes(sd_circles, request)

    assert len(circles_result) == 1
    assert circles_result.index[0] == 3
