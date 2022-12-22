from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spatialdata import PointsModel
from spatialdata._core._spatial_query import (
    BaseSpatialRequest,
    BoundingBoxRequest,
    _bounding_box_query_points,
)
from tests._core.conftest import c_cs, cyx_cs, czyx_cs, xy_cs


def _make_points_element():
    """Helper function to make a Points element."""
    coordinates = np.array([[10, 10], [20, 20], [20, 30]], dtype=float)
    return PointsModel.parse(coordinates)


def test_bounding_box_request_immutable():
    """Test that the bounding box request is immutable."""
    request = BoundingBoxRequest(
        coordinate_system=cyx_cs, min_coordinate=np.array([0, 0]), max_coordinate=np.array([10, 10])
    )
    isinstance(request, BaseSpatialRequest)

    # fields should be immutable
    with pytest.raises(FrozenInstanceError):
        request.coordinate_system = czyx_cs
    with pytest.raises(FrozenInstanceError):
        request.min_coordinate = np.array([5, 5, 5])
    with pytest.raises(FrozenInstanceError):
        request.max_coordinate = np.array([5, 5, 5])


def test_bounding_box_request_no_spatial_axes():
    """Requests with no spatial axes should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(coordinate_system=c_cs, min_coordinate=np.array([0]), max_coordinate=np.array([10]))


def test_bounding_box_points():
    """test the points bounding box_query"""
    points_element = _make_points_element()
    original_x = np.array(points_element["x"])
    original_y = np.array(points_element["y"])

    request = BoundingBoxRequest(
        coordinate_system=xy_cs, min_coordinate=np.array([18, 25]), max_coordinate=np.array([22, 35])
    )
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
    request = BoundingBoxRequest(
        coordinate_system=xy_cs, min_coordinate=np.array([40, 50]), max_coordinate=np.array([45, 55])
    )
    points_result = _bounding_box_query_points(points_element, request)
    assert len(points_result) == 0

    # result should be valid points element
    PointsModel.validate(points_result)
