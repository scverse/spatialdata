from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spatialdata._core._spatial_query import BaseSpatialRequest, BoundingBoxRequest
from tests._core.conftest import c_cs, cyx_cs, czyx_cs


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
