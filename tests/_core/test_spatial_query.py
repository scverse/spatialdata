from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from spatialdata._core._spatial_query import BaseSpatialRequest, BoundingBoxRequest
from tests._core.conftest import cyx_cs, czyx_cs


def test_bounding_box_request_immutable():
    """Test that the bounding box query is immutable."""
    query = BoundingBoxRequest(
        coordinate_system=cyx_cs, min_coordinate=np.array([0, 0]), max_coordinate=np.array([10, 10])
    )
    isinstance(query, BaseSpatialRequest)

    # fields should be immutable
    with pytest.raises(FrozenInstanceError):
        query.coordinate_system = czyx_cs
    with pytest.raises(FrozenInstanceError):
        query.min_coordinate = np.array([5, 5, 5])
    with pytest.raises(FrozenInstanceError):
        query.max_coordinate = np.array([5, 5, 5])
