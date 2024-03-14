import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely import Point
from spatialdata._core.operations.vectorize import to_circles
from spatialdata.datasets import blobs
from spatialdata.models.models import ShapesModel
from spatialdata.testing import assert_elements_are_identical

# each of the tests operates on different elements, hence we can initialize the data once without conflicts
sdata = blobs()


@pytest.mark.parametrize("is_multiscale", [False, True])
def test_labels_2d_to_circles(is_multiscale: bool) -> None:
    key = "blobs" + ("_multiscale" if is_multiscale else "") + "_labels"
    element = sdata[key]
    new_circles = to_circles(element)

    assert np.isclose(new_circles.loc[1].geometry.x, 330.59258152354386)
    assert np.isclose(new_circles.loc[1].geometry.y, 78.85026897788404)
    assert np.isclose(new_circles.loc[1].radius, 69.229993)
    assert 7 not in new_circles.index


@pytest.mark.skip(reason="Not implemented")
# @pytest.mark.parametrize("background", [0, 1])
# @pytest.mark.parametrize("is_multiscale", [False, True])
def test_labels_3d_to_circles() -> None:
    pass


def test_circles_to_circles() -> None:
    element = sdata["blobs_circles"]
    new_circles = to_circles(element)
    assert_elements_are_identical(element, new_circles)


def test_polygons_to_circles() -> None:
    element = sdata["blobs_polygons"].iloc[:2]
    new_circles = to_circles(element)

    data = {
        "geometry": [Point(315.8120722406787, 220.18894606643332), Point(270.1386975678398, 417.8747936281634)],
        "radius": [16.608781, 17.541365],
    }
    expected = ShapesModel.parse(GeoDataFrame(data, geometry="geometry"))

    assert_elements_are_identical(new_circles, expected)


def test_multipolygons_to_circles() -> None:
    element = sdata["blobs_multipolygons"]
    new_circles = to_circles(element)

    data = {
        "geometry": [Point(340.37951022629096, 250.76310705786318), Point(337.1680699150594, 316.39984581697314)],
        "radius": [23.488363, 19.059285],
    }
    expected = ShapesModel.parse(GeoDataFrame(data, geometry="geometry"))
    assert_elements_are_identical(new_circles, expected)


def test_points_images_to_circles() -> None:
    with pytest.raises(RuntimeError, match=r"Cannot apply to_circles\(\) to images."):
        to_circles(sdata["blobs_image"])
    with pytest.raises(RuntimeError, match="Unsupported type"):
        to_circles(sdata["blobs_points"])
