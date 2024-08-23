import math

import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely import MultiPoint, Point

from spatialdata._core.operations.vectorize import to_circles, to_polygons
from spatialdata.datasets import blobs
from spatialdata.models.models import ShapesModel
from spatialdata.testing import assert_elements_are_identical

# each of the tests operates on different elements, hence we can initialize the data once without conflicts
sdata = blobs()


# conversion from labels
@pytest.mark.parametrize("is_multiscale", [False, True])
def test_labels_2d_to_circles(is_multiscale: bool) -> None:
    key = "blobs" + ("_multiscale" if is_multiscale else "") + "_labels"
    element = sdata[key]
    new_circles = to_circles(element)

    assert np.isclose(new_circles.loc[1].geometry.x, 330.59258152354386)
    assert np.isclose(new_circles.loc[1].geometry.y, 78.85026897788404)
    assert np.isclose(new_circles.loc[1].radius, 69.229993)
    assert 7 not in new_circles.index


@pytest.mark.parametrize("is_multiscale", [False, True])
def test_labels_2d_to_polygons(is_multiscale: bool) -> None:
    key = "blobs" + ("_multiscale" if is_multiscale else "") + "_labels"
    element = sdata[key]
    new_polygons = to_polygons(element)

    assert 7 not in new_polygons.index

    unique, counts = np.unique(sdata["blobs_labels"].compute().data, return_counts=True)
    new_polygons.loc[unique[1:], "pixel_count"] = counts[1:]

    assert ((new_polygons.area - new_polygons.pixel_count) / new_polygons.pixel_count < 0.01).all()


def test_chunked_labels_2d_to_polygons() -> None:
    no_chunks_polygons = to_polygons(sdata["blobs_labels"])

    sdata["blobs_labels_chunked"] = sdata["blobs_labels"].copy()
    sdata["blobs_labels_chunked"].data = sdata["blobs_labels_chunked"].data.rechunk((200, 200))

    chunks_polygons = to_polygons(sdata["blobs_labels_chunked"])

    union = chunks_polygons.union(no_chunks_polygons)

    (no_chunks_polygons.area == union.area).all()


# conversion from circles
def test_circles_to_circles() -> None:
    element = sdata["blobs_circles"]
    new_circles = to_circles(element)
    assert_elements_are_identical(element, new_circles)


def test_circles_to_polygons() -> None:
    element = sdata["blobs_circles"]
    polygons = to_polygons(element, buffer_resolution=1000)
    areas = element.radius**2 * math.pi
    assert np.allclose(polygons.area, areas)


# conversion from polygons/multipolygons
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


def test_polygons_multipolygons_to_polygons() -> None:
    polygons = sdata["blobs_multipolygons"]
    assert polygons is to_polygons(polygons)


# conversion from points
def test_points_to_circles() -> None:
    element = sdata["blobs_points"]
    with pytest.raises(RuntimeError, match="`radius` must either be provided, either be a column"):
        to_circles(element)
    circles = to_circles(element, radius=1)
    x = circles.geometry.x
    y = circles.geometry.y
    assert np.array_equal(element["x"], x)
    assert np.array_equal(element["y"], y)
    assert np.array_equal(np.ones_like(x), circles["radius"])


def test_points_to_polygons() -> None:
    with pytest.raises(RuntimeError, match="Cannot convert points to polygons"):
        to_polygons(sdata["blobs_points"])


# conversion from images (invalid)
def test_images_to_circles() -> None:
    with pytest.raises(RuntimeError, match=r"Cannot apply to_circles\(\) to images"):
        to_circles(sdata["blobs_image"])


def test_images_to_polygons() -> None:
    with pytest.raises(RuntimeError, match=r"Cannot apply to_polygons\(\) to images"):
        to_polygons(sdata["blobs_image"])


# conversion from other types (invalid)
def test_invalid_geodataframe_to_circles() -> None:
    gdf = GeoDataFrame(geometry=[MultiPoint([[0, 0], [1, 1]])])
    with pytest.raises(RuntimeError, match="Unsupported geometry type"):
        to_circles(gdf)


def test_invalid_geodataframe_to_polygons() -> None:
    gdf = GeoDataFrame(geometry=[MultiPoint([[0, 0], [1, 1]])])
    with pytest.raises(RuntimeError, match="Unsupported geometry type"):
        to_polygons(gdf)
