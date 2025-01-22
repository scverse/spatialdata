import math

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from numpy.random import default_rng
from shapely.geometry import MultiPolygon, Point, Polygon

from spatialdata import SpatialData, get_extent, transform
from spatialdata._core._deepcopy import deepcopy as _deepcopy
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, PointsModel, ShapesModel
from spatialdata.transformations import Affine, Translation, remove_transformation, set_transformation

# for faster tests; we will pay attention not to modify the original data
sdata = blobs()
RNG = default_rng(seed=0)


def check_test_results0(extent, min_coordinates, max_coordinates, axes):
    for i, ax in enumerate(axes):
        assert np.isclose(extent[ax][0], min_coordinates[i])
        assert np.isclose(extent[ax][1], max_coordinates[i])
    extend_axes = list(extent.keys())
    extend_axes.sort()
    assert tuple(extend_axes) == axes


def check_test_results1(extent0, extent1):
    assert extent0.keys() == extent1.keys()
    for ax in extent0:
        assert np.allclose(extent0[ax], extent1[ax])


@pytest.mark.parametrize("shape_type", ["circles", "polygons", "multipolygons"])
def test_get_extent_shapes(shape_type):
    extent = get_extent(sdata[f"blobs_{shape_type}"])
    if shape_type == "circles":
        min_coordinates = np.array([98.92618679, 137.62348969])
        max_coordinates = np.array([420.42236303, 422.31870626])
    elif shape_type == "polygons":
        min_coordinates = np.array([149.92618679, 188.62348969])
        max_coordinates = np.array([446.70264371, 461.85209239])
    else:
        assert shape_type == "multipolygons"
        min_coordinates = np.array([291.06219195, 197.06539872])
        max_coordinates = np.array([389.3319439, 375.89584037])

    check_test_results0(
        extent,
        min_coordinates=min_coordinates,
        max_coordinates=max_coordinates,
        axes=("x", "y"),
    )


@pytest.mark.parametrize("exact", [True, False])
def test_get_extent_points(exact: bool):
    # 2d case
    extent = get_extent(sdata["blobs_points"], exact=exact)
    check_test_results0(
        extent,
        min_coordinates=np.array([3.0, 4.0]),
        max_coordinates=np.array([509.0, 507.0]),
        axes=("x", "y"),
    )

    # 3d case
    data = np.array([[1, 2, 3], [4, 5, 6]])
    df = pd.DataFrame(data, columns=["zeta", "x", "y"])
    points_3d = PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "zeta"})
    extent_3d = get_extent(points_3d, exact=exact)
    check_test_results0(
        extent_3d,
        min_coordinates=np.array([2, 3, 1]),
        max_coordinates=np.array([5, 6, 4]),
        axes=("x", "y", "z"),
    )


@pytest.mark.parametrize("raster_type", ["image", "labels"])
@pytest.mark.parametrize("multiscale", [False, True])
def test_get_extent_raster(raster_type, multiscale):
    raster = sdata[f"blobs_multiscale_{raster_type}"] if multiscale else sdata[f"blobs_{raster_type}"]

    extent = get_extent(raster)
    check_test_results0(
        extent,
        min_coordinates=np.array([0, 0]),
        max_coordinates=np.array([512, 512]),
        axes=("x", "y"),
    )


def test_get_extent_spatialdata():
    sdata2 = SpatialData(shapes={"circles": sdata["blobs_circles"], "polygons": sdata["blobs_polygons"]})
    extent = get_extent(sdata2)
    check_test_results0(
        extent,
        min_coordinates=np.array([98.92618679, 137.62348969]),
        max_coordinates=np.array([446.70264371, 461.85209239]),
        axes=("x", "y"),
    )


def test_get_extent_invalid_coordinate_system():
    # element without the coordinate system
    with pytest.raises(ValueError):
        _ = get_extent(sdata["blobs_circles"], coordinate_system="invalid")
    # sdata object with no element with the coordinate system
    with pytest.raises(ValueError):
        _ = get_extent(sdata, coordinate_system="invalid")


def _rotate_point(point: tuple[float, float], angle_degrees=45) -> tuple[float, float]:
    angle_radians = math.radians(angle_degrees)
    x, y = point

    x_prime = x * math.cos(angle_radians) - y * math.sin(angle_radians)
    y_prime = x * math.sin(angle_radians) + y * math.cos(angle_radians)

    return (x_prime, y_prime)


@pytest.mark.parametrize("exact", [True, False])
def test_rotate_vector_data(exact):
    """
    To test for the ability to correctly compute the exact and approximate extent of vector datasets.
    In particular tests for the solution to this issue: https://github.com/scverse/spatialdata/issues/353
    """
    circles = []
    for p in [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]]:
        circles.append(Point(p))
    circles_gdf = GeoDataFrame(geometry=circles)
    circles_gdf["radius"] = 0.1
    circles_gdf = ShapesModel.parse(circles_gdf)

    polygons = []
    polygons.append(Polygon([(0.5, 0.5), (0.5, 0), (0.6, 0.1), (0.5, 0.5)]))
    polygons.append(Polygon([(0.5, 0.5), (1, 0.5), (0.9, 0.6), (0.5, 0.5)]))
    polygons.append(Polygon([(0.5, 0.5), (0.5, 1), (0.4, 0.9), (0.5, 0.5)]))
    polygons.append(Polygon([(0.5, 0.5), (0, 0.5), (0.1, 0.4), (0.5, 0.5)]))
    polygons_gdf = GeoDataFrame(geometry=polygons)
    polygons_gdf = ShapesModel.parse(polygons_gdf)

    multipolygons = []
    multipolygons.append(MultiPolygon([polygons[0], Polygon([(0.7, 0.1), (0.9, 0.1), (0.9, 0.3), (0.7, 0.1)])]))
    multipolygons.append(MultiPolygon([polygons[1], Polygon([(0.9, 0.7), (0.9, 0.9), (0.7, 0.9), (0.9, 0.7)])]))
    multipolygons.append(MultiPolygon([polygons[2], Polygon([(0.3, 0.9), (0.1, 0.9), (0.1, 0.7), (0.3, 0.9)])]))
    multipolygons.append(MultiPolygon([polygons[3], Polygon([(0.1, 0.3), (0.1, 0.1), (0.3, 0.1), (0.1, 0.3)])]))
    multipolygons_gdf = GeoDataFrame(geometry=multipolygons)
    multipolygons_gdf = ShapesModel.parse(multipolygons_gdf)

    points_df = PointsModel.parse(np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]))

    sdata = SpatialData(
        shapes={"circles": circles_gdf, "polygons": polygons_gdf, "multipolygons": multipolygons_gdf},
        points={"points": points_df},
    )

    theta = math.pi / 4
    rotation = Affine(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    for element_name in ["circles", "polygons", "multipolygons", "points"]:
        set_transformation(element=sdata[element_name], transformation=rotation, to_coordinate_system="transformed")

    # manually computing the extent results and verifying it is correct
    for e in [sdata, circles_gdf, polygons_gdf, multipolygons_gdf, points_df]:
        extent = get_extent(e, coordinate_system="global")
        check_test_results1(extent, {"x": (0.0, 1.0), "y": (0.0, 1.0)})

    EXPECTED_NON_EXACT = {"x": (-math.sqrt(2) / 2, math.sqrt(2) / 2), "y": (0.0, math.sqrt(2))}
    extent = get_extent(circles_gdf, coordinate_system="transformed", exact=exact)
    if exact:
        expected = {
            "x": (_rotate_point((0.1, 0.5))[0] - 0.1, _rotate_point((0.5, 0.1))[0] + 0.1),
            "y": (_rotate_point((0.5, 0.1))[1] - 0.1, _rotate_point((0.9, 0.5))[1] + 0.1),
        }
    else:
        expected = EXPECTED_NON_EXACT
    check_test_results1(extent, expected)

    extent = get_extent(polygons_gdf, coordinate_system="transformed", exact=exact)
    if exact:
        expected = {
            "x": (_rotate_point((0, 0.5))[0], _rotate_point((0.5, 0))[0]),
            "y": (_rotate_point((0.5, 0))[1], _rotate_point((1, 0.5))[1]),
        }
    else:
        expected = EXPECTED_NON_EXACT
    check_test_results1(extent, expected)

    extent = get_extent(multipolygons_gdf, coordinate_system="transformed", exact=exact)
    if exact:
        expected = {
            "x": (_rotate_point((0.1, 0.9))[0], _rotate_point((0.9, 0.1))[0]),
            "y": (_rotate_point((0.1, 0.1))[1], _rotate_point((0.9, 0.9))[1]),
        }
    else:
        expected = EXPECTED_NON_EXACT
    check_test_results1(extent, expected)

    extent = get_extent(points_df, coordinate_system="transformed", exact=exact)
    if exact:
        expected = {
            "x": (_rotate_point((0, 0.5))[0], _rotate_point((0.5, 0))[0]),
            "y": (_rotate_point((0.5, 0))[1], _rotate_point((1, 0.5))[1]),
        }
    else:
        expected = EXPECTED_NON_EXACT
    check_test_results1(extent, expected)

    extent = get_extent(sdata, coordinate_system="transformed", exact=exact)
    if exact:
        expected = {
            "x": (_rotate_point((0.1, 0.9))[0], _rotate_point((0.9, 0.1))[0]),
            "y": (_rotate_point((0.1, 0.1))[1], _rotate_point((0.9, 0.9))[1]),
        }
    else:
        expected = EXPECTED_NON_EXACT
    check_test_results1(extent, expected)


def test_get_extent_affine_circles():
    """
    Verify that the extent of the transformed circles, computed with exact = False, gives the same result as
    transforming the bounding box of the original circles
    """
    from tests.core.operations.test_transform import _get_affine

    affine = _get_affine(small_translation=True)

    # let's do a deepcopy of the circles since we don't want to modify the original data
    circles = _deepcopy(sdata["blobs_circles"])

    set_transformation(element=circles, transformation=affine, to_coordinate_system="transformed")

    extent = get_extent(circles)
    transformed_extent = get_extent(circles, coordinate_system="transformed", exact=False)

    axes = list(extent.keys())
    transformed_axes = list(extent.keys())
    assert axes == transformed_axes
    for ax in axes:
        assert not np.allclose(extent[ax], transformed_extent[ax])

    # Create a list of points
    points = [
        (extent["x"][0], extent["y"][0]),  # lower left corner
        (extent["x"][0], extent["y"][1]),  # upper left corner
        (extent["x"][1], extent["y"][1]),  # upper right corner
        (extent["x"][1], extent["y"][0]),  # lower right corner
        (extent["x"][0], extent["y"][0]),  # back to start to close the polygon
    ]

    # Create a Polygon from the points
    bounding_box = Polygon(points)
    gdf = GeoDataFrame(geometry=[bounding_box])
    gdf = ShapesModel.parse(gdf, transformations={"transformed": affine})
    transformed_bounding_box = transform(gdf, to_coordinate_system="transformed")

    transformed_bounding_box_extent = get_extent(transformed_bounding_box, coordinate_system="transformed")

    assert transformed_axes == list(transformed_bounding_box_extent.keys())
    for ax in transformed_axes:
        assert np.allclose(transformed_extent[ax], transformed_bounding_box_extent[ax])


def test_get_extent_affine_points3d():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    points_2d = PointsModel.parse(data[:, :2])

    points_3d = PointsModel.parse(data)
    extent_3d = get_extent(points_3d)

    from tests.core.operations.test_transform import _get_affine

    affine = _get_affine(small_translation=True)

    set_transformation(element=points_2d, transformation=affine, to_coordinate_system="transformed")
    set_transformation(element=points_3d, transformation=affine, to_coordinate_system="transformed")

    transformed_extent_2d = get_extent(points_2d, coordinate_system="transformed")
    transformed_extent_3d = get_extent(points_3d, coordinate_system="transformed")

    assert list(transformed_extent_2d.keys()) == ["x", "y"]
    assert list(transformed_extent_3d.keys()) == ["x", "y", "z"]

    # the x and y extent for the 2d and 3d points are identical
    for ax in ["x", "y"]:
        assert np.allclose(transformed_extent_2d[ax], transformed_extent_3d[ax])

    # the z extent for the 3d points didn't get transformed, so it's the same as the original
    assert np.allclose(transformed_extent_3d["z"], extent_3d["z"])


def test_get_extent_affine_sdata():
    # let's make a copy since we don't want to modify the original data
    sdata2 = SpatialData(
        shapes={
            "circles": _deepcopy(sdata["blobs_circles"]),
            "polygons": _deepcopy(sdata["blobs_polygons"]),
        }
    )
    translation0 = Translation([10], axes=("x",))
    translation1 = Translation([1000], axes=("x",))
    set_transformation(sdata2["circles"], translation0, to_coordinate_system="global")
    set_transformation(sdata2["polygons"], translation1, to_coordinate_system="translated")
    remove_transformation(sdata2["polygons"], to_coordinate_system="global")

    extent0 = get_extent(sdata2)
    extent1 = get_extent(sdata2, coordinate_system="translated")

    min_coordinates0 = np.array([98.92618679, 137.62348969]) + np.array([10.0, 0.0])
    max_coordinates0 = np.array([420.42236303, 422.31870626]) + np.array([10.0, 0.0])
    min_coordinates1 = np.array([149.92618679, 188.62348969]) + np.array([1000.0, 0.0])
    max_coordinates1 = np.array([446.70264371, 461.85209239]) + np.array([1000.0, 0.0])

    check_test_results0(
        extent0,
        min_coordinates=min_coordinates0,
        max_coordinates=max_coordinates0,
        axes=("x", "y"),
    )

    check_test_results0(
        extent1,
        min_coordinates=min_coordinates1,
        max_coordinates=max_coordinates1,
        axes=("x", "y"),
    )


def test_bug_get_extent_swap_xy_for_images():
    # https://github.com/scverse/spatialdata/issues/335#issue-1842914360
    x = RNG.random((1, 10, 20))
    im = Image2DModel.parse(x, dims=("c", "x", "y"))
    extent = get_extent(im)
    check_test_results1(extent, {"x": (0.0, 10.0), "y": (0.0, 20.0)})
