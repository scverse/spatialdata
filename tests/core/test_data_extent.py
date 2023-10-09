import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from spatialdata import SpatialData, get_extent, transform
from spatialdata._utils import _deepcopy_geodataframe
from spatialdata.datasets import blobs
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Translation, remove_transformation, set_transformation

# for faster tests; we will pay attention not to modify the original data
sdata = blobs()


def check_test_results(extent, min_coordinates, max_coordinates, axes):
    assert np.allclose([extent["x"][0], extent["y"][0]], min_coordinates)
    assert np.allclose([extent["x"][1], extent["y"][1]], max_coordinates)
    extend_axes = list(extent.keys())
    extend_axes.sort()
    assert tuple(extend_axes) == axes


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

    check_test_results(
        extent,
        min_coordinates=min_coordinates,
        max_coordinates=max_coordinates,
        axes=("x", "y"),
    )


def test_get_extent_points():
    # 2d case
    extent = get_extent(sdata["blobs_points"])
    check_test_results(
        extent,
        min_coordinates=np.array([12.0, 13.0]),
        max_coordinates=np.array([500.0, 498.0]),
        axes=("x", "y"),
    )

    # 3d case
    data = np.array([[1, 2, 3], [4, 5, 6]])
    df = pd.DataFrame(data, columns=["zeta", "x", "y"])
    points_3d = PointsModel.parse(df, coordinates={"x": "x", "y": "y", "z": "zeta"})
    extent_3d = get_extent(points_3d)
    check_test_results(
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
    check_test_results(
        extent,
        min_coordinates=np.array([0, 0]),
        max_coordinates=np.array([512, 512]),
        axes=("x", "y"),
    )


def test_get_extent_spatialdata():
    sdata2 = SpatialData(shapes={"circles": sdata["blobs_circles"], "polygons": sdata["blobs_polygons"]})
    extent = get_extent(sdata2)
    check_test_results(
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


def test_get_extent_affine_circles():
    from tests.core.operations.test_transform import _get_affine

    affine = _get_affine(small_translation=True)

    # let's do a deepcopy of the circles since we don't want to modify the original data
    circles = _deepcopy_geodataframe(sdata["blobs_circles"])

    set_transformation(element=circles, transformation=affine, to_coordinate_system="transformed")

    extent = get_extent(circles)
    transformed_extent = get_extent(circles, coordinate_system="transformed")

    assert extent[2] == transformed_extent[2]
    assert not np.allclose(extent[0], transformed_extent[0])
    assert not np.allclose(extent[1], transformed_extent[1])

    min_coordinates, max_coordinates, axes = extent

    # Create a list of points
    points = [
        (min_coordinates[0], min_coordinates[1]),  # lower left corner
        (min_coordinates[0], max_coordinates[1]),  # upper left corner
        (max_coordinates[0], max_coordinates[1]),  # upper right corner
        (max_coordinates[0], min_coordinates[1]),  # lower right corner
        (min_coordinates[0], min_coordinates[1]),  # back to start to close the polygon
    ]

    # Create a Polygon from the points
    bounding_box = Polygon(points)
    gdf = GeoDataFrame(geometry=[bounding_box])
    gdf = ShapesModel.parse(gdf)
    transformed_bounding_box = transform(gdf, affine)

    min_coordinates0, max_coordinates0, axes0 = transformed_extent
    min_coordinates1, max_coordinates1, axes1 = get_extent(transformed_bounding_box)

    assert np.allclose(min_coordinates0, min_coordinates1)
    assert np.allclose(max_coordinates0, max_coordinates1)
    assert axes0 == axes1


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

    assert transformed_extent_2d[2] == ("x", "y")
    assert transformed_extent_3d[2] == ("x", "y", "z")

    # the x and y extent for the 2d and 3d points are identical
    assert np.allclose(transformed_extent_2d[0], transformed_extent_3d[0][:2])
    assert np.allclose(transformed_extent_2d[1], transformed_extent_3d[1][:2])

    # the z extent for the 3d points didn't get transformed, so it's the same as the original
    assert np.allclose(transformed_extent_3d[0][2], extent_3d[0][2])
    assert np.allclose(transformed_extent_3d[1][2], extent_3d[1][2])


def test_get_extent_affine_sdata():
    # let's make a copy since we don't want to modify the original data
    sdata2 = SpatialData(
        shapes={
            "circles": _deepcopy_geodataframe(sdata["blobs_circles"]),
            "polygons": _deepcopy_geodataframe(sdata["blobs_polygons"]),
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

    check_test_results(
        extent0,
        min_coordinates=min_coordinates0,
        max_coordinates=max_coordinates0,
        axes=("x", "y"),
    )

    check_test_results(
        extent1,
        min_coordinates=min_coordinates1,
        max_coordinates=max_coordinates1,
        axes=("x", "y"),
    )
