from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from anndata import AnnData
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata._core.query.spatial_query import (
    BaseSpatialRequest,
    BoundingBoxRequest,
    bounding_box_query,
    polygon_query,
)
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)

from tests.conftest import _make_points, _make_squares


# ---------------- test bounding box queries ---------------[
def test_bounding_box_request_immutable():
    """Test that the bounding box request is immutable."""
    request = BoundingBoxRequest(
        axes=("y", "x"),
        min_coordinate=np.array([0, 0]),
        max_coordinate=np.array([10, 10]),
        target_coordinate_system="global",
    )
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
        _ = BoundingBoxRequest(
            axes=("c", "x"),
            min_coordinate=np.array([0, 0]),
            max_coordinate=np.array([10, 10]),
            target_coordinate_system="global",
        )


def test_bounding_box_request_wrong_number_of_coordinates():
    """Requests which specify coordinates not consistent with the axes should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(
            axes=("y", "x"),
            min_coordinate=np.array([0, 0, 0]),
            max_coordinate=np.array([10, 10]),
            target_coordinate_system="global",
        )

    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(
            axes=("y", "x"),
            min_coordinate=np.array([0, 0]),
            max_coordinate=np.array([10, 10, 10]),
            target_coordinate_system="global",
        )

    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(
            axes=("y", "x"),
            min_coordinate=np.array([0, 0, 0]),
            max_coordinate=np.array([10, 10, 10]),
            target_coordinate_system="global",
        )


def test_bounding_box_request_wrong_coordinate_order():
    """Requests where the min coordinate is greater than the max coordinate should raise an error"""
    with pytest.raises(ValueError):
        _ = BoundingBoxRequest(
            axes=("y", "x"),
            min_coordinate=np.array([0, 10]),
            max_coordinate=np.array([10, 0]),
            target_coordinate_system="global",
        )


def test_bounding_box_points():
    """test the points bounding box_query"""
    points_element = _make_points(np.array([[10, 10], [20, 20], [20, 30]]))
    original_x = np.array(points_element["x"])
    original_y = np.array(points_element["y"])

    points_result = bounding_box_query(
        points_element,
        axes=("x", "y"),
        min_coordinate=np.array([18, 25]),
        max_coordinate=np.array([22, 35]),
        target_coordinate_system="global",
    )

    # Check that the correct point was selected
    np.testing.assert_allclose(points_result["x"].compute(), [20])
    np.testing.assert_allclose(points_result["y"].compute(), [30])

    # result should be valid points element
    PointsModel.validate(points_result)

    # original element should be unchanged
    np.testing.assert_allclose(points_element["x"].compute(), original_x)
    np.testing.assert_allclose(points_element["y"].compute(), original_y)


def test_bounding_box_points_no_points():
    """Points bounding box query with no points in range should
    return a points element with length 0.
    """
    points_element = _make_points(np.array([[10, 10], [20, 20], [20, 30]]))
    request = bounding_box_query(
        points_element,
        axes=("x", "y"),
        min_coordinate=np.array([40, 50]),
        max_coordinate=np.array([45, 55]),
        target_coordinate_system="global",
    )
    assert request is None


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_bounding_box_image_2d(n_channels):
    """Apply a bounding box to a 2D image"""
    image = np.zeros((n_channels, 10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[:, 5::, 0:5] = 1
    image_element = Image2DModel.parse(image)
    image_element_multiscale = Image2DModel.parse(image, scale_factors=[2, 2])

    for image in [image_element, image_element_multiscale]:
        # bounding box: y: [5, 10[, x: [0, 5[
        image_result = bounding_box_query(
            image,
            axes=("y", "x"),
            min_coordinate=np.array([5, 0]),
            max_coordinate=np.array([10, 5]),
            target_coordinate_system="global",
        )
        expected_image = np.ones((n_channels, 5, 5))  # c dimension is preserved
        if isinstance(image, SpatialImage):
            assert isinstance(image, SpatialImage)
            np.testing.assert_allclose(image_result, expected_image)
        elif isinstance(image, MultiscaleSpatialImage):
            assert isinstance(image_result, MultiscaleSpatialImage)
            v = image_result["scale0"].values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            np.testing.assert_allclose(xdata, expected_image)
        else:
            raise ValueError("Unexpected type")


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_bounding_box_image_3d(n_channels):
    """Apply a bounding box to a 3D image"""
    image = np.zeros((n_channels, 10, 10, 10))
    # z: [5, 9], y: [0, 4], x: [2, 6] has value 1
    image[:, 5::, 0:5, 2:7] = 1
    image_element = Image3DModel.parse(image)
    image_element_multiscale = Image3DModel.parse(image, scale_factors=[2, 2])

    for image in [image_element, image_element_multiscale]:
        # bounding box: z: [5, 10[, y: [0, 5[, x: [2, 7[
        image_result = bounding_box_query(
            image,
            axes=("z", "y", "x"),
            min_coordinate=np.array([5, 0, 2]),
            max_coordinate=np.array([10, 5, 7]),
            target_coordinate_system="global",
        )
        expected_image = np.ones((n_channels, 5, 5, 5))  # c dimension is preserved
        if isinstance(image, SpatialImage):
            assert isinstance(image, SpatialImage)
            np.testing.assert_allclose(image_result, expected_image)
        elif isinstance(image, MultiscaleSpatialImage):
            assert isinstance(image_result, MultiscaleSpatialImage)
            v = image_result["scale0"].values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            np.testing.assert_allclose(xdata, expected_image)
        else:
            raise ValueError("Unexpected type")


def test_bounding_box_labels_2d():
    """Apply a bounding box to a 2D label image"""
    # in this test let's try some affine transformations, we could do that also for the other tests
    image = np.zeros((10, 10))
    # y: [5, 9], x: [0, 4] has value 1
    image[5::, 0:5] = 1
    labels_element = Labels2DModel.parse(image)
    labels_element_multiscale = Labels2DModel.parse(image, scale_factors=[2, 2])

    for labels in [labels_element, labels_element_multiscale]:
        # bounding box: y: [5, 10[, x: [0, 5[
        labels_result = bounding_box_query(
            labels,
            axes=("y", "x"),
            min_coordinate=np.array([5, 0]),
            max_coordinate=np.array([10, 5]),
            target_coordinate_system="global",
        )
        expected_image = np.ones((5, 5))
        if isinstance(labels, SpatialImage):
            assert isinstance(labels, SpatialImage)
            np.testing.assert_allclose(labels_result, expected_image)
        elif isinstance(labels, MultiscaleSpatialImage):
            assert isinstance(labels_result, MultiscaleSpatialImage)
            v = labels_result["scale0"].values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            np.testing.assert_allclose(xdata, expected_image)
        else:
            raise ValueError("Unexpected type")


def test_bounding_box_labels_3d():
    """Apply a bounding box to a 3D label image"""
    image = np.zeros((10, 10, 10), dtype=int)
    # z: [5, 9], y: [0, 4], x: [2, 6] has value 1
    image[5::, 0:5, 2:7] = 1
    labels_element = Labels3DModel.parse(image)
    labels_element_multiscale = Labels3DModel.parse(image, scale_factors=[2, 2])

    for labels in [labels_element, labels_element_multiscale]:
        # bounding box: z: [5, 10[, y: [0, 5[, x: [2, 7[
        labels_result = bounding_box_query(
            labels,
            axes=("z", "y", "x"),
            min_coordinate=np.array([5, 0, 2]),
            max_coordinate=np.array([10, 5, 7]),
            target_coordinate_system="global",
        )
        expected_image = np.ones((5, 5, 5))
        if isinstance(labels, SpatialImage):
            assert isinstance(labels, SpatialImage)
            np.testing.assert_allclose(labels_result, expected_image)
        elif isinstance(labels, MultiscaleSpatialImage):
            assert isinstance(labels_result, MultiscaleSpatialImage)
            v = labels_result["scale0"].values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            np.testing.assert_allclose(xdata, expected_image)
        else:
            raise ValueError("Unexpected type")


# TODO: more tests can be added for spatial queries after the cases 2, 3, 4 are implemented
#  (see https://github.com/scverse/spatialdata/pull/151, also for details on more tests)


def test_bounding_box_polygons():
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])
    half_widths = [6] * 4
    sd_polygons = _make_squares(centroid_coordinates=centroids, half_widths=half_widths)

    polygons_result = bounding_box_query(
        sd_polygons,
        axes=("y", "x"),
        target_coordinate_system="global",
        min_coordinate=np.array([40, 40]),
        max_coordinate=np.array([100, 100]),
    )

    assert len(polygons_result) == 1
    assert polygons_result.index[0] == 3


def test_bounding_box_circles():
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])

    sd_circles = ShapesModel.parse(centroids, geometry=0, radius=10)

    circles_result = bounding_box_query(
        sd_circles,
        axes=("y", "x"),
        target_coordinate_system="global",
        min_coordinate=np.array([40, 40]),
        max_coordinate=np.array([100, 100]),
    )

    assert len(circles_result) == 1
    assert circles_result.index[0] == 3


def test_bounding_box_spatial_data(full_sdata):
    request = BoundingBoxRequest(
        target_coordinate_system="global",
        axes=("y", "x"),
        min_coordinate=np.array([2, 1]),
        max_coordinate=np.array([40, 60]),
    )
    result = bounding_box_query(full_sdata, **request.to_dict(), filter_table=True)
    # filter table is True by default when calling query(request)
    result2 = full_sdata.query(request, filter_table=True)
    from tests.core.operations.test_spatialdata_operations import (
        _assert_spatialdata_objects_seem_identical,
    )

    _assert_spatialdata_objects_seem_identical(result, result2)

    for element in result._gen_elements_values():
        d = get_transformation(element, get_all=True)
        new_d = {k.replace("global", "cropped"): v for k, v in d.items()}
        set_transformation(element, new_d, set_all=True)


def test_bounding_box_filter_table():
    coords0 = np.array([[10, 10], [20, 20]])
    coords1 = np.array([[30, 30]])
    circles0 = ShapesModel.parse(coords0, geometry=0, radius=1)
    circles1 = ShapesModel.parse(coords1, geometry=0, radius=1)
    table = AnnData(shape=(3, 0))
    table.obs["region"] = ["circles0", "circles0", "circles1"]
    table.obs["instance"] = [0, 1, 0]
    table = TableModel.parse(table, region=["circles0", "circles1"], region_key="region", instance_key="instance")
    sdata = SpatialData(shapes={"circles0": circles0, "circles1": circles1}, table=table)
    queried0 = sdata.query.bounding_box(
        axes=("y", "x"),
        min_coordinate=np.array([15, 15]),
        max_coordinate=np.array([25, 25]),
        filter_table=True,
        target_coordinate_system="global",
    )
    queried1 = sdata.query.bounding_box(
        axes=("y", "x"),
        min_coordinate=np.array([15, 15]),
        max_coordinate=np.array([25, 25]),
        filter_table=False,
        target_coordinate_system="global",
    )
    assert len(queried0.table) == 1
    assert len(queried1.table) == 3


# ----------------- test polygon query -----------------
def test_polygon_query_points(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    polygon = sdata["by_polygons"].geometry.iloc[0]
    queried = polygon_query(sdata, polygons=polygon, target_coordinate_system="global", shapes=False, points=True)
    points = queried["points"].compute()
    assert len(points) == 6
    assert len(queried.table) == 0

    # TODO: the case of querying points with multiple polygons is not currently implemented


def test_polygon_query_shapes(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    values_sdata = SpatialData(
        shapes={"values_polygons": sdata["values_polygons"], "values_circles": sdata["values_circles"]},
        table=sdata.table,
    )
    polygon = sdata["by_polygons"].geometry.iloc[0]
    circle = sdata["by_circles"].geometry.iloc[0]
    circle_pol = circle.buffer(sdata["by_circles"].radius.iloc[0])

    queried = polygon_query(
        values_sdata, polygons=polygon, target_coordinate_system="global", shapes=True, points=False
    )
    assert len(queried["values_polygons"]) == 4
    assert len(queried["values_circles"]) == 4
    assert len(queried.table) == 8

    queried = polygon_query(
        values_sdata, polygons=[polygon, circle_pol], target_coordinate_system="global", shapes=True, points=False
    )
    assert len(queried["values_polygons"]) == 8
    assert len(queried["values_circles"]) == 8
    assert len(queried.table) == 16

    queried = polygon_query(
        values_sdata, polygons=[polygon, polygon], target_coordinate_system="global", shapes=True, points=False
    )
    assert len(queried["values_polygons"]) == 4
    assert len(queried["values_circles"]) == 4
    assert len(queried.table) == 8

    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        queried.pl.render_shapes("values_polygons").pl.show(ax=ax)
        queried.pl.render_shapes("values_circles").pl.show(ax=ax)
        plt.show()


@pytest.mark.skip
def test_polygon_query_multipolygons():
    pass


def test_polygon_query_spatial_data(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    values_sdata = SpatialData(
        shapes={
            "values_polygons": sdata["values_polygons"],
            "values_circles": sdata["values_circles"],
        },
        points={"points": sdata["points"]},
        table=sdata.table,
    )
    polygon = sdata["by_polygons"].geometry.iloc[0]
    queried = polygon_query(values_sdata, polygons=polygon, target_coordinate_system="global", shapes=True, points=True)
    assert len(queried["values_polygons"]) == 4
    assert len(queried["values_circles"]) == 4
    assert len(queried["points"]) == 6
    assert len(queried.table) == 8


@pytest.mark.skip
def test_polygon_query_image2d():
    # single image case
    # multiscale case
    pass


@pytest.mark.skip
def test_polygon_query_image3d():
    # single image case
    # multiscale case
    pass


@pytest.mark.skip
def test_polygon_query_labels2d():
    # single image case
    # multiscale case
    pass


@pytest.mark.skip
def test_polygon_query_labels3d():
    # single image case
    # multiscale case
    pass
