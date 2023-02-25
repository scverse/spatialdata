from dataclasses import FrozenInstanceError

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import linearrings, polygons
from spatial_image import SpatialImage

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
    bounding_box_query,
)
from spatialdata._core._spatialdata_ops import (
    get_transformation,
    remove_transformation,
    set_transformation,
)
from spatialdata._core.transformations import Affine, Scale, Sequence


def _make_points_element():
    """Helper function to make a Points element."""
    coordinates = np.array([[10, 10], [20, 20], [20, 30]], dtype=float)
    return PointsModel.parse(
        coordinates, annotation=pd.DataFrame({"genes": np.repeat("a", len(coordinates))}), feature_key="genes"
    )


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
    points_element = _make_points_element()
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
    points_element = _make_points_element()
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


# TODO: more tests can be added for spatial queries after the cases 2, 3, 4 are implemented (see https://github.com/scverse/spatialdata/pull/151, also for details on more tests)


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
    result = bounding_box_query(full_sdata, **request.to_dict())
    result2 = full_sdata.query(request)
    from tests._core.test_spatialdata_operations import (
        _assert_spatialdata_objects_seem_identical,
    )

    _assert_spatialdata_objects_seem_identical(result, result2)

    for element in result._gen_elements_values():
        d = get_transformation(element, get_all=True)
        new_d = {k.replace("global", "cropped"): v for k, v in d.items()}
        set_transformation(element, new_d, set_all=True)

    VISUALIZE = False
    if VISUALIZE:
        from napari_spatialdata import Interactive

        Interactive([full_sdata, result])


def _visualize_crop_affine_labels_2d():
    """
    This examples show how the bounding box spatial query works for data that has been rotated.

    Notes
    -----
    The bounding box query gives the data, from the intrinsic coordinate system, that is inside the bounding box of
    the inverse-transformed query bounding box.
    In this example I show this data, and I also show how to obtain the data back inside the original bounding box.

    To undertand the example I suggest to run it and then:
    1) select the "rotated" coordinate system from napari
    2) disable all the layers but "0 original"
    3) then enable "1 cropped global", this shows the data in the extrinsic coordinate system we care ("rotated"),
    and the bounding box we want to query
    4) then enable "2 cropped rotated", this show the data that has been queries (this is a bounding box of the
    requested crop, as exaplained above)
    5) then enable "3 cropped rotated processed", this shows the data that we wanted to query in the first place,
    in the target coordinate system ("rotated"). This is probaly the data you care about if for instance you want to
    use tiles for deep learning.
    6) Note that for obtaning the previous answer there is also a better function rasterize().
    This is what "4 rasterized" shows, which is faster and more accurate, so it should be used instead. The function
    rasterize() transforms all the coordinates of the data into the target coordinate system, and it returns only
    SpatialImage objects. So it has different use cases than the bounding box query.
    7) finally switch to the "global" coordinate_system. This is, for how we constructed the example, showing the
    original image as it would appear its intrinsic coordinate system (since the transformation that maps the
    original image to "global" is an identity. It then shows how the data showed at the point 5), localizes in the
    original image.
    """
    ##
    # in this test let's try some affine transformations, we could do that also for the other tests
    image = np.random.randint(low=10, high=100, size=(100, 100))
    # y: [5, 9], x: [0, 4] has value 1
    image[50:, :50] = 2
    labels_element = Labels2DModel.parse(image)
    affine = Affine(
        np.array(
            [
                [np.cos(np.pi / 6), np.sin(-np.pi / 6), 20],
                [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    set_transformation(
        labels_element,
        affine,
        "rotated",
    )

    # bounding box: y: [5, 9], x: [0, 4]
    labels_result_rotated = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="rotated",
    )
    labels_result_global = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="global",
    )
    from napari_spatialdata import Interactive

    from spatialdata import SpatialData

    old_transformation = get_transformation(labels_result_global, "global")
    remove_transformation(labels_result_global, "global")
    set_transformation(labels_result_global, old_transformation, "rotated")
    d = {
        "1 cropped_global": labels_result_global,
        "0 original": labels_element,
    }
    if labels_result_rotated is not None:
        d["2 cropped_rotated"] = labels_result_rotated

        transform = labels_result_rotated.attrs["transform"]["rotated"]
        transform_rotated_processed = transform.transform(labels_result_rotated, maintain_positioning=True)
        transform_rotated_processed_recropped = bounding_box_query(
            transform_rotated_processed,
            axes=("y", "x"),
            min_coordinate=np.array([25, 25]),
            max_coordinate=np.array([75, 100]),
            target_coordinate_system="rotated",
        )
        d["3 cropped_rotated_processed_recropped"] = transform_rotated_processed_recropped
        remove_transformation(labels_result_rotated, "global")

    multiscale_image = np.random.randint(low=10, high=100, size=(400, 400))
    multiscale_image[200:, :200] = 2
    # multiscale_labels = Labels2DModel.parse(multiscale_image)
    multiscale_labels = Labels2DModel.parse(multiscale_image, scale_factors=[2, 2, 2, 2])
    sequence = Sequence([Scale([0.5, 0.5], axes=("x", "y")), affine])
    set_transformation(multiscale_labels, sequence, "rotated")

    from spatialdata._core._rasterize import rasterize

    rasterized = rasterize(
        multiscale_labels,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="rotated",
        target_width=300,
    )
    d["4 rasterized"] = rasterized

    sdata = SpatialData(labels=d)
    Interactive(sdata)
    ##


if __name__ == "__main__":
    _visualize_crop_affine_labels_2d()
