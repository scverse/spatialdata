from dataclasses import FrozenInstanceError

import dask.dataframe as dd
import geopandas.testing
import numpy as np
import pytest
import xarray
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import Polygon
from spatial_image import SpatialImage
from spatialdata._core.query.spatial_query import (
    BaseSpatialRequest,
    BoundingBoxRequest,
    bounding_box_query,
    polygon_query,
)
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import Identity, set_transformation

from tests.conftest import _make_points, _make_squares


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


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize("is_bb_3d", [True, False])
@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_points(is_3d: bool, is_bb_3d: bool, with_polygon_query: bool):
    """test the points bounding box_query"""
    data_x = np.array([10, 20, 20, 20])
    data_y = np.array([10, 20, 30, 30])
    data_z = np.array([100, 200, 200, 300])

    data = np.stack((data_x, data_y), axis=1)
    if is_3d:
        data = np.hstack((data, data_z.reshape(-1, 1)))
    points_element = _make_points(data)

    original_x = points_element["x"]
    original_y = points_element["y"]
    if is_3d:
        original_z = points_element["z"]

    if is_bb_3d:
        _min_coordinate = np.array([18, 25, 250])
        _max_coordinate = np.array([22, 35, 350])
        _axes = ("x", "y", "z")
    else:
        _min_coordinate = np.array([18, 25])
        _max_coordinate = np.array([22, 35])
        _axes = ("x", "y")

    if with_polygon_query:
        if is_bb_3d:
            return
        polygon = Polygon([(18, 25), (18, 35), (22, 35), (22, 25)])
        points_result = polygon_query(points_element, polygon=polygon, target_coordinate_system="global")
    else:
        points_result = bounding_box_query(
            points_element,
            axes=_axes,
            min_coordinate=_min_coordinate,
            max_coordinate=_max_coordinate,
            target_coordinate_system="global",
        )

    # Check that the correct point was selected
    if is_3d:
        if is_bb_3d:
            np.testing.assert_allclose(points_result["x"].compute(), [20])
            np.testing.assert_allclose(points_result["y"].compute(), [30])
            np.testing.assert_allclose(points_result["z"].compute(), [300])
        else:
            np.testing.assert_allclose(points_result["x"].compute(), [20, 20])
            np.testing.assert_allclose(points_result["y"].compute(), [30, 30])
            np.testing.assert_allclose(points_result["z"].compute(), [200, 300])
    else:
        np.testing.assert_allclose(points_result["x"].compute(), [20, 20])
        np.testing.assert_allclose(points_result["y"].compute(), [30, 30])

    # result should be valid points element
    PointsModel.validate(points_result)

    # original element should be unchanged
    np.testing.assert_allclose(points_element["x"].compute(), original_x)
    np.testing.assert_allclose(points_element["y"].compute(), original_y)
    if is_3d:
        np.testing.assert_allclose(points_element["z"].compute(), original_z)


def test_query_points_no_points():
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
@pytest.mark.parametrize("is_labels", [True, False])
@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize("is_bb_3d", [True, False])
@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_raster(n_channels: int, is_labels: bool, is_3d: bool, is_bb_3d: bool, with_polygon_query: bool):
    """Apply a bounding box to a raster element."""
    if is_labels and n_channels > 1:
        # labels cannot have multiple channels, let's ignore this combination of parameters
        return

    shape = (10, 10)
    if is_3d:
        shape = (10,) + shape
    shape = (n_channels,) + shape if not is_labels else (1,) + shape

    image = np.zeros(shape)
    axes = ["y", "x"]
    if is_3d:
        image[:, 2:7, 5::, 0:5] = 1
        axes = ["z"] + axes
    else:
        image[:, 5::, 0:5] = 1

    if is_labels:
        image = np.squeeze(image, axis=0)
    else:
        axes = ["c"] + axes

    ximage = xarray.DataArray(image, dims=axes)
    model = (
        Labels3DModel
        if is_labels and is_3d
        else Labels2DModel if is_labels else Image3DModel if is_3d else Image2DModel
    )

    image_element = model.parse(image)
    image_element_multiscale = model.parse(image, scale_factors=[2, 2])

    images = [image_element, image_element_multiscale]

    for image in images:
        if is_bb_3d:
            _min_coordinate = np.array([2, 5, 0])
            _max_coordinate = np.array([7, 10, 5])
            _axes = ("z", "y", "x")
        else:
            _min_coordinate = np.array([5, 0])
            _max_coordinate = np.array([10, 5])
            _axes = ("y", "x")

        if with_polygon_query:
            if is_bb_3d:
                return
            # make a triangle whose bounding box is the same as the bounding box specified with the query
            polygon = Polygon([(0, 5), (5, 5), (5, 10)])
            image_result = polygon_query(image, polygon=polygon, target_coordinate_system="global")
        else:
            image_result = bounding_box_query(
                image,
                axes=_axes,
                min_coordinate=_min_coordinate,
                max_coordinate=_max_coordinate,
                target_coordinate_system="global",
            )

        slices = {"y": slice(5, 10), "x": slice(0, 5)}
        if is_bb_3d and is_3d:
            slices["z"] = slice(2, 7)
        expected_image = ximage.sel(**slices)

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


@pytest.mark.parametrize("is_bb_3d", [True, False])
@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_polygons(is_bb_3d: bool, with_polygon_query: bool):
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])
    half_widths = [6] * 4
    sd_polygons = _make_squares(centroid_coordinates=centroids, half_widths=half_widths)

    if with_polygon_query:
        if is_bb_3d:
            return
        polygon = Polygon([(40, 40), (40, 100), (100, 100), (100, 40)])
        polygons_result = polygon_query(
            sd_polygons,
            polygon=polygon,
            target_coordinate_system="global",
        )
    else:
        if is_bb_3d:
            _min_coordinate = np.array([2, 40, 40])
            _max_coordinate = np.array([7, 100, 100])
            _axes = ("z", "y", "x")
        else:
            _min_coordinate = np.array([40, 40])
            _max_coordinate = np.array([100, 100])
            _axes = ("y", "x")

        polygons_result = bounding_box_query(
            sd_polygons,
            axes=_axes,
            target_coordinate_system="global",
            min_coordinate=_min_coordinate,
            max_coordinate=_max_coordinate,
        )

    assert len(polygons_result) == 1
    assert polygons_result.index[0] == 3


@pytest.mark.parametrize("is_bb_3d", [True, False])
@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_circles(is_bb_3d: bool, with_polygon_query: bool):
    centroids = np.array([[10, 10], [10, 80], [80, 20], [70, 60]])

    sd_circles = ShapesModel.parse(centroids, geometry=0, radius=10)

    if with_polygon_query:
        if is_bb_3d:
            return
        polygon = Polygon([(40, 40), (40, 100), (100, 100), (100, 40)])
        circles_result = polygon_query(
            sd_circles,
            polygon=polygon,
            target_coordinate_system="global",
        )
    else:
        if is_bb_3d:
            _min_coordinate = np.array([2, 40, 40])
            _max_coordinate = np.array([7, 100, 100])
            _axes = ("z", "y", "x")
        else:
            _min_coordinate = np.array([40, 40])
            _max_coordinate = np.array([100, 100])
            _axes = ("y", "x")

        circles_result = bounding_box_query(
            sd_circles,
            axes=_axes,
            target_coordinate_system="global",
            min_coordinate=_min_coordinate,
            max_coordinate=_max_coordinate,
        )

    assert len(circles_result) == 1
    assert circles_result.index[0] == 3


def test_query_spatial_data(full_sdata):
    request = BoundingBoxRequest(
        target_coordinate_system="global",
        axes=("y", "x"),
        min_coordinate=np.array([2, 1]),
        max_coordinate=np.array([40, 60]),
    )
    result0 = bounding_box_query(full_sdata, **request.to_dict(), filter_table=True)
    # filter table is True by default when calling query(request)
    result1 = full_sdata.query(request, filter_table=True)
    result2 = full_sdata.query.bounding_box(**request.to_dict(), filter_table=True)

    assert_spatial_data_objects_are_identical(result0, result1)
    assert_spatial_data_objects_are_identical(result0, result2)

    polygon = Polygon([(1, 2), (60, 2), (60, 40), (1, 40)])
    result3 = polygon_query(full_sdata, polygon=polygon, target_coordinate_system="global", filter_table=True)
    result4 = full_sdata.query.polygon(polygon=polygon, target_coordinate_system="global", filter_table=True)

    assert_spatial_data_objects_are_identical(result0, result3, check_transformations=False)
    assert_spatial_data_objects_are_identical(result0, result4, check_transformations=False)


@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_filter_table(with_polygon_query: bool):
    coords0 = np.array([[10, 10], [20, 20]])
    coords1 = np.array([[30, 30]])
    circles0 = ShapesModel.parse(coords0, geometry=0, radius=1)
    circles1 = ShapesModel.parse(coords1, geometry=0, radius=1)
    table = AnnData(shape=(3, 0))
    table.obs["region"] = ["circles0", "circles0", "circles1"]
    table.obs["instance"] = [0, 1, 0]
    table = TableModel.parse(table, region=["circles0", "circles1"], region_key="region", instance_key="instance")
    sdata = SpatialData(shapes={"circles0": circles0, "circles1": circles1}, table=table)

    if with_polygon_query:
        polygon = Polygon([(15, 15), (15, 25), (25, 25), (25, 15)])
        queried0 = polygon_query(
            sdata,
            polygon=polygon,
            target_coordinate_system="global",
            filter_table=True,
        )
        queried1 = polygon_query(
            sdata,
            polygon=polygon,
            target_coordinate_system="global",
            filter_table=False,
        )
    else:
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

    assert len(queried0["table"]) == 1
    assert len(queried1["table"]) == 3


def test_polygon_query_with_multipolygon(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    values_sdata = SpatialData(
        shapes={"values_polygons": sdata["values_polygons"], "values_circles": sdata["values_circles"]},
        tables=sdata["table"],
    )
    polygon = sdata["by_polygons"].geometry.iloc[0]
    circle = sdata["by_circles"].geometry.iloc[0]
    circle_pol = circle.buffer(sdata["by_circles"].radius.iloc[0])

    queried = polygon_query(
        values_sdata,
        polygon=polygon,
        target_coordinate_system="global",
        shapes=True,
        points=False,
    )
    assert len(queried["values_polygons"]) == 4
    assert len(queried["values_circles"]) == 4
    assert len(queried["table"]) == 8

    multipolygon = GeoDataFrame(geometry=[polygon, circle_pol]).unary_union
    queried = polygon_query(values_sdata, polygon=multipolygon, target_coordinate_system="global")
    assert len(queried["values_polygons"]) == 8
    assert len(queried["values_circles"]) == 8
    assert len(queried["table"]) == 16

    multipolygon = GeoDataFrame(geometry=[polygon, polygon]).unary_union
    queried = polygon_query(values_sdata, polygon=multipolygon, target_coordinate_system="global")
    assert len(queried["values_polygons"]) == 4
    assert len(queried["values_circles"]) == 4
    assert len(queried["table"]) == 8

    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        queried.pl.render_shapes("values_polygons").pl.show(ax=ax)
        queried.pl.render_shapes("values_circles").pl.show(ax=ax)
        plt.show()


@pytest.mark.parametrize("with_polygon_query", [False, True])
@pytest.mark.parametrize("name", ["image2d", "labels2d", "points_0", "circles", "multipoly", "poly"])
def test_query_affine_transformation(full_sdata, with_polygon_query: bool, name: str):
    from spatialdata import transform
    from spatialdata.transformations import Affine, set_transformation

    sdata = full_sdata.subset([name])

    theta = np.pi / 6
    t = Affine(
        np.array(
            [
                [np.cos(theta), -np.sin(theta), 100],
                [np.sin(theta), np.cos(theta), -50],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    set_transformation(sdata[name], transformation=t, to_coordinate_system="aligned")

    x0 = 99
    x1 = 101
    y0 = -51
    y1 = -46

    polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    back_polygon = transform(
        ShapesModel.parse(GeoDataFrame(geometry=[polygon]), transformations={"inverse": t.inverse()}),
        to_coordinate_system="inverse",
    ).geometry.iloc[0]

    def _query(
        sdata: SpatialData, polygon: Polygon, target_coordinate_system: str, with_polygon_query: bool
    ) -> SpatialData:
        px0, py0, px1, py1 = polygon.bounds
        if with_polygon_query:
            return polygon_query(sdata, polygon=polygon, target_coordinate_system=target_coordinate_system)
        return bounding_box_query(
            sdata,
            axes=("x", "y"),
            target_coordinate_system=target_coordinate_system,
            min_coordinate=[px0, py0],
            max_coordinate=[px1, py1],
        )

    queried = _query(sdata, polygon=polygon, target_coordinate_system="aligned", with_polygon_query=with_polygon_query)
    queried_back = _query(
        sdata, polygon=back_polygon, target_coordinate_system="global", with_polygon_query=with_polygon_query
    )
    queried_back_vector = _query(
        sdata, polygon=back_polygon, target_coordinate_system="global", with_polygon_query=True
    )

    if name in ["image2d", "labels2d"]:
        assert np.array_equal(queried[name], queried_back[name])
    elif name in ["points_0"]:
        assert dd.assert_eq(queried[name], queried_back_vector[name])
    elif name in ["circles", "multipoly", "poly"]:
        geopandas.testing.assert_geodataframe_equal(queried[name], queried_back[name])


@pytest.mark.parametrize("with_polygon_query", [True, False])
def test_query_points_multiple_partitions(points, with_polygon_query: bool):
    p0 = points["points_0"]
    p1 = PointsModel.parse(dd.from_pandas(p0.compute(), npartitions=10))

    def _query(p: DaskDataFrame) -> DaskDataFrame:
        if with_polygon_query:
            polygon = Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])
            return polygon_query(
                p,
                polygon=polygon,
                target_coordinate_system="global",
            )
        return bounding_box_query(
            p,
            axes=("x", "y"),
            target_coordinate_system="global",
            min_coordinate=[-1, -1],
            max_coordinate=[1, 1],
        )

    q0 = _query(p0)
    q1 = _query(p1)
    assert np.array_equal(q0.index.compute(), q1.index.compute())
    pass


@pytest.mark.parametrize("with_polygon_query", [True, False])
@pytest.mark.parametrize("name", ["image2d", "labels2d", "points_0", "circles", "multipoly", "poly"])
def test_attributes_are_copied(full_sdata, with_polygon_query: bool, name: str):
    """Test that attributes are copied over to the new spatial data object."""
    sdata = full_sdata.subset([name])

    # let's add a second transformation, to make sure that later we are not checking for the presence of default values
    set_transformation(sdata[name], transformation=Identity(), to_coordinate_system="aligned")

    old_attrs = sdata[name].attrs
    old_transform = sdata[name].attrs["transform"]

    old_attrs_value = old_attrs.copy()
    old_transform_value = old_transform.copy()

    if with_polygon_query:
        queried = polygon_query(
            sdata,
            polygon=Polygon([(-1, -1), (-1, 1), (1, 1), (1, -1)]),
            target_coordinate_system="aligned",
        )
    else:
        queried = bounding_box_query(
            sdata,
            axes=("x", "y"),
            target_coordinate_system="aligned",
            min_coordinate=[-1, -1],
            max_coordinate=[1, 1],
        )

    # check that the old attribute didn't change, neither in reference nor in value
    assert sdata[name].attrs is old_attrs
    assert sdata[name].attrs["transform"] is old_transform

    assert sdata[name].attrs == old_attrs_value
    assert sdata[name].attrs["transform"] == old_transform_value

    # check that the attributes of the queried element are not the same as the old ones
    assert sdata[name].attrs is not queried[name].attrs
    assert sdata[name].attrs["transform"] is not queried[name].attrs["transform"]
    pass
