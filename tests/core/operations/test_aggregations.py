from typing import Optional

import numpy as np
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from geopandas import GeoDataFrame
from numpy.random import default_rng
from spatialdata import SpatialData
from spatialdata._core.operations.aggregate import aggregate
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel

RNG = default_rng(42)


def _parse_shapes(
    sdata_query_aggregation: SpatialData, by_shapes: Optional[str] = None, values_shapes: Optional[str] = None
) -> GeoDataFrame:
    # only one between by_shapes and values_shapes can be None
    assert by_shapes is None or values_shapes is None
    assert by_shapes is not None or values_shapes is not None

    if by_shapes is not None:
        assert by_shapes in ["by_circles", "by_polygons"]
        return sdata_query_aggregation[by_shapes]
    if values_shapes is not None:
        assert values_shapes in ["values_circles", "values_polygons"]
        return sdata_query_aggregation[values_shapes]
    raise ValueError("by_shapes and values_shapes cannot be both None")


@pytest.mark.parametrize("by_shapes", ["by_circles", "by_polygons"])
# @pytest.mark.parametrize("value_key", ["categorical_in_ddf", "numerical_in_ddf"])
@pytest.mark.parametrize("value_key", ["numerical_in_ddf"])
def test_aggregate_points_by_shapes(sdata_query_aggregation, by_shapes: str, value_key: str) -> None:
    sdata = sdata_query_aggregation
    _parse_shapes(sdata, by_shapes=by_shapes)
    points = sdata["points"]
    shapes = sdata[by_shapes]
    result_adata = aggregate(values=points, by=shapes, value_key=value_key, agg_func="sum")

    if by_shapes == "by_circles":
        assert result_adata.obs_names.to_list() == ["0", "1"]
    else:
        assert result_adata.obs_names.to_list() == ["0", "1", "2", "3", "4"]

    if value_key == "categorical_in_ddf":
        assert result_adata.var_names.to_list() == ["a", "b", "c"]
        if by_shapes == "by_circles":
            np.testing.assert_equal(result_adata.X.A, np.array([[3, 3, 0], [0, 0, 0]]))
        else:
            np.testing.assert_equal(result_adata.X.A, np.array([[3, 2, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]]))
    else:
        assert result_adata.var_names.to_list() == ["numerical_in_ddf"]
        if by_shapes == "by_circles":
            np.testing.assert_equal(result_adata.X.A, np.array([[1.841450277084701], [0]]))
        else:
            np.testing.assert_equal(
                result_adata.X.A, np.array([[3.579436217876709], [0], [0], [0], [0.440377154715784]])
            )

    # id_key can be implicit for points
    points.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY] = value_key
    result_adata_implicit = aggregate(values=points, by=shapes, agg_func="sum")
    assert_equal(result_adata, result_adata_implicit)

    # in the categorical case, check that sum and count behave the same
    result_adata_count = aggregate(values=points, by=shapes, value_key=value_key, agg_func="count")
    assert_equal(result_adata, result_adata_count)

    # querying multiple values at the same time
    points["another_" + value_key] = points[value_key]
    new_value_key = [value_key, "another_" + value_key]
    if value_key == "categorical_in_ddf":
        with pytest.raises(ValueError):
            aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum")
    else:
        with pytest.raises(ValueError):
            aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum")


# def test_aggregate_points_by_circles_categorical(sdata_query_aggregation) -> None:
#     sdata = sdata_query_aggregation
#     # checks also that cound and sum behave the same for categorical variables
#     adata0 = aggregate(
#         values=sdata["points"],
#         by=sdata["by_circles"],
#         id_key="genes",
#         agg_func="count",
#         target_coordinate_system="global",
#     )
#     adata1 = aggregate(
#         values=sdata["points"],
#         by=sdata["by_circles"],
#         id_key="genes",
#         agg_func="sum",
#         target_coordinate_system="global",
#     )
#
#     assert adata0.var_names.tolist() == ["a", "b"]
#     assert adata1.var_names.tolist() == ["a", "b"]
#     X0 = adata0.X.todense()
#     X1 = adata1.X.todense()
#
#     assert np.all(np.matrix([[3, 3], [0, 0]]) == X0)
#     assert np.all(np.matrix([[3, 3], [0, 0]]) == X1)
#


@pytest.mark.parametrize("by_shapes", ["by_circles", "by_polygons"])
@pytest.mark.parametrize("values_shapes", ["values_circles", "values_polygons"])
@pytest.mark.parametrize(
    "value_key",
    [
        "categorical_in_var",
        "numerical_in_var",
        "categorical_in_obs",
        "numerical_in_obs",
        "categorical_in_gdf",
        "numerical_in_gdf",
    ],
)
def test_aggregate_shapes_by_shapes(
    sdata_query_aggregation, by_shapes: str, values_shapes: str, value_key: str
) -> None:
    sdata = sdata_query_aggregation
    _parse_shapes(sdata, by_shapes=by_shapes)
    _parse_shapes(sdata, values_shapes=values_shapes)
    pass


# def test_aggregate_polygons_by_polygons_categorical() -> None:
#     cellular = ShapesModel.parse(
#         gpd.GeoDataFrame(
#             geometry=[
#                 shapely.Polygon([(0.5, 7.0), (4.0, 2.0), (5.0, 8.0)]),
#                 shapely.Polygon([(3.0, 8.0), (7.0, 2.0), (10.0, 6.0), (7.0, 10.0)]),
#             ],
#             index=["shape_0", "shape_1"],
#         )
#     )
#     subcellular = ShapesModel.parse(
#         gpd.GeoDataFrame(
#             {"structure": pd.Categorical.from_codes([0, 0, 0, 1, 1, 1, 1], ["nucleus", "mitochondria"])},
#             index=[f"shape_{i}" for i in range(1, 8)],
#         ).set_geometry(
#             gpd.points_from_xy([1.2, 2.3, 4.1, 6.0, 6.1, 8.0, 9.0], [3.5, 4.8, 7.5, 4.0, 9.0, 5.5, 9.8]).buffer(0.1)
#         )
#     )
#
#     result_adata = aggregate(subcellular, cellular, "structure", agg_func="sum")
#     assert result_adata.obs_names.to_list() == ["shape_0", "shape_1"]
#     assert result_adata.var_names.to_list() == ["nucleus", "mitochondria"]
#     np.testing.assert_equal(result_adata.X.A, np.array([[2, 0], [1, 3]]))
#
#     with pytest.raises(ValueError):
#         aggregate(subcellular, cellular, agg_func="mean")
#
# def test_aggregate_circles_by_polygons() -> None:
#     # Basically the same as above, but not buffering explicitly
#     cellular = ShapesModel.parse(
#         gpd.GeoDataFrame(
#             geometry=[
#                 shapely.Polygon([(0.5, 7.0), (4.0, 2.0), (5.0, 8.0)]),
#                 shapely.Polygon([(3.0, 8.0), (7.0, 2.0), (10.0, 6.0), (7.0, 10.0)]),
#             ],
#             index=["shape_0", "shape_1"],
#         )
#     )
#     subcellular = ShapesModel.parse(
#         gpd.GeoDataFrame(
#             {
#                 "structure": pd.Categorical.from_codes([0, 0, 0, 1, 1, 1, 1], ["nucleus", "mitochondria"]),
#                 "radius": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#             },
#             index=[f"shape_{i}" for i in range(1, 8)],
#         ).set_geometry(gpd.points_from_xy([1.2, 2.3, 4.1, 6.0, 6.1, 8.0, 9.0], [3.5, 4.8, 7.5, 4.0, 9.0, 5.5, 9.8]))
#     )
#
#     result_adata = aggregate(subcellular, cellular, "structure", agg_func="sum")
#     assert result_adata.obs_names.to_list() == ["shape_0", "shape_1"]
#     assert result_adata.var_names.to_list() == ["nucleus", "mitochondria"]
#     np.testing.assert_equal(result_adata.X.A, np.array([[2, 0], [1, 3]]))


@pytest.mark.parametrize("image_schema", [Image2DModel])
@pytest.mark.parametrize("labels_schema", [Labels2DModel])
def test_aggregate_image_by_labels(labels_blobs, image_schema, labels_schema) -> None:
    image = RNG.normal(size=(3,) + labels_blobs.shape)

    image = image_schema.parse(image)
    labels = labels_schema.parse(labels_blobs)

    out = aggregate(image, labels)
    assert len(out) + 1 == len(np.unique(labels_blobs))
    assert isinstance(out, AnnData)
    np.testing.assert_array_equal(out.var_names, [f"channel_{i}_mean" for i in image.coords["c"].values])

    out = aggregate(image, labels, agg_func=["mean", "sum", "count"])
    assert len(out) + 1 == len(np.unique(labels_blobs))

    out = aggregate(image, labels, zone_ids=[1, 2, 3])
    assert len(out) == 3


def test_aggregate_spatialdata(sdata_blobs: SpatialData) -> None:
    sdata = sdata_blobs.aggregate(sdata_blobs.points["blobs_points"], by="blobs_polygons")
    assert isinstance(sdata, SpatialData)
    assert len(sdata.shapes["blobs_polygons"]) == 3
    assert sdata.table.shape == (3, 2)
    assert len(sdata.points["points"].compute()) == 300
