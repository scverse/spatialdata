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
@pytest.mark.parametrize("value_key", ["categorical_in_ddf", "numerical_in_ddf"])
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
            s = points.compute().iloc[[0, 1, 2, 11, 12, 13]]["numerical_in_ddf"].sum()
            assert np.all(np.isclose(result_adata.X.A, np.array([[s], [0]])))
        else:
            s0 = points.compute().iloc[[5, 6, 7, 16, 17, 18]]["numerical_in_ddf"].sum()
            s4 = points.compute().iloc[10]["numerical_in_ddf"]
            assert np.all(np.isclose(result_adata.X.A, np.array([[s0], [0], [0], [0], [s4]])))

    # id_key can be implicit for points
    points.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY] = value_key
    result_adata_implicit = aggregate(values=points, by=shapes, agg_func="sum")
    assert_equal(result_adata, result_adata_implicit)

    # in the categorical case, check that sum and count behave the same
    result_adata_count = aggregate(values=points, by=shapes, value_key=value_key, agg_func="count")
    assert_equal(result_adata, result_adata_count)

    # querying multiple values at the same time
    new_value_key = [value_key, "another_" + value_key]
    if value_key == "categorical_in_ddf":
        with pytest.raises(ValueError):
            aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum")
    else:
        points["another_" + value_key] = points[value_key] + 10
        result_adata_multiple = aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum")
        assert result_adata_multiple.var_names.to_list() == new_value_key
        if by_shapes == "by_circles":
            row = (
                points.compute()
                .iloc[[0, 1, 2, 11, 12, 13]][["numerical_in_ddf", "another_numerical_in_ddf"]]
                .sum()
                .tolist()
            )
            assert np.all(np.isclose(result_adata_multiple.X.A, np.array([row, [0, 0]])))
        else:
            row0 = (
                points.compute()
                .iloc[[5, 6, 7, 16, 17, 18]][["numerical_in_ddf", "another_numerical_in_ddf"]]
                .sum()
                .tolist()
            )
            row1 = np.zeros(2)
            row2 = np.zeros(2)
            row3 = np.zeros(2)
            row4 = points.compute().iloc[10][["numerical_in_ddf", "another_numerical_in_ddf"]].tolist()
            assert np.all(np.isclose(result_adata_multiple.X.A, np.array([row0, row1, row2, row3, row4])))


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


@pytest.mark.parametrize("image_schema", [Image2DModel])
@pytest.mark.parametrize("labels_schema", [Labels2DModel])
def test_aggregate_image_by_labels(labels_blobs, image_schema, labels_schema) -> None:
    image = RNG.normal(size=(3,) + labels_blobs.shape)

    image = image_schema.parse(image)
    labels = labels_schema.parse(labels_blobs)

    out = aggregate(values=image, by=labels)
    assert len(out) + 1 == len(np.unique(labels_blobs))
    assert isinstance(out, AnnData)
    np.testing.assert_array_equal(out.var_names, [f"channel_{i}_mean" for i in image.coords["c"].values])

    out = aggregate(values=image, by=labels, agg_func=["mean", "sum", "count"])
    assert len(out) + 1 == len(np.unique(labels_blobs))

    out = aggregate(values=image, by=labels, zone_ids=[1, 2, 3])
    assert len(out) == 3


def test_aggregate_spatialdata(sdata_blobs: SpatialData) -> None:
    sdata = sdata_blobs.aggregate(sdata_blobs.points["blobs_points"], by="blobs_polygons")
    assert isinstance(sdata, SpatialData)
    assert len(sdata.shapes["blobs_polygons"]) == 3
    assert sdata.table.shape == (3, 2)
    assert len(sdata.points["points"].compute()) == 300
