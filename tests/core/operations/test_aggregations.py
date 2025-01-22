import geopandas
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from geopandas import GeoDataFrame
from numpy.random import default_rng

from spatialdata import aggregate, to_polygons
from spatialdata._core._deepcopy import deepcopy as _deepcopy
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Affine, Identity, set_transformation

RNG = default_rng(42)


def _parse_shapes(
    sdata_query_aggregation: SpatialData, by_shapes: str | None = None, values_shapes: str | None = None
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

    # testing that we can call aggregate with the two equivalent syntaxes for the values argument
    result_adata = aggregate(values=points, by=shapes, value_key=value_key, agg_func="sum").tables["table"]
    result_adata_bis = aggregate(
        values_sdata=sdata, values="points", by=shapes, value_key=value_key, agg_func="sum", table_name="table"
    ).tables["table"]
    np.testing.assert_equal(result_adata.X.todense().A, result_adata_bis.X.todense().A)

    # check that the obs of aggregated values are correct
    if by_shapes == "by_circles":
        assert result_adata.obs_names.to_list() == ["0", "1"]
    else:
        assert result_adata.obs_names.to_list() == ["0", "1", "2", "3", "4"]

    # check that the aggregated values are correct
    if value_key == "categorical_in_ddf":
        assert result_adata.var_names.to_list() == ["a", "b", "c"]
        if by_shapes == "by_circles":
            np.testing.assert_equal(result_adata.X.todense().A, np.array([[3, 3, 0], [0, 0, 0]]))
        else:
            np.testing.assert_equal(
                result_adata.X.todense().A, np.array([[3, 2, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]])
            )
    else:
        assert result_adata.var_names.to_list() == ["numerical_in_ddf"]
        if by_shapes == "by_circles":
            s = points.compute().iloc[[0, 1, 2, 11, 12, 13]]["numerical_in_ddf"].sum()
            assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
        else:
            s0 = points.compute().iloc[[5, 6, 7, 16, 17, 18]]["numerical_in_ddf"].sum()
            s4 = points.compute().iloc[10]["numerical_in_ddf"]
            assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [s4]])))

    # id_key can be implicit for points
    points.attrs[PointsModel.ATTRS_KEY] = {}
    points.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY] = value_key
    result_adata_implicit = aggregate(values=points, by=shapes, agg_func="sum").tables["table"]
    assert_equal(result_adata, result_adata_implicit)

    # in the categorical case, check that sum and count behave the same
    if value_key == "categorical_in_ddf":
        result_adata_count = aggregate(values=points, by=shapes, value_key=value_key, agg_func="count").tables["table"]
        assert_equal(result_adata, result_adata_count)

    # querying multiple values at the same time
    new_value_key = [value_key, "another_" + value_key]
    if value_key == "categorical_in_ddf":
        # can't aggregate multiple categorical values
        with pytest.raises(ValueError):
            aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum")
    else:
        points["another_" + value_key] = points[value_key] + 10
        result_adata_multiple = aggregate(values=points, by=shapes, value_key=new_value_key, agg_func="sum").tables[
            "table"
        ]
        assert result_adata_multiple.var_names.to_list() == new_value_key
        if by_shapes == "by_circles":
            row = (
                points.compute()
                .iloc[[0, 1, 2, 11, 12, 13]][["numerical_in_ddf", "another_numerical_in_ddf"]]
                .sum()
                .tolist()
            )
            assert np.all(np.isclose(result_adata_multiple.X.todense().A, np.array([row, [0, 0]])))
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
            assert np.all(np.isclose(result_adata_multiple.X.todense().A, np.array([row0, row1, row2, row3, row4])))

    # test we can't aggregate from mixed categorical and numerical sources
    with pytest.raises(ValueError):
        aggregate(
            values=points,
            by=shapes,
            value_key=["numerical_in_ddf", "categorical_in_ddf"],
            agg_func="sum",
        )


# TODO: refactor in smaller functions for easier understanding
@pytest.mark.parametrize("by_shapes", ["by_circles", "by_polygons"])
@pytest.mark.parametrize("values_shapes", ["values_circles", "values_polygons"])
@pytest.mark.parametrize(
    "value_key",
    [
        "numerical_in_var",
        "numerical_in_obs",
        "numerical_in_gdf",
        "categorical_in_obs",
        "categorical_in_gdf",
    ],
)
def test_aggregate_shapes_by_shapes(
    sdata_query_aggregation, by_shapes: str, values_shapes: str, value_key: str
) -> None:
    sdata = sdata_query_aggregation
    by = _parse_shapes(sdata, by_shapes=by_shapes)
    values = _parse_shapes(sdata, values_shapes=values_shapes)

    result_adata = aggregate(
        values_sdata=sdata, values=values_shapes, by=by, value_key=value_key, agg_func="sum", table_name="table"
    ).tables["table"]

    # testing that we can call aggregate with the two equivalent syntaxes for the values argument (only relevant when
    # the values to aggregate are not in the table, for which only one of the two syntaxes is possible)
    if value_key.endswith("_in_gdf"):
        result_adata_bis = aggregate(values=values, by=by, value_key=value_key, agg_func="sum").tables["table"]
        np.testing.assert_equal(result_adata.X.todense().A, result_adata_bis.X.todense().A)

    # check that the obs of the aggregated values are correct
    if by_shapes == "by_circles":
        assert result_adata.obs_names.tolist() == ["0", "1"]
    else:
        assert result_adata.obs_names.tolist() == ["0", "1", "2", "3", "4"]

    # check that the aggregated values are correct
    if value_key == "numerical_in_var":
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                s = sdata.tables["table"][np.array([0, 1, 2, 3]), "numerical_in_var"].X.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = sdata.tables["table"][np.array([5, 6, 7, 8]), "numerical_in_var"].X.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [0]])))
        else:
            if by_shapes == "by_circles":
                s = sdata.tables["table"][np.array([9, 10, 11, 12]), "numerical_in_var"].X.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = sdata.tables["table"][np.array([14, 15, 16, 17]), "numerical_in_var"].X.sum()
                s1 = sdata.tables["table"][np.array([20]), "numerical_in_var"].X.sum()
                s2 = sdata.tables["table"][np.array([20]), "numerical_in_var"].X.sum()
                s3 = 0
                s4 = sdata.tables["table"][np.array([18, 19]), "numerical_in_var"].X.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [s1], [s2], [s3], [s4]])))
    elif value_key == "numerical_in_obs":
        # these cases are basically identically to the one above
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                s = sdata.tables["table"][np.array([0, 1, 2, 3]), :].obs["numerical_in_obs"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = sdata.tables["table"][np.array([5, 6, 7, 8]), :].obs["numerical_in_obs"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [0]])))
        else:
            if by_shapes == "by_circles":
                s = sdata.tables["table"][np.array([9, 10, 11, 12]), :].obs["numerical_in_obs"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = sdata.tables["table"][np.array([14, 15, 16, 17]), :].obs["numerical_in_obs"].sum()
                s1 = sdata.tables["table"][np.array([20]), :].obs["numerical_in_obs"].sum()
                s2 = sdata.tables["table"][np.array([20]), :].obs["numerical_in_obs"].sum()
                s3 = 0
                s4 = sdata.tables["table"][np.array([18, 19]), :].obs["numerical_in_obs"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [s1], [s2], [s3], [s4]])))
    elif value_key == "numerical_in_gdf":
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                s = values.iloc[np.array([0, 1, 2, 3])]["numerical_in_gdf"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = values.iloc[np.array([5, 6, 7, 8])]["numerical_in_gdf"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [0]])))
        else:
            if by_shapes == "by_circles":
                s = values.iloc[np.array([0, 1, 2, 3])]["numerical_in_gdf"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = values.iloc[np.array([5, 6, 7, 8]), :]["numerical_in_gdf"].sum()
                s1 = values.iloc[np.array([11]), :]["numerical_in_gdf"].sum()
                s2 = values.iloc[np.array([11]), :]["numerical_in_gdf"].sum()
                s3 = 0
                s4 = values.iloc[np.array([9, 10]), :]["numerical_in_gdf"].sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [s1], [s2], [s3], [s4]])))
    elif value_key == "categorical_in_obs":
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[4.0, 0, 0], [0, 0, 0]])))
            else:
                assert np.all(
                    np.isclose(
                        result_adata.X.todense().A, np.array([[4.0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
                    )
                )
        else:
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[0, 4.0, 0], [0, 0, 0]])))
            else:
                assert np.all(
                    np.isclose(
                        result_adata.X.todense().A,
                        np.array([[0, 4.0, 0], [0, 0, 1.0], [0, 0, 1.0], [0, 0, 0], [0, 0, 2.0]]),
                    )
                )
    elif value_key == "categorical_in_gdf":
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[4.0], [0]])))
            else:
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[4.0], [0], [0], [0], [0]])))
        else:
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[4.0, 0], [0, 0]])))
            else:
                assert np.all(
                    np.isclose(result_adata.X.todense().A, np.array([[4.0, 0], [0, 1.0], [0, 1.0], [0, 0], [0, 2.0]]))
                )
    else:
        raise ValueError("Unexpected value key")

    # in the categorical case, check that sum and count behave the same
    if value_key in ["categorical_in_obs", "categorical_in_gdf"]:
        result_adata_count = aggregate(
            values_sdata=sdata, values=values_shapes, by=by, value_key=value_key, agg_func="count", table_name="table"
        ).tables["table"]
        assert_equal(result_adata, result_adata_count)

    # querying multiple values at the same time
    new_value_key = [value_key, "another_" + value_key]
    if value_key in ["categorical_in_obs", "categorical_in_gdf"]:
        # can't aggregate multiple categorical values
        with pytest.raises(ValueError):
            aggregate(
                values_sdata=sdata,
                values=values_shapes,
                by=by,
                value_key=new_value_key,
                agg_func="sum",
                table_name="table",
            )
    else:
        if value_key == "numerical_in_obs":
            sdata.tables["table"].obs["another_numerical_in_obs"] = 1.0
        elif value_key == "numerical_in_gdf":
            values["another_numerical_in_gdf"] = 1.0
        else:
            assert value_key == "numerical_in_var"
            new_var = pd.concat((sdata.tables["table"].var, pd.DataFrame(index=["another_numerical_in_var"])))
            new_x = np.concatenate((sdata.tables["table"].X, np.ones_like(sdata.tables["table"].X[:, :1])), axis=1)
            new_table = AnnData(X=new_x, obs=sdata.tables["table"].obs, var=new_var, uns=sdata.tables["table"].uns)
            del sdata.tables["table"]
            sdata.tables["table"] = new_table

        result_adata = aggregate(
            values_sdata=sdata, values=values_shapes, by=by, value_key=new_value_key, agg_func="sum", table_name="table"
        ).tables["table"]
        assert result_adata.var_names.to_list() == new_value_key

        # since we added only columns of 1., we just have 4 cases to check all the aggregations, and not 12 like before
        # (4 cases: 2 options for "values" and 2 options for "by")
        # (12 cases: as above and 3 options for "value_key")
        if values_shapes == "values_circles":
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A[:, 1], np.array([4.0, 0])))
            else:
                assert np.all(np.isclose(result_adata.X.todense().A[:, 1], np.array([4.0, 0, 0, 0, 0])))
        else:
            if by_shapes == "by_circles":
                assert np.all(np.isclose(result_adata.X.todense().A[:, 1], np.array([4.0, 0])))
            else:
                assert np.all(np.isclose(result_adata.X.todense().A[:, 1], np.array([4.0, 1, 1, 0, 2])))

        # test can't aggregate multiple values from mixed sources
        with pytest.raises(ValueError):
            value_keys = [
                ["numerical_values_in_obs", "numerical_values_in_var"],
                ["numerical_values_in_obs", "numerical_values_in_gdf"],
                ["numerical_values_in_var", "numerical_values_in_gdf"],
            ]
            for value_key in value_keys:
                aggregate(
                    values_sdata=sdata,
                    values=values_shapes,
                    by=by,
                    value_key=value_key,
                    agg_func="sum",
                    table_name="table",
                )
    # test we can't aggregate from mixed categorical and numerical sources (let's just test one case)
    with pytest.raises(ValueError):
        aggregate(
            values_sdata=sdata,
            values=values_shapes,
            by=by,
            value_key=["numerical_values_in_obs", "categorical_values_in_obs"],
            agg_func="sum",
            table_name="table",
        )


@pytest.mark.parametrize("image_schema", [Image2DModel])
@pytest.mark.parametrize("labels_schema", [Labels2DModel])
def test_aggregate_image_by_labels(labels_blobs, image_schema, labels_schema) -> None:
    image = RNG.normal(size=(3,) + labels_blobs.shape)

    image = image_schema.parse(image)
    labels = labels_schema.parse(labels_blobs)

    out_sdata = aggregate(values=image, by=labels, agg_func="mean", table_name="aggregation")
    out = out_sdata.tables["aggregation"]
    assert len(out) + 1 == len(np.unique(labels_blobs))
    assert isinstance(out, AnnData)
    np.testing.assert_array_equal(out.var_names, [f"channel_{i}_mean" for i in image.coords["c"].values])

    out = aggregate(values=image, by=labels, agg_func=["mean", "sum", "count"]).tables["table"]
    assert len(out) + 1 == len(np.unique(labels_blobs))

    out = aggregate(values=image, by=labels, zone_ids=[1, 2, 3]).tables["table"]
    assert len(out) == 3


@pytest.mark.parametrize("values", ["blobs_image", "blobs_points", "blobs_circles", "blobs_polygons"])
@pytest.mark.parametrize("by", ["blobs_labels", "blobs_circles", "blobs_polygons"])
def test_aggregate_requiring_alignment(sdata_blobs: SpatialData, values, by) -> None:
    if values == "blobs_image" or by == "blobs_labels" and not (values == "blobs_image" and by == "blobs_labels"):
        raise pytest.skip("Aggregation mixing raster and vector data is not currently supported.")
    values = sdata_blobs[values]
    by = sdata_blobs[by]
    if values is by:
        # warning: this will give problems when aggregation labels by labels (not supported yet), because of this: https://github.com/scverse/spatialdata/issues/269
        by = _deepcopy(by)
        assert by.attrs["transform"] is not values.attrs["transform"]

    sdata = SpatialData.init_from_elements({"values": values, "by": by})
    out0 = aggregate(values=values, by=by, agg_func="sum").tables["table"]

    theta = np.pi / 7
    affine = Affine(
        np.array(
            [
                [np.cos(theta), -np.sin(theta), 120],
                [np.sin(theta), np.cos(theta), -213],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )

    # by doesn't map to the "other" coordinate system
    set_transformation(values, affine, "other")
    with pytest.raises(ValueError):
        _ = aggregate(values=values, by=by, target_coordinate_system="other", agg_func="sum")

    # both values and by map to the "other" coordinate system, but they are not aligned
    set_transformation(by, Identity(), "other")
    out1 = aggregate(values=values, by=by, target_coordinate_system="other", agg_func="sum").tables["table"]
    assert not np.allclose(out0.X.todense().A, out1.X.todense().A)

    # both values and by map to the "other" coordinate system, and they are aligned
    set_transformation(by, affine, "other")
    out2 = aggregate(values=values, by=by, target_coordinate_system="other", agg_func="sum").tables["table"]
    assert np.allclose(out0.X.todense().A, out2.X.todense().A)

    # actually transforming the data still lead to a correct the result
    transformed_sdata = sdata.transform_to_coordinate_system("other")
    sdata2 = SpatialData.init_from_elements({"values": sdata["values"], "by": transformed_sdata["by"]})
    # let's take values from the original sdata (non-transformed but aligned to 'other'); let's take by from the
    # transformed sdata
    out3 = aggregate(values=sdata2["values"], by=sdata2["by"], target_coordinate_system="other", agg_func="sum").tables[
        "table"
    ]
    assert np.allclose(out0.X.todense().A, out3.X.todense().A)


@pytest.mark.parametrize("by_name", ["by_circles", "by_polygons"])
@pytest.mark.parametrize("values_name", ["values_circles", "values_polygons"])
@pytest.mark.parametrize(
    "value_key",
    [
        "numerical_in_gdf",
        "categorical_in_gdf",
    ],
)
def test_aggregate_considering_fractions_single_values(
    sdata_query_aggregation: SpatialData, by_name, values_name, value_key
) -> None:
    sdata = sdata_query_aggregation
    values = sdata[values_name]
    by = sdata[by_name]
    result_adata = aggregate(values=values, by=by, value_key=value_key, agg_func="sum", fractions=True).tables["table"]
    # to manually compute the fractions of overlap that we use to test that aggregate() works
    values = to_polygons(values)
    values["__index"] = values.index
    by = to_polygons(by)
    by["__index"] = by.index
    overlayed = geopandas.overlay(by, values, how="intersection")
    overlayed.index = overlayed["__index_2"]
    overlaps = overlayed.geometry.area
    full_areas = values.geometry.area
    overlaps = (overlaps / full_areas).dropna()
    if value_key == "numerical_in_gdf":
        if values_name == "values_circles":
            if by_name == "by_circles":
                s = (values.iloc[np.array([0, 1, 2, 3])]["numerical_in_gdf"] * overlaps).sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = (values.iloc[np.array([5, 6, 7, 8])]["numerical_in_gdf"] * overlaps).sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [0]])))
        else:
            if by_name == "by_circles":
                s = (values.iloc[np.array([0, 1, 2, 3])]["numerical_in_gdf"] * overlaps).sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s], [0]])))
            else:
                s0 = (values.iloc[np.array([5, 6, 7, 8]), :]["numerical_in_gdf"] * overlaps).dropna().sum()
                # I manually computed and verified the following two values (in the aggregation code they are handled by
                # the .groupby logic). s1 and s2 are the values of two distinct shapes in "by" that intersect with the
                # same shape in "values", so the code above "* ovrelaps" would not work as there are non-unique indices
                s1 = (values.iloc[np.array([11]), :]["numerical_in_gdf"] * 0.15).dropna().sum()
                s2 = (values.iloc[np.array([11]), :]["numerical_in_gdf"] * 0.225).dropna().sum()
                s3 = 0
                s4 = (values.iloc[np.array([9, 10]), :]["numerical_in_gdf"] * overlaps).dropna().sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [s1], [s2], [s3], [s4]])))
    else:
        assert value_key == "categorical_in_gdf"
        if values_name == "values_circles":
            if by_name == "by_circles":
                s0 = overlaps.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0]])))
            else:
                s0 = overlaps.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0], [0], [0], [0], [0]])))
        else:
            if by_name == "by_circles":
                s0 = overlaps.sum()
                assert np.all(np.isclose(result_adata.X.todense().A, np.array([[s0, 0], [0, 0]])))
            else:
                s0 = overlaps[[5, 6, 7, 8]].sum()
                s4 = overlaps[[9, 10]].sum()
                assert np.all(
                    np.isclose(result_adata.X.todense().A, np.array([[s0, 0], [0, 0.15], [0, 0.225], [0, 0], [0, s4]]))
                )
    # not adding these as tests because would need to change the mark.parametrize etc.
    # TODO: the image by labels case and the labels by labels case is not supported yet
    # TODO: the mixed cases raster by vector are not supported yet


@pytest.mark.parametrize("by_name", ["by_circles", "by_polygons"])
@pytest.mark.parametrize("values_name", ["values_circles", "values_polygons"])
@pytest.mark.parametrize(
    "value_key",
    [
        "numerical_in_gdf",
        "categorical_in_gdf",
    ],
)
def test_aggregate_considering_fractions_multiple_values(
    sdata_query_aggregation: SpatialData, by_name, values_name, value_key
) -> None:
    sdata = sdata_query_aggregation
    new_var = pd.concat((sdata.tables["table"].var, pd.DataFrame(index=["another_numerical_in_var"])))
    new_x = np.concatenate((sdata.tables["table"].X, np.ones_like(sdata.tables["table"].X[:, :1])), axis=1)
    new_table = AnnData(X=new_x, obs=sdata.tables["table"].obs, var=new_var, uns=sdata.tables["table"].uns)
    del sdata.tables["table"]
    sdata.tables["table"] = new_table
    out = aggregate(
        values_sdata=sdata,
        values="values_circles",
        by=sdata["by_circles"],
        value_key=["numerical_in_var", "another_numerical_in_var"],
        agg_func="sum",
        fractions=True,
        table_name="table",
    ).tables["table"]
    overlaps = np.array([0.655781239649211, 1.0000000000000002, 1.0000000000000004, 0.1349639285777728])
    row0 = np.sum(sdata.tables["table"].X[[0, 1, 2, 3], :] * overlaps.reshape(-1, 1), axis=0)
    assert np.all(np.isclose(out.X.todense().A, np.array([row0, [0, 0]])))


def test_aggregation_invalid_cases(sdata_query_aggregation):
    # invalid case: categorical points / shapes by shapes with agg_func = "mean"
    with pytest.raises(AssertionError):
        aggregate(
            values=sdata_query_aggregation["values_circles"],
            by=sdata_query_aggregation["by_circles"],
            value_key="categorical_in_gdf",
            agg_func="mean",
        )

    # invalid case: numerical points by shapes with fractions = True
    with pytest.raises(AssertionError):
        aggregate(
            values=sdata_query_aggregation["points"],
            by=sdata_query_aggregation["by_circles"],
            value_key="numerical_in_ddf",
            agg_func="sum",
            fractions=True,
        )

    # invalid case: categorical shapes by shapes with fractions = True and agg_func = "count"
    with pytest.raises(AssertionError):
        aggregate(
            values=sdata_query_aggregation["values_circles"],
            by=sdata_query_aggregation["by_circles"],
            value_key="categorical_in_gdf",
            agg_func="count",
            fractions=True,
        )


def test_aggregate_spatialdata(sdata_blobs: SpatialData) -> None:
    sdata0 = sdata_blobs.aggregate(values="blobs_points", by="blobs_polygons", agg_func="sum")
    sdata1 = sdata_blobs.aggregate(values=sdata_blobs["blobs_points"], by="blobs_polygons", agg_func="sum")
    sdata2 = sdata_blobs.aggregate(values="blobs_points", by=sdata_blobs["blobs_polygons"], agg_func="sum")
    sdata3 = sdata_blobs.aggregate(values=sdata_blobs["blobs_points"], by=sdata_blobs["blobs_polygons"], agg_func="sum")

    assert_equal(sdata0.tables["table"], sdata1.tables["table"])
    assert_equal(sdata2.tables["table"], sdata3.tables["table"])

    # in sdata2 the name of the "by" region was not passed, so a default one is used
    assert sdata2.tables["table"].obs["region"].value_counts()["by"] == 3
    # let's change it so we can make the objects comparable
    sdata2.tables["table"].obs["region"] = "blobs_polygons"
    sdata2.tables["table"].obs["region"] = sdata2.tables["table"].obs["region"].astype("category")
    sdata2.tables["table"].uns[TableModel.ATTRS_KEY]["region"] = "blobs_polygons"
    assert_equal(sdata0.tables["table"], sdata2.tables["table"])

    assert len(sdata0.shapes["blobs_polygons"]) == 3
    assert sdata0.tables["table"].shape == (3, 2)


def test_aggregate_deepcopy(sdata_blobs: SpatialData) -> None:
    sdata0 = sdata_blobs.aggregate(values="blobs_points", by="blobs_polygons", agg_func="sum")
    sdata1 = sdata_blobs.aggregate(values="blobs_points", by="blobs_polygons", agg_func="sum", deepcopy=False)

    assert sdata0["blobs_polygons"] is not sdata_blobs["blobs_polygons"]
    assert sdata1["blobs_polygons"] is sdata_blobs["blobs_polygons"]
