import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import get_values, match_table_to_element
from spatialdata._core.query.relational_query import _locate_value, _ValueOrigin, join_sdata_spatialelement_table
from spatialdata.models.models import TableModel


def test_match_table_to_element(sdata_query_aggregation):
    # table can't annotate points
    with pytest.raises(AssertionError):
        match_table_to_element(sdata=sdata_query_aggregation, element_name="points")
    # table is not annotating "by_circles"
    with pytest.raises(AssertionError, match="No row matches in the table annotates the element"):
        match_table_to_element(sdata=sdata_query_aggregation, element_name="by_circles")
    matched_table = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    arr = np.array(list(reversed(sdata_query_aggregation["values_circles"].index)))
    sdata_query_aggregation["values_circles"].index = arr
    matched_table_reversed = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    assert matched_table.obs.index.tolist() == list(reversed(matched_table_reversed.obs.index.tolist()))

    # TODO: add tests for labels


def test_left_inner_right_exclusive_join(sdata_query_aggregation):
    element_dict, table = join_sdata_spatialelement_table(
        sdata_query_aggregation, "values_polygons", "table", "right_exclusive"
    )
    assert table is None
    assert all(element_dict[key] is None for key in element_dict)

    sdata_query_aggregation["values_polygons"] = sdata_query_aggregation["values_polygons"].drop([10, 11])
    with pytest.raises(AssertionError, match="No table with"):
        join_sdata_spatialelement_table(sdata_query_aggregation, "values_polygons", "not_existing_table", "left")

    # Should we reindex before returning the table?
    element_dict, table = join_sdata_spatialelement_table(sdata_query_aggregation, "values_polygons", "table", "left")
    assert all(element_dict["values_polygons"].index == table.obs["instance_id"].values)

    # Check no matches in table for element not annotated by table
    element_dict, table = join_sdata_spatialelement_table(sdata_query_aggregation, "by_polygons", "table", "left")
    assert table is None
    assert element_dict["by_polygons"] is sdata_query_aggregation["by_polygons"]

    # Check multiple elements, one of which not annotated by table
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_sdata_spatialelement_table(
            sdata_query_aggregation, ["by_polygons", "values_polygons"], "table", "left"
        )
    assert "by_polygons" in element_dict

    # check multiple elements joined to table.
    sdata_query_aggregation["values_circles"] = sdata_query_aggregation["values_circles"].drop([7, 8])
    element_dict, table = join_sdata_spatialelement_table(
        sdata_query_aggregation, ["values_circles", "values_polygons"], "table", "left"
    )
    indices = pd.concat(
        [element_dict["values_circles"].index.to_series(), element_dict["values_polygons"].index.to_series()]
    )
    assert all(table.obs["instance_id"] == indices.values)

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_sdata_spatialelement_table(
            sdata_query_aggregation, ["values_circles", "values_polygons", "by_polygons"], "table", "right_exclusive"
        )
    assert all(element_dict[key] is None for key in element_dict)
    assert all(table.obs.index == ["7", "8", "19", "20"])
    assert all(table.obs["instance_id"].values == [7, 8, 10, 11])
    assert all(table.obs["region"].values == ["values_circles", "values_circles", "values_polygons", "values_polygons"])

    # the triggered warning is: UserWarning: The element `{name}` is not annotated by the table. Skipping
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_sdata_spatialelement_table(
            sdata_query_aggregation, ["values_circles", "values_polygons", "by_polygons"], "table", "inner"
        )
    indices = pd.concat(
        [element_dict["values_circles"].index.to_series(), element_dict["values_polygons"].index.to_series()]
    )
    assert all(table.obs["instance_id"] == indices.values)
    assert element_dict["by_polygons"] is None


def test_join_spatialelement_table_fail(full_sdata):
    with pytest.warns(UserWarning, match="Images:"):
        join_sdata_spatialelement_table(full_sdata, ["image2d", "labels2d"], "table", "left_exclusive")
    with pytest.warns(UserWarning, match="Tables:"):
        join_sdata_spatialelement_table(full_sdata, ["labels2d", "table"], "table", "left_exclusive")
    with pytest.raises(ValueError, match="`not_join` is not a"):
        join_sdata_spatialelement_table(full_sdata, "labels2d", "table", "not_join")


def test_left_exclusive_and_right_join(sdata_query_aggregation):
    # Test case in which all table rows match rows in elements
    element_dict, table = join_sdata_spatialelement_table(
        sdata_query_aggregation, ["values_circles", "values_polygons"], "table", "left_exclusive"
    )
    assert all(element_dict[key] is None for key in element_dict)
    assert table is None

    # Dropped indices correspond to instance ids 7, 8 for 'values_circles' and 10, 11 for 'values_polygons'
    sdata_query_aggregation["table"] = sdata_query_aggregation["table"][
        sdata_query_aggregation["table"].obs.index.drop(["7", "8", "19", "20"])
    ]
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_sdata_spatialelement_table(
            sdata_query_aggregation, ["values_polygons", "by_polygons"], "table", "left_exclusive"
        )
    assert table is None
    assert not set(element_dict["values_polygons"].index).issubset(sdata_query_aggregation["table"].obs["instance_id"])

    # test right join
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_sdata_spatialelement_table(
            sdata_query_aggregation, ["values_circles", "values_polygons", "by_polygons"], "table", "right"
        )
    assert table is sdata_query_aggregation["table"]
    assert not {7, 8}.issubset(element_dict["values_circles"].index)
    assert not {10, 11}.issubset(element_dict["values_polygons"].index)

    element_dict, table = join_sdata_spatialelement_table(
        sdata_query_aggregation, ["values_circles", "values_polygons"], "table", "left_exclusive"
    )
    assert table is None
    assert not np.array_equal(
        sdata_query_aggregation["table"].obs.iloc[7:9]["instance_id"].values,
        element_dict["values_circles"].index.values,
    )
    assert not np.array_equal(
        sdata_query_aggregation["table"].obs.iloc[19:21]["instance_id"].values,
        element_dict["values_polygons"].index.values,
    )


def test_locate_value(sdata_query_aggregation):
    def _check_location(locations: list[_ValueOrigin], origin: str, is_categorical: bool):
        assert len(locations) == 1
        assert locations[0].origin == origin
        assert locations[0].is_categorical == is_categorical

    # var, numerical
    _check_location(
        _locate_value(value_key="numerical_in_var", sdata=sdata_query_aggregation, element_name="values_circles"),
        origin="var",
        is_categorical=False,
    )
    # obs, categorical
    _check_location(
        _locate_value(value_key="categorical_in_obs", sdata=sdata_query_aggregation, element_name="values_circles"),
        origin="obs",
        is_categorical=True,
    )
    # obs, numerical
    _check_location(
        _locate_value(value_key="numerical_in_obs", sdata=sdata_query_aggregation, element_name="values_circles"),
        origin="obs",
        is_categorical=False,
    )
    # gdf, categorical
    # sdata + element_name
    _check_location(
        _locate_value(value_key="categorical_in_gdf", sdata=sdata_query_aggregation, element_name="values_circles"),
        origin="df",
        is_categorical=True,
    )
    # element
    _check_location(
        _locate_value(value_key="categorical_in_gdf", element=sdata_query_aggregation["values_circles"]),
        origin="df",
        is_categorical=True,
    )
    # gdf, numerical
    # sdata + element_name
    _check_location(
        _locate_value(value_key="numerical_in_gdf", sdata=sdata_query_aggregation, element_name="values_circles"),
        origin="df",
        is_categorical=False,
    )
    # element
    _check_location(
        _locate_value(value_key="numerical_in_gdf", element=sdata_query_aggregation["values_circles"]),
        origin="df",
        is_categorical=False,
    )
    # ddf, categorical
    # sdata + element_name
    _check_location(
        _locate_value(value_key="categorical_in_ddf", sdata=sdata_query_aggregation, element_name="points"),
        origin="df",
        is_categorical=True,
    )
    # element
    _check_location(
        _locate_value(value_key="categorical_in_ddf", element=sdata_query_aggregation["points"]),
        origin="df",
        is_categorical=True,
    )
    # ddf, numerical
    # sdata + element_name
    _check_location(
        _locate_value(value_key="numerical_in_ddf", sdata=sdata_query_aggregation, element_name="points"),
        origin="df",
        is_categorical=False,
    )
    # element
    _check_location(
        _locate_value(value_key="numerical_in_ddf", element=sdata_query_aggregation["points"]),
        origin="df",
        is_categorical=False,
    )


def test_get_values_df(sdata_query_aggregation):
    # test with a single value, in the dataframe; using sdata + element_name
    v = get_values(value_key="numerical_in_gdf", sdata=sdata_query_aggregation, element_name="values_circles")
    assert v.shape == (9, 1)

    # test with multiple values, in the dataframe; using element
    sdata_query_aggregation.shapes["values_circles"]["another_numerical_in_gdf"] = v
    v = get_values(
        value_key=["numerical_in_gdf", "another_numerical_in_gdf"], element=sdata_query_aggregation["values_circles"]
    )
    assert v.shape == (9, 2)

    # test with a single value, in the obs
    v = get_values(value_key="numerical_in_obs", sdata=sdata_query_aggregation, element_name="values_circles")
    assert v.shape == (9, 1)

    # test with multiple values, in the obs
    sdata_query_aggregation.table.obs["another_numerical_in_obs"] = v
    v = get_values(
        value_key=["numerical_in_obs", "another_numerical_in_obs"],
        sdata=sdata_query_aggregation,
        element_name="values_circles",
    )
    assert v.shape == (9, 2)

    # test with a single value, in the var
    v = get_values(value_key="numerical_in_var", sdata=sdata_query_aggregation, element_name="values_circles")
    assert v.shape == (9, 1)

    # test with multiple values, in the var
    # prepare the data
    adata = sdata_query_aggregation.table
    X = adata.X
    new_X = np.hstack([X, X[:, 0:1]])
    new_adata = AnnData(
        X=new_X, obs=adata.obs, var=pd.DataFrame(index=["numerical_in_var", "another_numerical_in_var"]), uns=adata.uns
    )
    del sdata_query_aggregation.table
    sdata_query_aggregation.table = new_adata
    # test
    v = get_values(
        value_key=["numerical_in_var", "another_numerical_in_var"],
        sdata=sdata_query_aggregation,
        element_name="values_circles",
    )
    assert v.shape == (9, 2)

    # test exceptions
    # value found in multiple locations
    sdata_query_aggregation.table.obs["another_numerical_in_gdf"] = np.zeros(21)
    with pytest.raises(ValueError):
        get_values(value_key="another_numerical_in_gdf", sdata=sdata_query_aggregation, element_name="values_circles")

    # value not found
    with pytest.raises(ValueError):
        get_values(value_key="not_present", sdata=sdata_query_aggregation, element_name="values_circles")

    # mixing categorical and numerical values
    with pytest.raises(ValueError):
        get_values(
            value_key=["numerical_in_gdf", "categorical_in_gdf"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
        )

    # multiple categorical values
    sdata_query_aggregation.shapes["values_circles"]["another_categorical_in_gdf"] = np.zeros(9)
    with pytest.raises(ValueError):
        get_values(
            value_key=["categorical_in_gdf", "another_categorical_in_gdf"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
        )

    # mixing different origins
    with pytest.raises(ValueError):
        get_values(
            value_key=["numerical_in_gdf", "numerical_in_obs"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
        )


def test_get_values_labels_bug(sdata_blobs):
    # https://github.com/scverse/spatialdata-plot/issues/165
    from spatialdata import get_values

    get_values("channel_0_sum", sdata=sdata_blobs, element_name="blobs_labels")


def test_filter_table_categorical_bug(shapes):
    # one bug that was triggered by: https://github.com/scverse/anndata/issues/1210
    adata = AnnData(obs={"categorical": pd.Categorical(["a", "a", "a", "b", "c"])})
    adata.obs["region"] = "circles"
    adata.obs["cell_id"] = np.arange(len(adata))
    adata = TableModel.parse(adata, region=["circles"], region_key="region", instance_key="cell_id")
    adata_subset = adata[adata.obs["categorical"] == "a"].copy()
    shapes.table = adata_subset
    shapes.filter_by_coordinate_system("global")
