import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import get_values, match_table_to_element
from spatialdata._core.query.relational_query import _locate_value, _ValueOrigin


def test_match_table_to_element(sdata_query_aggregation):
    # table can't annotate points
    with pytest.raises(AssertionError):
        match_table_to_element(sdata=sdata_query_aggregation, element_name="points")
    # table is not annotating "by_circles"
    with pytest.raises(AssertionError, match="No row matches in the table annotates the element"):
        match_table_to_element(sdata=sdata_query_aggregation, element_name="by_circles")
    matched_table = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    sdata_query_aggregation["values_circles"].index = np.array(
        reversed(sdata_query_aggregation["values_circles"].index)
    )
    matched_table_reversed = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    assert matched_table.obs.index.tolist() == list(reversed(matched_table_reversed.obs.index.tolist()))

    # TODO: add tests for labels


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
