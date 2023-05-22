import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import get_values, locate_value
from spatialdata._core.query.relational_query import _ValueOrigin


def test_locate_value(sdata_query_aggregation):
    def _check_location(locations: list[_ValueOrigin], origin: str, is_categorical: bool):
        assert len(locations) == 1
        assert locations[0].origin == origin
        assert locations[0].is_categorical == is_categorical

    _check_location(
        locate_value(sdata_query_aggregation, "values_circles", "numerical_in_var"), origin="var", is_categorical=False
    )
    _check_location(
        locate_value(sdata_query_aggregation, "values_circles", "categorical_in_obs"), origin="obs", is_categorical=True
    )
    _check_location(
        locate_value(sdata_query_aggregation, "values_circles", "numerical_in_obs"), origin="obs", is_categorical=False
    )
    _check_location(
        locate_value(sdata_query_aggregation, "values_circles", "categorical_in_gdf"), origin="df", is_categorical=True
    )
    _check_location(
        locate_value(sdata_query_aggregation, "values_circles", "numerical_in_gdf"), origin="df", is_categorical=False
    )
    _check_location(
        locate_value(sdata_query_aggregation, "points", "categorical_in_ddf"), origin="df", is_categorical=True
    )
    _check_location(
        locate_value(sdata_query_aggregation, "points", "numerical_in_ddf"), origin="df", is_categorical=False
    )


def test_get_values_df(sdata_query_aggregation):
    # test with a single value, in the dataframe
    v = get_values(sdata_query_aggregation, "values_circles", "numerical_in_gdf")
    assert v.shape == (9, 1)

    # test with multiple values, in the dataframe
    sdata_query_aggregation.shapes["values_circles"]["another_numerical_in_gdf"] = v
    v = get_values(sdata_query_aggregation, "values_circles", ["numerical_in_gdf", "another_numerical_in_gdf"])
    assert v.shape == (9, 2)

    # test with a single value, in the obs
    v = get_values(sdata_query_aggregation, "values_circles", "numerical_in_obs")
    assert v.shape == (21, 1)

    # test with multiple values, in the obs
    sdata_query_aggregation.table.obs["another_numerical_in_obs"] = v
    v = get_values(sdata_query_aggregation, "values_circles", ["numerical_in_obs", "another_numerical_in_obs"])
    assert v.shape == (21, 2)

    # test with a single value, in the var
    v = get_values(sdata_query_aggregation, "values_circles", "numerical_in_var")
    assert v.shape == (21, 1)

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
    v = get_values(sdata_query_aggregation, "values_circles", ["numerical_in_var", "another_numerical_in_var"])
    assert v.shape == (21, 2)

    # test exceptions
    # value found in multiple locations
    sdata_query_aggregation.table.obs["another_numerical_in_gdf"] = np.zeros(21)
    with pytest.raises(ValueError):
        get_values(sdata_query_aggregation, "values_circles", "another_numerical_in_gdf")

    # value not found
    with pytest.raises(ValueError):
        get_values(sdata_query_aggregation, "values_circles", "not_present")

    # mixing categorical and numerical values
    with pytest.raises(ValueError):
        get_values(sdata_query_aggregation, "values_circles", ["numerical_in_gdf", "categorical_in_gdf"])

    # multiple categorical values
    sdata_query_aggregation.shapes["values_circles"]["another_categorical_in_gdf"] = np.zeros(9)
    with pytest.raises(ValueError):
        get_values(sdata_query_aggregation, "values_circles", ["categorical_in_gdf", "another_categorical_in_gdf"])

    # mixing different origins
    with pytest.raises(ValueError):
        get_values(sdata_query_aggregation, "values_circles", ["numerical_in_gdf", "numerical_in_obs"])
