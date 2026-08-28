from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import annsel as an
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from spatialdata import SpatialData, get_values, match_table_to_element
from spatialdata._core.query.relational_query import (
    _locate_value,
    _ValueOrigin,
    filter_by_table_query,
    get_element_annotators,
    join_spatialelement_table,
)
from spatialdata.models.models import TableModel
from spatialdata.testing import assert_anndata_equal, assert_geodataframe_equal


def test_match_table_to_element(sdata_query_aggregation):
    matched_table = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    arr = np.array(list(reversed(sdata_query_aggregation["values_circles"].index)))
    sdata_query_aggregation["values_circles"].index = arr
    matched_table_reversed = match_table_to_element(sdata=sdata_query_aggregation, element_name="values_circles")
    assert matched_table.obs.index.tolist() == list(reversed(matched_table_reversed.obs.index.tolist()))

    # TODO: add tests for labels


def test_join_using_string_instance_id_and_index(sdata_query_aggregation):
    sdata_query_aggregation["table"].obs["instance_id"] = [
        f"string_{i}" for i in sdata_query_aggregation["table"].obs["instance_id"]
    ]
    sdata_query_aggregation["values_circles"].index = pd.Index(
        [f"string_{i}" for i in sdata_query_aggregation["values_circles"].index]
    )
    sdata_query_aggregation["values_polygons"].index = pd.Index(
        [f"string_{i}" for i in sdata_query_aggregation["values_polygons"].index]
    )

    sdata_query_aggregation["values_polygons"] = sdata_query_aggregation["values_polygons"][:5]
    sdata_query_aggregation["values_circles"] = sdata_query_aggregation["values_circles"][:5]

    element_dict, table = join_spatialelement_table(
        sdata=sdata_query_aggregation,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="inner",
    )
    # Note that we started with 21 n_obs.
    assert table.n_obs == 10

    element_dict, table = join_spatialelement_table(
        sdata=sdata_query_aggregation,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="right_exclusive",
    )
    assert table.n_obs == 11

    element_dict, table = join_spatialelement_table(
        sdata=sdata_query_aggregation,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="right",
    )
    assert table.n_obs == 21


# TODO: there is a lot of dublicate code, simplify with a function that tests both the case sdata=None and sdata=sdata
def test_left_inner_right_exclusive_join(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names="values_polygons",
        table_name="table",
        how="right_exclusive",
    )
    assert table is None
    assert all(element_dict[key] is None for key in element_dict)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_polygons"],
        spatial_elements=[sdata["values_polygons"]],
        table=sdata["table"],
        how="right_exclusive",
    )
    assert table is None
    assert all(element_dict[key] is None for key in element_dict)

    sdata["values_polygons"] = sdata["values_polygons"].drop([10, 11])
    with pytest.raises(ValueError, match="No table with"):
        join_spatialelement_table(
            sdata=sdata,
            spatial_element_names="values_polygons",
            table_name="not_existing_table",
            how="left",
        )

    # Should we reindex before returning the table?
    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names="values_polygons",
        table_name="table",
        how="left",
    )
    assert all(element_dict["values_polygons"].index == table.obs["instance_id"].values)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_polygons"],
        spatial_elements=[sdata["values_polygons"]],
        table=sdata["table"],
        how="left",
    )
    assert all(element_dict["values_polygons"].index == table.obs["instance_id"].values)

    # Check no matches in table for element not annotated by table
    element_dict, table = join_spatialelement_table(
        sdata=sdata, spatial_element_names="by_polygons", table_name="table", how="left"
    )
    assert table is None
    assert element_dict["by_polygons"] is sdata["by_polygons"]

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["by_polygons"],
        spatial_elements=[sdata["by_polygons"]],
        table=sdata["table"],
        how="left",
    )
    assert table is None
    assert element_dict["by_polygons"] is sdata["by_polygons"]

    # Check multiple elements, one of which not annotated by table
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["by_polygons", "values_polygons"],
            table_name="table",
            how="left",
        )
    assert "by_polygons" in element_dict

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            spatial_element_names=["by_polygons", "values_polygons"],
            spatial_elements=[sdata["by_polygons"], sdata["values_polygons"]],
            table=sdata["table"],
            how="left",
        )
    assert "by_polygons" in element_dict

    # check multiple elements joined to table.
    sdata["values_circles"] = sdata["values_circles"].drop([7, 8])
    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="left",
    )
    indices = pd.concat(
        [
            element_dict["values_circles"].index.to_series(),
            element_dict["values_polygons"].index.to_series(),
        ]
    )
    assert all(table.obs["instance_id"] == indices.values)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="left",
    )
    indices = pd.concat(
        [
            element_dict["values_circles"].index.to_series(),
            element_dict["values_polygons"].index.to_series(),
        ]
    )
    assert all(table.obs["instance_id"] == indices.values)

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            table_name="table",
            how="right_exclusive",
        )
    assert all(element_dict[key] is None for key in element_dict)
    assert all(table.obs.index == ["7", "8", "19", "20"])
    assert all(table.obs["instance_id"].values == [7, 8, 10, 11])
    assert all(table.obs["region"].values == ["values_circles", "values_circles", "values_polygons", "values_polygons"])

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            spatial_elements=[
                sdata["values_circles"],
                sdata["values_polygons"],
                sdata["by_polygons"],
            ],
            table=sdata["table"],
            how="right_exclusive",
        )
    assert all(element_dict[key] is None for key in element_dict)
    assert all(table.obs.index == ["7", "8", "19", "20"])
    assert all(table.obs["instance_id"].values == [7, 8, 10, 11])
    assert all(table.obs["region"].values == ["values_circles", "values_circles", "values_polygons", "values_polygons"])

    # the triggered warning is: UserWarning: The element `{name}` is not annotated by the table. Skipping
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            table_name="table",
            how="inner",
        )
    indices = pd.concat(
        [
            element_dict["values_circles"].index.to_series(),
            element_dict["values_polygons"].index.to_series(),
        ]
    )
    assert all(table.obs["instance_id"] == indices.values)
    assert element_dict["by_polygons"] is None

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            spatial_elements=[
                sdata["values_circles"],
                sdata["values_polygons"],
                sdata["by_polygons"],
            ],
            table=sdata["table"],
            how="inner",
        )
    indices = pd.concat(
        [
            element_dict["values_circles"].index.to_series(),
            element_dict["values_polygons"].index.to_series(),
        ]
    )
    assert all(table.obs["instance_id"] == indices.values)
    assert element_dict["by_polygons"] is None


def test_join_updates_spatialdata_attrs(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    # table annotates ["values_circles", "values_polygons"]
    original_regions = sdata["table"].uns["spatialdata_attrs"]["region"]
    assert set(original_regions) == {"values_circles", "values_polygons"}

    # left join on a single element: region list must shrink to just that element
    _, table = join_spatialelement_table(
        sdata=sdata, spatial_element_names="values_circles", table_name="table", how="left"
    )
    assert table.uns["spatialdata_attrs"]["region"] == "values_circles"

    # inner join on a single element
    _, table = join_spatialelement_table(
        sdata=sdata, spatial_element_names="values_circles", table_name="table", how="inner"
    )
    assert table.uns["spatialdata_attrs"]["region"] == "values_circles"

    # right_exclusive join: pass a truncated circles element so some table rows have no match.
    # values_circles has 9 instances (0-8); keep only 5 → 4 table rows are exclusive.
    # Use sdata=None mode so we can pass a truncated element under the original region name.
    _, table = join_spatialelement_table(
        spatial_element_names=["values_circles"],
        spatial_elements=[sdata["values_circles"].iloc[:5]],
        table=sdata["table"],
        how="right_exclusive",
    )
    assert table is not None
    assert table.n_obs == 4
    assert table.uns["spatialdata_attrs"]["region"] == "values_circles"

    # original table metadata must be unchanged
    assert set(sdata["table"].uns["spatialdata_attrs"]["region"]) == {"values_circles", "values_polygons"}


def test_join_spatialelement_table_fail(full_sdata):
    with pytest.raises(ValueError, match=" not supported for join operation."):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names=["image2d", "labels2d"],
            table_name="table",
            how="left_exclusive",
        )
    with pytest.raises(ValueError, match=" not supported for join operation."):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names=["labels2d", "table"],
            table_name="table",
            how="left_exclusive",
        )
    with pytest.raises(TypeError, match="`not_join` is not a"):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names="labels2d",
            table_name="table",
            how="not_join",
        )


# TODO: there is a lot of dublicate code, simplify with a function that tests both the case sdata=None and sdata=sdata
def test_left_exclusive_and_right_join(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    # Test case in which all table rows match rows in elements
    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="left_exclusive",
    )
    assert all(element_dict[key] is None for key in element_dict)
    assert table is None

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="left_exclusive",
    )
    assert all(element_dict[key] is None for key in element_dict)
    assert table is None

    # Dropped indices correspond to instance ids 7, 8 for 'values_circles' and 10, 11 for 'values_polygons'
    sdata["table"] = sdata["table"][sdata["table"].obs.index.drop(["7", "8", "19", "20"])]
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["values_polygons", "by_polygons"],
            table_name="table",
            how="left_exclusive",
        )
    assert table is None
    assert not set(element_dict["values_polygons"].index).issubset(sdata["table"].obs["instance_id"])

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            spatial_element_names=["values_polygons", "by_polygons"],
            spatial_elements=[sdata["values_polygons"], sdata["by_polygons"]],
            table=sdata["table"],
            how="left_exclusive",
        )
    assert table is None
    assert not set(element_dict["values_polygons"].index).issubset(sdata["table"].obs["instance_id"])

    # test right join
    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            table_name="table",
            how="right",
        )
    assert table is sdata["table"]
    assert not {7, 8}.issubset(element_dict["values_circles"].index)
    assert not {10, 11}.issubset(element_dict["values_polygons"].index)

    with pytest.warns(UserWarning, match="The element"):
        element_dict, table = join_spatialelement_table(
            spatial_element_names=["values_circles", "values_polygons", "by_polygons"],
            spatial_elements=[
                sdata["values_circles"],
                sdata["values_polygons"],
                sdata["by_polygons"],
            ],
            table=sdata["table"],
            how="right",
        )
    assert table is sdata["table"]
    assert not {7, 8}.issubset(element_dict["values_circles"].index)
    assert not {10, 11}.issubset(element_dict["values_polygons"].index)

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="left_exclusive",
    )
    assert table is None
    assert not np.array_equal(
        sdata["table"].obs.iloc[7:9]["instance_id"].values,
        element_dict["values_circles"].index.values,
    )
    assert not np.array_equal(
        sdata["table"].obs.iloc[19:21]["instance_id"].values,
        element_dict["values_polygons"].index.values,
    )

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="left_exclusive",
    )
    assert table is None
    assert not np.array_equal(
        sdata["table"].obs.iloc[7:9]["instance_id"].values,
        element_dict["values_circles"].index.values,
    )
    assert not np.array_equal(
        sdata["table"].obs.iloc[19:21]["instance_id"].values,
        element_dict["values_polygons"].index.values,
    )


def test_match_rows_inner_join_non_matching_element(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    sdata["values_circles"] = sdata["values_circles"][4:]
    original_index = sdata["values_circles"].index
    reversed_instance_id = [3, 5, 8, 7, 6, 4, 1, 2, 0] + list(reversed(range(12)))
    sdata["table"].obs["instance_id"] = reversed_instance_id

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names="values_circles",
        table_name="table",
        how="inner",
        match_rows="left",
    )
    assert all(table.obs["instance_id"].values == original_index)

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names="values_circles",
        table_name="table",
        how="inner",
        match_rows="right",
    )

    assert all(element_dict["values_circles"].index == [5, 8, 7, 6, 4])


def test_match_rows_inner_join_non_matching_table(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    table = sdata["table"][3:].copy()
    original_instance_id = table.obs["instance_id"].copy()
    reversed_instance_id = [6, 7, 8, 3, 4, 5] + list(reversed(range(12)))
    table.obs["instance_id"] = reversed_instance_id
    sdata["table"] = table

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="inner",
        match_rows="left",
    )

    assert all(table.obs["instance_id"].values == original_instance_id.values)

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="inner",
        match_rows="right",
    )

    indices = element_dict["values_circles"].index.append(element_dict["values_polygons"].index)

    assert all(indices == reversed_instance_id)


# TODO: 'left_exclusive' is currently not working, reported in this issue:
@pytest.mark.parametrize("join_type", ["left", "right", "inner", "right_exclusive"])
def test_inner_join_match_rows_duplicate_obs_indices(sdata_query_aggregation: SpatialData, join_type: str) -> None:
    sdata = sdata_query_aggregation
    sdata["table"].obs.index = ["a"] * sdata["table"].n_obs
    sdata["values_circles"] = sdata_query_aggregation["values_circles"][:4]
    sdata["values_polygons"] = sdata_query_aggregation["values_polygons"][:5]

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how=join_type,
    )

    if join_type in ["left", "inner"]:
        # table check
        assert table.n_obs == 9
        assert np.array_equal(table.obs["instance_id"][:4], sdata["values_circles"].index)
        assert np.array_equal(table.obs["instance_id"][4:], sdata["values_polygons"].index)
        # shapes check
        assert_geodataframe_equal(element_dict["values_circles"], sdata["values_circles"])
        assert_geodataframe_equal(element_dict["values_polygons"], sdata["values_polygons"])
    elif join_type == "right":
        # table check
        assert_anndata_equal(table.obs, sdata["table"].obs)
        # shapes check
        assert_geodataframe_equal(element_dict["values_circles"], sdata["values_circles"])
        assert_geodataframe_equal(element_dict["values_polygons"], sdata["values_polygons"])
    elif join_type == "left_exclusive":
        # TODO: currently not working, reported in this issue
        pass
    else:
        assert join_type == "right_exclusive"
        # table check
        assert table.n_obs == sdata["table"].n_obs - len(sdata["values_circles"]) - len(sdata["values_polygons"])
        # shapes check
        assert element_dict["values_circles"] is None
        assert element_dict["values_polygons"] is None


# TODO: there is a lot of dublicate code, simplify with a function that tests both the case sdata=None and sdata=sdata
def test_match_rows_join(sdata_query_aggregation):
    sdata = sdata_query_aggregation
    reversed_instance_id = [3, 4, 5, 6, 7, 8, 1, 2, 0] + list(reversed(range(12)))
    original_instance_id = sdata["table"].obs["instance_id"]
    sdata["table"].obs["instance_id"] = reversed_instance_id

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="left",
        match_rows="left",
    )
    assert all(table.obs["instance_id"].values == original_instance_id.values)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="left",
        match_rows="left",
    )
    assert all(table.obs["instance_id"].values == original_instance_id.values)

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="right",
        match_rows="right",
    )
    indices = [
        *element_dict["values_circles"].index,
        *element_dict[("values_polygons")].index,
    ]
    assert all(indices == table.obs["instance_id"])

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="right",
        match_rows="right",
    )
    assert all(indices == table.obs["instance_id"])

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="inner",
        match_rows="left",
    )
    assert all(table.obs["instance_id"].values == original_instance_id.values)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="inner",
        match_rows="left",
    )
    assert all(table.obs["instance_id"].values == original_instance_id.values)

    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="inner",
        match_rows="right",
    )
    indices = [
        *element_dict["values_circles"].index,
        *element_dict[("values_polygons")].index,
    ]
    assert all(indices == table.obs["instance_id"])

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="inner",
        match_rows="right",
    )
    assert all(indices == table.obs["instance_id"])

    # check whether table ordering is preserved if not matching
    element_dict, table = join_spatialelement_table(
        sdata=sdata,
        spatial_element_names=["values_circles", "values_polygons"],
        table_name="table",
        how="left",
    )
    assert all(table.obs["instance_id"] == reversed_instance_id)

    element_dict, table = join_spatialelement_table(
        spatial_element_names=["values_circles", "values_polygons"],
        spatial_elements=[sdata["values_circles"], sdata["values_polygons"]],
        table=sdata["table"],
        how="left",
    )
    assert all(table.obs["instance_id"] == reversed_instance_id)


def test_locate_value(sdata_query_aggregation):
    def _check_location(locations: list[_ValueOrigin], origin: str, is_categorical: bool):
        assert len(locations) == 1
        assert locations[0].origin == origin
        assert locations[0].is_categorical == is_categorical

    # var, numerical
    _check_location(
        _locate_value(
            value_key="numerical_in_var",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        ),
        origin="var",
        is_categorical=False,
    )
    # obs, categorical
    _check_location(
        _locate_value(
            value_key="categorical_in_obs",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        ),
        origin="obs",
        is_categorical=True,
    )
    # obs, numerical
    _check_location(
        _locate_value(
            value_key="numerical_in_obs",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        ),
        origin="obs",
        is_categorical=False,
    )
    # gdf, categorical
    # sdata + element_name
    _check_location(
        _locate_value(
            value_key="categorical_in_gdf",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        ),
        origin="df",
        is_categorical=True,
    )
    # element
    _check_location(
        _locate_value(
            value_key="categorical_in_gdf",
            element=sdata_query_aggregation["values_circles"],
            table_name="table",
        ),
        origin="df",
        is_categorical=True,
    )
    # gdf, numerical
    # sdata + element_name
    _check_location(
        _locate_value(
            value_key="numerical_in_gdf",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        ),
        origin="df",
        is_categorical=False,
    )
    # element
    _check_location(
        _locate_value(
            value_key="numerical_in_gdf",
            element=sdata_query_aggregation["values_circles"],
            table_name="table",
        ),
        origin="df",
        is_categorical=False,
    )
    # ddf, categorical
    # sdata + element_name
    _check_location(
        _locate_value(
            value_key="categorical_in_ddf",
            sdata=sdata_query_aggregation,
            element_name="points",
        ),
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
        _locate_value(
            value_key="numerical_in_ddf",
            sdata=sdata_query_aggregation,
            element_name="points",
        ),
        origin="df",
        is_categorical=False,
    )
    # element
    _check_location(
        _locate_value(value_key="numerical_in_ddf", element=sdata_query_aggregation["points"]),
        origin="df",
        is_categorical=False,
    )


def test_get_values_df_shapes(sdata_query_aggregation):
    # test with a single value, in the dataframe; using sdata + element_name
    v = get_values(
        value_key="numerical_in_gdf",
        sdata=sdata_query_aggregation,
        element_name="values_circles",
        table_name="table",
    )
    assert v.shape == (9, 1)

    # test with multiple values, in the dataframe; using element
    sdata_query_aggregation.shapes["values_circles"]["another_numerical_in_gdf"] = v
    v = get_values(
        value_key=["numerical_in_gdf", "another_numerical_in_gdf"],
        element=sdata_query_aggregation["values_circles"],
    )
    assert v.shape == (9, 2)

    # test with a single value, in the obs
    v = get_values(
        value_key="numerical_in_obs",
        sdata=sdata_query_aggregation,
        element_name="values_circles",
        table_name="table",
    )
    assert v.shape == (9, 1)

    # test with multiple values, in the obs
    sdata_query_aggregation["table"].obs["another_numerical_in_obs"] = v
    v = get_values(
        value_key=["numerical_in_obs", "another_numerical_in_obs"],
        sdata=sdata_query_aggregation,
        element_name="values_circles",
        table_name="table",
    )
    assert v.shape == (9, 2)

    # test with a single value, in the var
    v = get_values(
        value_key="numerical_in_var",
        sdata=sdata_query_aggregation,
        element_name="values_circles",
        table_name="table",
    )
    assert v.shape == (9, 1)

    # test with multiple values, in the var
    # prepare the data
    adata = sdata_query_aggregation["table"]
    X = adata.X
    new_X = np.hstack([X, X[:, 0:1]])
    new_adata = AnnData(
        X=new_X,
        obs=adata.obs,
        var=pd.DataFrame(index=["numerical_in_var", "another_numerical_in_var"]),
        uns=adata.uns,
    )
    sdata_query_aggregation["table"] = new_adata
    # test
    v = get_values(
        value_key=["numerical_in_var", "another_numerical_in_var"],
        sdata=sdata_query_aggregation,
        element_name="values_circles",
        table_name="table",
    )
    assert v.shape == (9, 2)

    # test exceptions
    # value found in multiple locations
    sdata_query_aggregation["table"].obs["another_numerical_in_gdf"] = np.zeros(21)
    with pytest.raises(ValueError):
        get_values(
            value_key="another_numerical_in_gdf",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        )

    # value not found
    with pytest.raises(ValueError):
        get_values(
            value_key="not_present",
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        )

    # mixing categorical and numerical values
    with pytest.raises(ValueError):
        get_values(
            value_key=["numerical_in_gdf", "categorical_in_gdf"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        )

    # multiple categorical values
    sdata_query_aggregation.shapes["values_circles"]["another_categorical_in_gdf"] = np.zeros(9)
    with pytest.raises(ValueError):
        get_values(
            value_key=["categorical_in_gdf", "another_categorical_in_gdf"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        )

    # mixing different origins
    with pytest.raises(ValueError):
        get_values(
            value_key=["numerical_in_gdf", "numerical_in_obs"],
            sdata=sdata_query_aggregation,
            element_name="values_circles",
            table_name="table",
        )


def test_get_values_df_points(points):
    # testing get_values() for points, we keep the test more minimalistic than the one for shapes
    p = points["points_0"]
    p = p.drop("instance_id", axis=1)
    p.index.compute()
    n = len(p)
    obs = pd.DataFrame(index=p.index.astype(str), data={"region": ["points_0"] * n, "instance_id": range(n)})
    obs["region"] = obs["region"].astype("category")
    table = TableModel.parse(
        AnnData(shape=(n, 0), obs=obs),
        region="points_0",
        region_key="region",
        instance_key="instance_id",
    )
    points["points_0"] = p
    points["table"] = table

    assert get_values(value_key="region", element_name="points_0", sdata=points, table_name="table").shape == (300, 1)
    get_values(
        value_key="instance_id",
        element_name="points_0",
        sdata=points,
        table_name="table",
    )
    get_values(value_key=["x", "y"], element_name="points_0", sdata=points, table_name="table")
    get_values(value_key="genes", element_name="points_0", sdata=points, table_name="table")

    pass


def test_get_values_obsm(adata_labels: AnnData):
    get_values(value_key="tensor", element=adata_labels)

    get_values(value_key=["tensor", "tensor_copy"], element=adata_labels)

    values = get_values(value_key="tensor", element=adata_labels, return_obsm_as_is=True)
    assert isinstance(values, np.ndarray)


def test_get_values_table(sdata_blobs):
    df = get_values(value_key="channel_0_sum", element=sdata_blobs["table"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 26


def test_get_values_table_different_layer(sdata_blobs):
    sdata_blobs["table"].layers["layer"] = np.log1p(sdata_blobs["table"].X)
    df = get_values(value_key="channel_0_sum", element=sdata_blobs["table"])
    df_layer = get_values(value_key="channel_0_sum", element=sdata_blobs["table"], table_layer="layer")
    assert np.allclose(np.log1p(df), df_layer)


def test_get_values_table_element_name(sdata_blobs):
    sdata_blobs["table"].obs["region"] = sdata_blobs["table"].obs["region"].cat.add_categories("another_region")
    sdata_blobs["table"].obs.loc["1", "region"] = "another_region"
    sdata_blobs["table"].uns["spatialdata_attrs"]["region"] = [
        "blobs_labels",
        "another_region",
    ]
    sdata_blobs["another_region"] = sdata_blobs["blobs_labels"]
    df = get_values(
        value_key="channel_0_sum",
        element=sdata_blobs["table"],
        element_name="blobs_labels",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25


def test_get_values_labels_bug(sdata_blobs):
    # https://github.com/scverse/spatialdata-plot/issues/165
    get_values(
        "channel_0_sum",
        sdata=sdata_blobs,
        element_name="blobs_labels",
        table_name="table",
    )


def test_filter_table_categorical_bug(shapes):
    # one bug that was triggered by: https://github.com/scverse/anndata/issues/1210
    adata = AnnData(
        obs=pd.DataFrame({"categorical": pd.Categorical(["a", "a", "a", "b", "c"])}, index=list(map(str, range(5))))
    )
    adata.obs["region"] = pd.Categorical(["circles"] * adata.n_obs)
    adata.obs["cell_id"] = np.arange(len(adata))
    adata = TableModel.parse(adata, region=["circles"], region_key="region", instance_key="cell_id")
    adata_subset = adata[adata.obs["categorical"] == "a"].copy()
    shapes["table"] = adata_subset
    shapes.filter_by_coordinate_system("global")


@dataclass
class _JoinOutcome:
    """Expected outcome of a `join_spatialelement_table()` call, for a given `how`/`match_rows` pair."""

    # expected `joined_table.obs["label"]`, in order; `None` when no table is expected to be returned
    table_order: list[str] | None = None
    # whether a "Matching rows '<...>' is not supported for '<...>' join." UserWarning is expected to be emitted
    warns: bool = False
    # expected values of `element_dict[name].index` for element name in {"a", "b"}; `None` for a element name whose
    # returned join result is expected to be `None` (e.g. fully excluded, or not returned by this join type)
    element_index: dict[str, list[int] | None] = field(default_factory=dict)


def _make_interleaved_regions_sdata() -> tuple[SpatialData, dict[str, dict[str, _JoinOutcome]]]:
    from geopandas import GeoDataFrame
    from shapely.geometry import Point

    from spatialdata.models import ShapesModel

    def circles(indices):
        # `indices` gives both the number of circles and the (non-default) row order of the element.
        gdf = GeoDataFrame(
            {"geometry": [Point(i, i) for i in range(len(indices))], "radius": [1.0] * len(indices)},
            index=pd.Index(indices),
        )
        return ShapesModel.parse(gdf)

    # assumptions/comments:
    # - no duplicate values in the index of each spatial element (duplicate values are tested elsewhere)
    # - no duplicate values for the instance_key column of the table (duplicate values are tested elsewhere)
    #
    # edge cases being tested:
    # - instance_id values are non-monotonic
    # - the index in each spatial element is non-monotonic
    # - we also set the index of the table obs to random values; these should be ignored (in the code we call .index on
    #   a region_key column, but the index is freshly reset by a nearby call of .reset_index() inside the join
    #   machinery)
    # - "b3" and "c7" are unmatched table rows: "b3" refers to a missing instance in a
    #   spatial element that exists, while "c7" refers to a region with no spatial element
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["b", "b", "a", "b", "a", "a", "b", "c"]),
            "instance_id": [2, 1, 2, 3, 1, 0, 0, 7],
            "label": ["b2", "b1", "a2", "b3", "a1", "a0", "b0", "c7"],
        },
        index=np.random.default_rng(0).integers(0, 3, size=8).astype(str),
    )
    # "a" additionally has unmatched instance ids 5, 4, and "b" has 4, 6.
    # These test that unmatched element rows are preserved in element order by
    # "left" and "left_exclusive" joins.
    shapes = {"a": circles([2, 1, 0, 5, 4]), "b": circles([1, 2, 0, 4, 6])}
    # to make understanding easier, you may want to refer to the figure on joins from the docs:
    # https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html
    expected = {
        "left": {
            "no": _JoinOutcome(
                table_order=["b2", "b1", "a2", "a1", "a0", "b0"],
                element_index={"a": [2, 1, 0, 5, 4], "b": [1, 2, 0, 4, 6]},
            ),
            "left": _JoinOutcome(
                table_order=["a2", "a1", "a0", "b1", "b2", "b0"],
                element_index={"a": [2, 1, 0, 5, 4], "b": [1, 2, 0, 4, 6]},
            ),
            "right": _JoinOutcome(
                table_order=["b2", "b1", "a2", "a1", "a0", "b0"],
                warns=True,
                element_index={"a": [2, 1, 0, 5, 4], "b": [1, 2, 0, 4, 6]},
            ),
        },
        "left_exclusive": {
            # by design, "left_exclusive" never returns a table (only filtered elements), regardless of
            # match_rows or whether anything was actually excluded.
            "no": _JoinOutcome(table_order=None, element_index={"a": [5, 4], "b": [4, 6]}),
            "left": _JoinOutcome(table_order=None, element_index={"a": [5, 4], "b": [4, 6]}),
            "right": _JoinOutcome(table_order=None, warns=True, element_index={"a": [5, 4], "b": [4, 6]}),
        },
        "inner": {
            "no": _JoinOutcome(
                table_order=["b2", "b1", "a2", "a1", "a0", "b0"],
                element_index={"a": [2, 1, 0], "b": [1, 2, 0]},
            ),
            "left": _JoinOutcome(
                table_order=["a2", "a1", "a0", "b1", "b2", "b0"], element_index={"a": [2, 1, 0], "b": [1, 2, 0]}
            ),
            "right": _JoinOutcome(
                table_order=["b2", "b1", "a2", "a1", "a0", "b0"], element_index={"a": [2, 1, 0], "b": [2, 1, 0]}
            ),
        },
        "right": {
            "no": _JoinOutcome(
                table_order=["b2", "b1", "a2", "b3", "a1", "a0", "b0", "c7"],
                element_index={"a": [2, 1, 0], "b": [1, 2, 0]},
            ),
            "left": _JoinOutcome(
                table_order=["b2", "b1", "a2", "b3", "a1", "a0", "b0", "c7"],
                warns=True,
                element_index={"a": [2, 1, 0], "b": [1, 2, 0]},
            ),
            "right": _JoinOutcome(
                table_order=["b2", "b1", "a2", "b3", "a1", "a0", "b0", "c7"],
                element_index={"a": [2, 1, 0], "b": [2, 1, 0]},
            ),
        },
        "right_exclusive": {
            "no": _JoinOutcome(table_order=["b3", "c7"], element_index={"a": None, "b": None}),
            "left": _JoinOutcome(table_order=["b3", "c7"], warns=True, element_index={"a": None, "b": None}),
            "right": _JoinOutcome(table_order=["b3", "c7"], element_index={"a": None, "b": None}),
        },
    }

    table = TableModel.parse(
        AnnData(X=np.zeros((len(obs), 1)), obs=obs),
        region=["a", "b", "c"],
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData(shapes=shapes, tables={"table": table})
    return sdata, expected


@pytest.mark.parametrize("match_rows", ["no", "left", "right"])
@pytest.mark.parametrize(
    "how",
    [
        "left",
        "left_exclusive",
        "inner",
        "right",
        pytest.param(
            "right_exclusive",
            marks=pytest.mark.xfail(
                reason="known bug (see https://github.com/scverse/spatialdata/issues/1162): 'right_exclusive' join "
                "drops unmatched table rows belonging to a region with no queried spatial element (e.g. 'c7')",
                strict=True,
            ),
        ),
    ],
)
def test_join_preserves_row_order_multiple_interleaved_regions(how, match_rows):
    # generalization to all the join types of the bug reported in https://github.com/scverse/spatialdata/issues/1162
    # covering all `how` values of `join_spatialelement_table`, crossed with all values of `match_rows`, and checking
    # whether the row orders of the returned spatial elements and table are correct and if the "match_rows not
    # supported" UserWarning is (or isn't) actually raised (see `_make_interleaved_regions_sdata` and `_JoinOutcome`).
    sdata, expected_by_how_and_match_rows = _make_interleaved_regions_sdata()
    outcome = expected_by_how_and_match_rows[how][match_rows]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        element_dict, joined_table = join_spatialelement_table(
            sdata=sdata,
            spatial_element_names=["a", "b"],
            table_name="table",
            how=how,
            match_rows=match_rows,
        )

    # other UserWarnings can also fire here (e.g. anndata's "Observation names are not unique", triggered by
    # the fixture's duplicated obs_names), so only look for the one this test is actually about. The message
    # looks like "Matching rows 'right' is not supported for 'left_exclusive' join; it will be treated as 'no'.",
    # with the two quoted values varying by `match_rows` / `how`.
    unsupported_match_rows_re = re.compile(
        r"Matching rows '[^']+' is not supported for '[^']+' join; it will be treated as 'no'\."
    )
    unsupported_match_rows_warnings = [
        w for w in record if issubclass(w.category, UserWarning) and unsupported_match_rows_re.search(str(w.message))
    ]
    assert bool(unsupported_match_rows_warnings) == outcome.warns

    if outcome.table_order is None:
        assert joined_table is None
    else:
        assert joined_table is not None
        assert list(joined_table.obs["label"]) == outcome.table_order

    for name, expected_index in outcome.element_index.items():
        actual_element = element_dict[name]
        if expected_index is None:
            assert actual_element is None
        else:
            assert actual_element is not None
            assert list(actual_element.index) == expected_index


def test_join_scoping_for_a_region_not_among_the_queried_elements():
    """
    A table can declare a region for which, at join time, no element is passed in `spatial_element_names`. This
    can happen for two different reasons:

      - the element genuinely exists elsewhere in the `SpatialData` object, it's just not part of this
        particular query (e.g. only some of the annotated elements are of interest right now);
      - the element doesn't exist anywhere in the `SpatialData` object at all (e.g. after
        `sdata.subset(..., filter_tables=False)`, which keeps the table whole while dropping elements; this
        also triggers `SpatialData`'s own "table is annotating '{name}', which is not present" warning).

    This test documents that `join_spatialelement_table` currently can't tell these two cases apart -- and,
    across the five join types, treats "not among the queried elements" inconsistently:

      - "left", "left_exclusive", "inner" and "right_exclusive" all scope themselves strictly to the elements
        actually passed in `spatial_element_names`: rows for any other region never appear in their output,
        matched or not.
      - "right" instead always returns the table completely unfiltered (real SQL right-join semantics: every
        right-hand row is kept, whether or not the left side covers its key at all), so rows for a region
        outside the query leak into the result regardless.

    This means "right_exclusive" is *not* simply "right minus inner" here, unlike what its name may suggest.
    Which of these two behaviors is actually intended is an open design question (see
    https://github.com/scverse/spatialdata/issues/1162); until it's resolved, this test only pins down the
    current, observed behavior so a future change is a deliberate, visible diff here.
    """
    from geopandas import GeoDataFrame
    from shapely.geometry import Point

    from spatialdata.models import ShapesModel

    def circle() -> GeoDataFrame:
        return ShapesModel.parse(GeoDataFrame({"geometry": [Point(0, 0)], "radius": [1.0]}, index=pd.Index([0])))

    obs = pd.DataFrame(
        {"region": pd.Categorical(["a", "b", "c"]), "instance_id": [0, 0, 0], "label": ["a0", "b0", "c0"]},
        index=["0", "1", "2"],
    )
    table = TableModel.parse(
        AnnData(X=np.zeros((3, 1)), obs=obs), region=["a", "b", "c"], region_key="region", instance_key="instance_id"
    )
    sdata_c_exists_unqueried = SpatialData(shapes={"a": circle(), "b": circle(), "c": circle()}, tables={"table": table})
    with pytest.warns(UserWarning, match="is annotating 'c'"):
        sdata_c_missing_entirely = sdata_c_exists_unqueried.subset(["a", "b"], filter_tables=False)

    expected_labels_by_how = {
        "left": ["a0", "b0"],
        "left_exclusive": None,
        "inner": ["a0", "b0"],
        "right": ["a0", "b0", "c0"],
        "right_exclusive": None,
    }
    for sdata in (sdata_c_exists_unqueried, sdata_c_missing_entirely):
        for how, expected_labels in expected_labels_by_how.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                _, joined_table = join_spatialelement_table(
                    sdata=sdata, spatial_element_names=["a", "b"], table_name="table", how=how
                )
            actual_labels = None if joined_table is None else joined_table.obs["label"].tolist()
            assert actual_labels == expected_labels, f"how={how!r}"


def test_filter_table_non_annotating(full_sdata):
    obs = pd.DataFrame({"test": ["a", "b", "c"]}, index=list(map(str, range(3))))
    adata = AnnData(obs=obs)
    table = TableModel.parse(adata)
    full_sdata["table"] = table
    full_sdata.filter_by_coordinate_system("global")


def test_labels_table_joins(full_sdata):
    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="labels2d",
        table_name="table",
        how="left",
    )

    assert all(table.obs["instance_id"] == range(1, 100))

    full_sdata["table"].obs["instance_id"] = list(reversed(range(100)))

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="labels2d",
        table_name="table",
        how="left",
        match_rows="left",
    )
    assert all(table.obs["instance_id"] == range(1, 100))

    with pytest.warns(UserWarning, match="Element type"):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names="labels2d",
            table_name="table",
            how="left_exclusive",
        )

    with pytest.warns(UserWarning, match="Element type"):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names="labels2d",
            table_name="table",
            how="inner",
        )

    with pytest.warns(UserWarning, match="Element type"):
        join_spatialelement_table(
            sdata=full_sdata,
            spatial_element_names="labels2d",
            table_name="table",
            how="right",
        )

    # all labels are present in table so should return None
    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="labels2d",
        table_name="table",
        how="right_exclusive",
    )
    assert element_dict["labels2d"] is None
    assert len(table) == 1
    assert all(table.obs["instance_id"] == 0)  # the background value, which is filtered out effectively


def test_points_table_joins(full_sdata):
    full_sdata["table"].uns["spatialdata_attrs"]["region"] = "points_0"
    full_sdata["table"].obs["region"] = ["points_0"] * 100
    full_sdata["table"].obs["region"] = full_sdata["table"].obs["region"].astype("category")

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="left",
    )

    # points should have the same number of rows as before and table as well
    assert len(element_dict["points_0"]) == 300
    assert all(table.obs["instance_id"] == range(100))

    full_sdata["table"].obs["instance_id"] = list(reversed(range(100)))

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="left",
        match_rows="left",
    )
    assert len(element_dict["points_0"]) == 300
    assert all(table.obs["instance_id"] == range(100))

    # We have 100 table instances so resulting length of points should be 200 as we started with 300
    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="left_exclusive",
    )
    assert len(element_dict["points_0"]) == 200
    assert table is None

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="inner",
    )

    assert len(element_dict["points_0"]) == 100
    assert all(table.obs["instance_id"] == list(reversed(range(100))))

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="right",
    )
    assert len(element_dict["points_0"]) == 100
    assert all(table.obs["instance_id"] == list(reversed(range(100))))

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="right",
        match_rows="right",
    )
    assert all(element_dict["points_0"].index.values.compute() == list(reversed(range(100))))
    assert all(table.obs["instance_id"] == list(reversed(range(100))))

    element_dict, table = join_spatialelement_table(
        sdata=full_sdata,
        spatial_element_names="points_0",
        table_name="table",
        how="right_exclusive",
    )
    assert element_dict["points_0"] is None
    assert table is None


def test_get_element_annotators(full_sdata):
    names = get_element_annotators(full_sdata, "points_0")
    assert len(names) == 0

    names = get_element_annotators(full_sdata, "labels2d")
    assert names == {"table"}

    another_table = full_sdata.tables["table"].copy()
    full_sdata.tables["another_table"] = another_table
    names = get_element_annotators(full_sdata, "labels2d")
    assert names == {"another_table", "table"}


def test_filter_by_table_query(complex_sdata):
    """Test basic filtering functionality of filter_by_table_query."""
    sdata = complex_sdata

    # Test 1: Basic filtering on categorical obs column
    result = filter_by_table_query(sdata=sdata, table_name="labels_table", obs_expr=an.col("cell_type") == "T cell")

    # Check that the table was filtered properly
    assert all(result["labels_table"].obs["cell_type"] == "T cell")
    # Check that result has fewer rows than original
    assert result["labels_table"].n_obs < sdata["labels_table"].n_obs
    # Check that labels2d element is still present
    assert "labels2d" in result.labels

    # Test 2: Filtering on numerical obs column
    result = filter_by_table_query(sdata=sdata, table_name="labels_table", obs_expr=an.col("size") > 50)

    # Check that the table was filtered properly
    assert all(result["labels_table"].obs["size"] > 50)
    # Check that labels2d element is still present
    assert "labels2d" in result.labels

    # Test 3: Filtering with var expressions
    result = filter_by_table_query(
        sdata=sdata, table_name="shapes_table", var_expr=an.col("feature_type") == "feature_type1"
    )

    # Check that the filtered var dataframe only has 'spatial' feature_type
    assert all(result["shapes_table"].var["feature_type"] == "feature_type1")
    # Check that the filtered var dataframe has fewer rows than the original
    assert result["shapes_table"].n_vars < sdata["shapes_table"].n_vars

    # Test 4: Multiple filtering conditions (obs and var)
    result = filter_by_table_query(
        sdata=sdata, table_name="shapes_table", obs_expr=an.col("category") == "A", var_expr=an.col("score") > 2
    )

    # Check that both filters were applied
    assert all(result["shapes_table"].obs["category"] == "A")
    assert all(result["shapes_table"].var["score"] > 2)

    # Test 5: Using X expressions
    result = filter_by_table_query(sdata=sdata, table_name="labels_table", x_expr=an.col("feature_1") > 0.5)

    # Check that the filter was applied to X
    assert np.all(result["labels_table"][:, "feature_1"].X > 0.5)

    # Test 6: Using different join types
    # Test with inner join
    result = filter_by_table_query(
        sdata=sdata, table_name="shapes_table", obs_expr=an.col("category") == "A", how="inner"
    )

    # The elements should be filtered to only include instances in the table
    assert "poly" in result.shapes
    assert "circles" in result.shapes
    assert "labels2d" not in result.labels
    assert len(result["poly"]) == 1
    assert len(result["circles"]) == 2

    # Test with left join
    result = filter_by_table_query(
        sdata=sdata, table_name="shapes_table", obs_expr=an.col("category") == "A", how="left"
    )

    # Elements should be preserved but table should be filtered
    assert "poly" in result.shapes
    assert "circles" in result.shapes
    assert "labels2d" not in result.labels
    assert len(result["poly"]) == 5
    assert len(result["circles"]) == 5
    assert all(result["shapes_table"].obs["category"] == "A")

    # Test 7: Filtering with specific element_names
    result = filter_by_table_query(
        sdata=sdata,
        table_name="shapes_table",
        element_names=["poly"],  # Only include poly, not circles
        obs_expr=an.col("category") == "A",
    )

    # Only specified elements should be in the result
    assert "poly" in result.shapes
    assert "circles" not in result.shapes

    # Test 8: Testing orphan table handling
    # First test with include_orphan_tables=False (default)
    result = filter_by_table_query(
        sdata=sdata,
        table_name="shapes_table",
        obs_expr=an.col("category") == "A",
        filter_tables=True,
    )

    # Orphan table should not be in the result
    assert "orphan_table" not in result.tables


def test_filter_by_table_query_with_layers(complex_sdata):
    """Test filtering using different layers."""
    sdata = complex_sdata

    # Test filtering using a specific layer
    result = filter_by_table_query(
        sdata=sdata,
        table_name="labels_table",
        x_expr=an.col("feature_1") > 1.0,
        layer="scaled",  # The 'scaled' layer has values 2x the original X
    )

    # Values in the scaled layer's feature_1 column should be > 1.0
    assert np.all(result["labels_table"][:, "feature_1"].layers["scaled"] > 1.0)


def test_filter_by_table_query_edge_cases(complex_sdata):
    """Test edge cases for filter_by_table_query."""
    sdata = complex_sdata

    # Test 1: Filter by obs_names
    result = filter_by_table_query(
        sdata=sdata,
        table_name="shapes_table",
        obs_names_expr=an.obs_names.str.starts_with("0"),  # Only rows with index starting with '0'
    )

    # Check that filtered table only has obs names starting with '0'
    assert all(str(idx).startswith("0") for idx in result["shapes_table"].obs_names)

    # Test 2: Invalid table name raises KeyError
    with pytest.raises(KeyError, match="nonexistent_table"):
        filter_by_table_query(sdata=sdata, table_name="nonexistent_table", obs_expr=an.col("category") == "A")

    # Test 3: Invalid column name in expression
    with pytest.raises(KeyError):  # The exact exception type may vary
        filter_by_table_query(sdata=sdata, table_name="shapes_table", obs_expr=an.col("nonexistent_column") == "A")

    # Test 4: Using layer that doesn't exist
    with pytest.raises(KeyError):
        filter_by_table_query(
            sdata=sdata, table_name="labels_table", x_expr=an.col("feature_1") > 0.5, layer="nonexistent_layer"
        )

    # Test 5: Filter by var_names
    result = filter_by_table_query(
        sdata=sdata,
        table_name="labels_table",
        var_names_expr=an.var_names.str.contains("feature_[0-4]"),  # Only features 0-4
    )

    # Check that filtered table only has var names matching the pattern
    for idx in result["labels_table"].var_names:
        var_name = str(idx)
        assert var_name.startswith("feature_") and int(var_name.split("_")[1]) < 5

    # Test 6: Invalid element_names (element doesn't exist)
    with pytest.raises(KeyError, match="shapes_table"):
        filter_by_table_query(
            sdata=sdata,
            table_name="shapes_table",
            element_names=["nonexistent_element"],
            obs_expr=an.col("category") == "A",
        )

    # Test 7: Invalid join type raises ValueError
    with pytest.raises(TypeError, match="not a valid type of join."):
        filter_by_table_query(
            sdata=sdata,
            table_name="shapes_table",
            how="invalid_join_type",  # Invalid join type
            obs_expr=an.col("category") == "A",
        )


def test_filter_by_table_query_complex_combination(complex_sdata):
    """Test complex combinations of filters."""
    sdata = complex_sdata

    # Combine multiple filtering criteria
    result = sdata.filter_by_table_query(
        table_name="shapes_table",
        obs_expr=(an.col("category") == "A", an.col("value") > -1),
        var_expr=an.col("feature_type") == "feature_type1",
        how="inner",
    )

    # Validate the combined filtering results
    assert all(result["shapes_table"].obs["category"] == "A")
    assert all(result["shapes_table"].obs["value"] > -1)
    assert all(result["shapes_table"].var["feature_type"] == "feature_type1")

    # Both elements should be present but filtered
    assert "circles" in result.shapes
    assert "poly" in result.shapes

    # The filtered shapes should only contain the instances from the filtered table
    table_instance_ids = set(
        zip(result["shapes_table"].obs["region"], result["shapes_table"].obs["instance_id"], strict=True)
    )
    # if "circles" in result.shapes:
    for idx in result["circles"].index:
        assert ("circles", idx) in table_instance_ids
    for idx in result["poly"].index:
        assert ("poly", idx) in table_instance_ids
