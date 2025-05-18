from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from spatialdata import SpatialData
from spatialdata._utils import sanitize_name, sanitize_table


@pytest.fixture
def invalid_table() -> AnnData:
    """AnnData with invalid obs column names to test basic sanitization."""
    return AnnData(
        obs=pd.DataFrame(
            {
                "@invalid#": [1, 2],
                "valid_name": [3, 4],
                "__private": [5, 6],
            }
        )
    )


@pytest.fixture
def invalid_table_with_index() -> AnnData:
    """AnnData with a name requiring whitespace→underscore and a dataframe index column."""
    return AnnData(
        obs=pd.DataFrame(
            {
                "invalid name": [1, 2],
                "_index": [3, 4],
            }
        )
    )


@pytest.fixture
def sdata_sanitized_tables(invalid_table, invalid_table_with_index) -> SpatialData:
    """SpatialData built from sanitized copies of the invalid tables."""
    table1 = invalid_table.copy()
    table2 = invalid_table_with_index.copy()
    sanitize_table(table1)
    sanitize_table(table2)
    return SpatialData(tables={"table1": table1, "table2": table2})


# -----------------------------------------------------------------------------
# sanitize_name tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("valid_name", "valid_name"),
        ("valid-name", "valid-name"),
        ("valid.name", "valid.name"),
        ("invalid@name", "invalid_name"),
        ("invalid#name", "invalid_name"),
        ("invalid name", "invalid_name"),
        ("", "unnamed"),
        (".", "unnamed"),
        ("..", "unnamed"),
        ("__private", "private"),
    ],
)
def test_sanitize_name_strips_special_chars(raw, expected):
    assert sanitize_name(raw) == expected


@pytest.mark.parametrize(
    "raw,is_df_col,expected",
    [
        ("_index", True, "index"),
        ("_index", False, "index"),
        ("valid@column", True, "valid_column"),
        ("__private", True, "private"),
    ],
)
def test_sanitize_name_dataframe_column(raw, is_df_col, expected):
    assert sanitize_name(raw, is_dataframe_column=is_df_col) == expected


# -----------------------------------------------------------------------------
# sanitize_table basic behaviors
# -----------------------------------------------------------------------------


def test_sanitize_table_basic_columns(invalid_table, invalid_table_with_index):
    ad1 = sanitize_table(invalid_table, inplace=False)
    assert isinstance(ad1, AnnData)
    assert list(ad1.obs.columns) == ["invalid_", "valid_name", "private"]

    ad2 = sanitize_table(invalid_table_with_index, inplace=False)
    assert list(ad2.obs.columns) == ["invalid_name", "index"]

    # original fixture remains unchanged
    assert list(invalid_table.obs.columns) == ["@invalid#", "valid_name", "__private"]


def test_sanitize_table_inplace_copy(invalid_table):
    ad = invalid_table.copy()
    sanitize_table(ad)  # inplace=True is now default
    assert list(ad.obs.columns) == ["invalid_", "valid_name", "private"]


def test_sanitize_table_case_insensitive_collisions():
    obs = pd.DataFrame(
        {
            "Column1": [1, 2],
            "column1": [3, 4],
            "COLUMN1": [5, 6],
        }
    )
    ad = AnnData(obs=obs)
    sanitized = sanitize_table(ad, inplace=False)
    cols = list(sanitized.obs.columns)
    assert sorted(cols) == sorted(["Column1", "column1_1", "COLUMN1_2"])


def test_sanitize_table_whitespace_collision():
    """Ensure 'a b' → 'a_b' doesn't collide silently with existing 'a_b'."""
    obs = pd.DataFrame({"a b": [1], "a_b": [2]})
    ad = AnnData(obs=obs)
    sanitized = sanitize_table(ad, inplace=False)
    cols = list(sanitized.obs.columns)
    assert "a_b" in cols
    assert "a_b_1" in cols


# -----------------------------------------------------------------------------
# sanitize_table attribute‐specific tests
# -----------------------------------------------------------------------------


def test_sanitize_table_obs_and_obs_columns():
    ad = AnnData(obs=pd.DataFrame({"@col": [1, 2]}))
    sanitized = sanitize_table(ad, inplace=False)
    assert list(sanitized.obs.columns) == ["col"]


def test_sanitize_table_obsm_and_obsp():
    ad = AnnData(obs=pd.DataFrame({"@col": [1, 2]}))
    ad.obsm["@col"] = np.array([[1, 2], [3, 4]])
    ad.obsp["bad name"] = np.array([[1, 2], [3, 4]])
    sanitized = sanitize_table(ad, inplace=False)
    assert list(sanitized.obsm.keys()) == ["col"]
    assert list(sanitized.obsp.keys()) == ["bad_name"]


def test_sanitize_table_varm_and_varp():
    ad = AnnData(obs=pd.DataFrame({"x": [1, 2]}), var=pd.DataFrame(index=["v1", "v2"]))
    ad.varm["__priv"] = np.array([[1, 2], [3, 4]])
    ad.varp["_index"] = np.array([[1, 2], [3, 4]])
    sanitized = sanitize_table(ad, inplace=False)
    assert list(sanitized.varm.keys()) == ["priv"]
    assert list(sanitized.varp.keys()) == ["index"]


def test_sanitize_table_uns_and_layers():
    ad = AnnData(obs=pd.DataFrame({"x": [1, 2]}), var=pd.DataFrame(index=["v1", "v2"]))
    ad.uns["bad@key"] = "val"
    ad.layers["bad#layer"] = np.array([[0, 1], [1, 0]])
    sanitized = sanitize_table(ad, inplace=False)
    assert list(sanitized.uns.keys()) == ["bad_key"]
    assert list(sanitized.layers.keys()) == ["bad_layer"]


def test_sanitize_table_empty_returns_empty():
    ad = AnnData()
    sanitized = sanitize_table(ad, inplace=False)
    assert isinstance(sanitized, AnnData)
    assert sanitized.obs.empty
    assert sanitized.var.empty


def test_sanitize_table_preserves_underlying_data():
    ad = AnnData(obs=pd.DataFrame({"@invalid#": [1, 2], "valid": [3, 4]}))
    ad.obsm["@invalid#"] = np.array([[1, 2], [3, 4]])
    ad.uns["invalid@key"] = "value"
    sanitized = sanitize_table(ad, inplace=False)
    assert sanitized.obs["invalid_"].tolist() == [1, 2]
    assert sanitized.obs["valid"].tolist() == [3, 4]
    assert np.array_equal(sanitized.obsm["invalid_"], np.array([[1, 2], [3, 4]]))
    assert sanitized.uns["invalid_key"] == "value"


# -----------------------------------------------------------------------------
# SpatialData integration
# -----------------------------------------------------------------------------


def test_sanitize_table_in_spatialdata_sanitized_fixture(sdata_sanitized_tables):
    t1 = sdata_sanitized_tables.tables["table1"]
    t2 = sdata_sanitized_tables.tables["table2"]
    assert list(t1.obs.columns) == ["invalid_", "valid_name", "private"]
    assert list(t2.obs.columns) == ["invalid_name", "index"]


def test_spatialdata_retains_other_elements(full_sdata, sdata_sanitized_tables):
    # Add another sanitized table into an existing full_sdata
    tbl = AnnData(obs=pd.DataFrame({"@foo#": [1, 2], "bar": [3, 4]}))
    sanitize_table(tbl)
    full_sdata.tables["new_table"] = tbl

    # Verify columns and presence of other SpatialData attributes
    assert list(full_sdata.tables["new_table"].obs.columns) == ["foo_", "bar"]
    assert "image2d" in full_sdata.images
    assert "labels2d" in full_sdata.labels
    assert "points_0" in full_sdata.points
