import dask.dataframe as dd
import pandas as pd
import pytest

from spatialdata.models._accessor import AttrsAccessor

# ============================================================================
# General tests
# ============================================================================


def test_dataframe_attrs_is_accessor():
    """Test that DataFrame.attrs is an AttrsAccessor, not a dict."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]}), npartitions=2)
    assert isinstance(df.attrs, AttrsAccessor)


def test_series_attrs_is_accessor():
    """Test that Series.attrs is an AttrsAccessor, not a dict."""
    s = dd.from_pandas(pd.Series([1, 2, 3], name="test"), npartitions=2)
    assert isinstance(s.attrs, AttrsAccessor)


def test_attrs_setitem_getitem():
    """Test setting and getting attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]}), npartitions=2)
    df.attrs["key"] = "value"
    assert df.attrs["key"] == "value"


def test_attrs_update():
    """Test that attrs.update() works."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]}), npartitions=2)
    df.attrs.update({"key1": "value1", "key2": "value2"})
    assert df.attrs["key1"] == "value1"
    assert df.attrs["key2"] == "value2"


def test_invalid_attrs_assignment_raises():
    """Test that assigning a dict to attrs raises an error on next operation."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)

    # This is the wrong way to do it
    df.attrs = {"key": "value"}

    # Should raise RuntimeError on next wrapped operation
    with pytest.raises(RuntimeError, match="Invalid .attrs.*expected an accessor"):
        df.set_index("a")


def test_chained_operations():
    """Test that attrs survive chained operations."""
    df = dd.from_pandas(
        pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]}),
        npartitions=2,
    )
    df.attrs["experiment"] = "test123"

    result = df.set_index("a").drop("c", axis=1)[["b"]].copy()

    assert result.attrs["experiment"] == "test123"
    assert isinstance(result.attrs, AttrsAccessor)


# ============================================================================
# DataFrame wrapped methods tests
# ============================================================================


def test_dataframe_getitem_preserves_attrs():
    """Test that DataFrame.__getitem__ preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"

    # Single column (returns Series)
    result = df["a"]
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)

    # Multiple columns (returns DataFrame)
    result = df[["a", "b"]]
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_dataframe_compute_preserves_attrs():
    """Test that DataFrame.compute preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.compute()
    # compute returns a pandas DataFrame, which has attrs as a dict
    assert result.attrs["key"] == "value"


def test_dataframe_copy_preserves_attrs():
    """Test that DataFrame.copy preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.copy()
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_dataframe_drop_preserves_attrs():
    """Test that DataFrame.drop preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.drop("b", axis=1)
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_dataframe_map_partitions_preserves_attrs():
    """Test that DataFrame.map_partitions preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.map_partitions(lambda x: x * 2)
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_dataframe_set_index_preserves_attrs():
    """Test that DataFrame.set_index preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.set_index("a")
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


# ============================================================================
# Series wrapped methods tests
# ============================================================================


def test_series_getitem_preserves_attrs():
    """Test that Series.__getitem__ preserves attrs."""
    s = dd.from_pandas(pd.Series([1, 2, 3, 4, 5], name="test"), npartitions=2)
    s.attrs["key"] = "value"
    result = s[1:3]
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_series_compute_preserves_attrs():
    """Test that Series.compute preserves attrs."""
    s = dd.from_pandas(pd.Series([1, 2, 3], name="test"), npartitions=2)
    s.attrs["key"] = "value"
    result = s.compute()
    # compute returns a pandas Series, which has attrs as a dict
    assert result.attrs["key"] == "value"


def test_series_copy_preserves_attrs():
    """Test that Series.copy preserves attrs."""
    s = dd.from_pandas(pd.Series([1, 2, 3], name="test"), npartitions=2)
    s.attrs["key"] = "value"
    result = s.copy()
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


def test_series_map_partitions_preserves_attrs():
    """Test that Series.map_partitions preserves attrs."""
    s = dd.from_pandas(pd.Series([1, 2, 3], name="test"), npartitions=2)
    s.attrs["key"] = "value"
    result = s.map_partitions(lambda x: x * 2)
    assert result.attrs["key"] == "value"
    assert isinstance(result.attrs, AttrsAccessor)


# ============================================================================
# Indexer tests
# ============================================================================


def test_dataframe_loc_preserves_attrs():
    """Test that DataFrame.loc preserves attrs."""
    df = dd.from_pandas(
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=[10, 20, 30]),
        npartitions=2,
    )
    df.attrs["key"] = "value"
    result = df.loc[10:20]
    assert result.attrs["key"] == "value"


def test_dataframe_iloc_preserves_attrs():
    """Test that DataFrame.iloc preserves attrs."""
    df = dd.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), npartitions=2)
    df.attrs["key"] = "value"
    result = df.iloc[:, 0:1]
    assert result.attrs["key"] == "value"


def test_series_loc_preserves_attrs():
    """Test that Series.loc preserves attrs."""
    s = dd.from_pandas(
        pd.Series([1, 2, 3, 4, 5], index=[10, 20, 30, 40, 50], name="test"),
        npartitions=2,
    )
    s.attrs["key"] = "value"
    result = s.loc[10:30]
    assert result.attrs["key"] == "value"


# dd.Series do not have .iloc, hence there is no test_series_iloc_preserves_attrs() test
