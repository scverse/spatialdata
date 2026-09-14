from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spatialdata import SpatialData


@pytest.fixture
def sdata_attrs(sdata_blobs):
    sdata_blobs.attrs["test"] = {"a": {"b": 12}, "c": 8}
    return sdata_blobs


def test_get_attrs_as_is(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as=None, flatten=False)
    expected = {"a": {"b": 12}, "c": 8}
    assert result == expected


def test_get_attrs_as_dict_flatten(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as="dict", flatten=True)
    expected = {"a_b": 12, "c": 8}
    assert result == expected


def test_get_attrs_as_json_flatten_false(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as="json", flatten=False)
    expected = '{"a": {"b": 12}, "c": 8}'
    assert result == expected


def test_get_attrs_as_json_flatten_true(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as="json", flatten=True)
    expected = '{"a_b": 12, "c": 8}'
    assert result == expected


def test_get_attrs_as_dataframe_flatten_false(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as="df", flatten=False)
    expected = pd.DataFrame([{"a": {"b": 12}, "c": 8}])
    pd.testing.assert_frame_equal(result, expected)


def test_get_attrs_as_dataframe_flatten_true(sdata_attrs):
    result = sdata_attrs.get_attrs(key="test", return_as="df", flatten=True)
    expected = pd.DataFrame([{"a_b": 12, "c": 8}])
    pd.testing.assert_frame_equal(result, expected)


# test invalid cases
def test_invalid_key(sdata_attrs):
    with pytest.raises(KeyError, match="was not found in sdata.attrs"):
        sdata_attrs.get_attrs(key="non_existent_key")


def test_invalid_return_as_value(sdata_attrs):
    with pytest.raises(ValueError, match="Invalid 'return_as' value"):
        sdata_attrs.get_attrs(key="test", return_as="invalid_option")


def test_non_string_key(sdata_attrs):
    with pytest.raises(TypeError, match="The key must be a string."):
        sdata_attrs.get_attrs(key=123)


def test_non_string_sep(sdata_attrs):
    with pytest.raises(TypeError, match="Parameter 'sep_for_nested_keys' must be a string."):
        sdata_attrs.get_attrs(key="test", sep=123)


def test_empty_attrs(sdata_blobs):
    with pytest.raises(KeyError, match="was not found in sdata.attrs."):
        sdata_blobs.get_attrs(key="test")


# the attrs must be JSON-serializable, since they are stored as Zarr attributes; the tests below cover the values
# that satisfy this invariant (and therefore survive a write/read round-trip), and the ones that violate it
JSONABLE_VALUES = [
    pytest.param({"a": {"b": 12}, "c": 8}, id="dict"),
    pytest.param([1, 2, 3], id="list"),
    pytest.param([{"a": 1}, [2, None]], id="nested_list"),
    pytest.param("a_string", id="str"),
    pytest.param(42, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="none"),
]

NON_JSONABLE_VALUES = [
    pytest.param(np.array([1, 2]), id="ndarray"),
    pytest.param(np.int64(5), id="numpy_scalar"),
    pytest.param({1, 2}, id="set"),
    pytest.param(pd.DataFrame({"a": [1]}), id="dataframe"),
    pytest.param(1 + 2j, id="complex"),
    pytest.param(b"some_bytes", id="bytes"),
]


@pytest.mark.parametrize("value", JSONABLE_VALUES)
def test_attrs_jsonable_value_roundtrips(full_sdata, tmp_path, value):
    """A JSON-serializable value is returned unchanged by `get_attrs` and survives a write/read round-trip."""
    full_sdata.attrs["test"] = value
    assert full_sdata.get_attrs(key="test", return_as=None, flatten=False) == value

    f = tmp_path / "data.zarr"
    full_sdata.write(f)
    assert SpatialData.read(f).attrs["test"] == value


@pytest.mark.parametrize("value", NON_JSONABLE_VALUES)
def test_attrs_non_jsonable_value_cannot_be_written(full_sdata, tmp_path, value):
    """A value that is not JSON-serializable violates the attrs invariant and is rejected when writing."""
    full_sdata.attrs["test"] = value
    f = tmp_path / "data.zarr"
    with pytest.raises(TypeError, match="Invalid attribute in SpatialData.attrs"):
        full_sdata.write(f)


def test_get_attrs_non_mapping_ignores_flatten(full_sdata):
    """`flatten` only applies to mappings, so a scalar or a list is returned as-is."""
    full_sdata.attrs["test"] = [1, 2, 3]
    assert full_sdata.get_attrs(key="test", return_as=None, flatten=True) == [1, 2, 3]


def test_get_attrs_non_dict_as_dict_raises(full_sdata):
    full_sdata.attrs["test"] = [1, 2, 3]
    with pytest.raises(TypeError, match="Cannot convert non-dictionary data to a dictionary."):
        full_sdata.get_attrs(key="test", return_as="dict")


@pytest.mark.parametrize("value", [[1, 2, 3], 42, "a_string", None])
def test_get_attrs_non_dict_as_json(full_sdata, value):
    full_sdata.attrs["test"] = value
    assert full_sdata.get_attrs(key="test", return_as="json") == json.dumps(value)
