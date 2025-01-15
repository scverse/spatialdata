import pandas as pd
import pytest

from spatialdata.datasets import blobs


@pytest.fixture
def sdata_attrs():
    sdata = blobs()
    sdata.attrs["test"] = {"a": {"b": 12}, "c": 8}
    return sdata


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


def test_empty_attrs():
    sdata = blobs()
    with pytest.raises(KeyError, match="was not found in sdata.attrs."):
        sdata.get_attrs(key="test")
