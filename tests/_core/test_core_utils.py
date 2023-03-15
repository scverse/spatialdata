import pytest

from spatialdata.models._utils import validate_axis_name


def test_validate_axis_name():
    for ax in ["c", "x", "y", "z"]:
        validate_axis_name(ax)
    with pytest.raises(TypeError):
        validate_axis_name("invalid")
