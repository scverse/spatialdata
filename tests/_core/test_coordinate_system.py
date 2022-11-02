import copy
import json

import pytest

from spatialdata._core.coordinate_system import Axis, CoordinateSystem

input_dict = {
    "name": "volume_micrometers",
    "axes": [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"},
    ],
}


def test_coordinate_system_instantiation_and_properties():
    coord_sys = CoordinateSystem.from_dict(input_dict)
    assert coord_sys.name == "volume_micrometers"
    assert [ax.name for ax in coord_sys._axes] == ["x", "y", "z"]
    assert coord_sys.axes_names == ["x", "y", "z"]

    assert [ax.type for ax in coord_sys._axes] == ["space", "space", "space"]
    assert coord_sys.axes_types == ["space", "space", "space"]

    output_dict = coord_sys.to_dict()
    assert input_dict == output_dict

    axes = [
        Axis(name="x", type="space", unit="micrometer"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="z", type="space", unit="micrometer"),
    ]
    coord_manual = CoordinateSystem(
        name="volume_micrometers",
        axes=axes,
    )

    assert coord_manual.to_dict() == coord_sys.to_dict()


def test_coordinate_system_exceptions():
    input_dict1 = copy.deepcopy(input_dict)
    input_dict1["axes"][0].pop("name")
    coord_sys = CoordinateSystem()
    with pytest.raises(ValueError):
        coord_sys.from_dict(input_dict1)

    input_dict2 = copy.deepcopy(input_dict)
    input_dict2["axes"][0].pop("type")
    coord_sys = CoordinateSystem()
    with pytest.raises(ValueError):
        coord_sys.from_dict(input_dict2)

    # not testing the cases with axis.type in ['channel', 'array'] as all the more complex checks are handled by the
    # validation schema


def test_coordinate_system_roundtrip():
    input_json = json.dumps(input_dict)
    cs = CoordinateSystem.from_json(input_json)
    output_json = cs.to_json()
    assert input_json == output_json
    cs2 = CoordinateSystem.from_json(output_json)
    assert cs == cs2
