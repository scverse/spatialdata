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
    assert coord_sys.axes_names == ("x", "y", "z")

    assert [ax.type for ax in coord_sys._axes] == ["space", "space", "space"]
    assert coord_sys.axes_types == ("space", "space", "space")

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
    with pytest.raises(ValueError):
        CoordinateSystem(
            name="non unique axes names",
            axes=[
                Axis(name="x", type="space", unit="micrometer"),
                Axis(name="x", type="space", unit="micrometer"),
            ],
        )


def test_coordinate_system_exceptions():
    input_dict1 = copy.deepcopy(input_dict)
    input_dict1["axes"][0].pop("name")
    coord_sys = CoordinateSystem(name="test")
    with pytest.raises(ValueError):
        coord_sys.from_dict(input_dict1)

    input_dict2 = copy.deepcopy(input_dict)
    input_dict2["axes"][0].pop("type")
    coord_sys = CoordinateSystem(name="test")
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


def test_repr():
    cs = CoordinateSystem(
        "some coordinate system",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
            Axis("T", "time"),
        ],
    )
    expected = """CoordinateSystem('some coordinate system', [Axis('X', 'space', 'micrometers'), Axis('Y', 'space', 'meters'), Axis('T', 'time')])"""
    as_str = repr(cs)

    assert as_str == expected


def test_equal_up_to_the_units():
    cs1 = CoordinateSystem(
        "some coordinate system",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
            Axis("T", "time"),
        ],
    )
    cs2 = CoordinateSystem(
        "some coordinate systema",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
            Axis("T", "time"),
        ],
    )
    cs3 = CoordinateSystem(
        "some coordinate system",
        [
            Axis("X", "space", "gigameters"),
            Axis("Y", "space", ""),
            Axis("T", "time"),
        ],
    )

    assert cs1.equal_up_to_the_units(cs1)
    assert not cs1.equal_up_to_the_units(cs2)
    assert cs1.equal_up_to_the_units(cs3)


def test_subset_coordinate_system():
    cs = CoordinateSystem(
        "some coordinate system",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
            Axis("Z", "space", "meters"),
            Axis("T", "time"),
        ],
    )
    cs0 = cs.subset(["X", "Z"])
    cs1 = cs.subset(["X", "Y"], new_name="XY")
    assert cs0 == CoordinateSystem(
        "some coordinate system_subset ['X', 'Z']",
        [
            Axis("X", "space", "micrometers"),
            Axis("Z", "space", "meters"),
        ],
    )
    assert cs1 == CoordinateSystem(
        "XY",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
        ],
    )


def test_merge_coordinate_systems():
    cs0 = CoordinateSystem(
        "cs0",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
        ],
    )
    cs1 = CoordinateSystem(
        "cs1",
        [
            Axis("X", "space", "micrometers"),
        ],
    )
    cs2 = CoordinateSystem(
        "cs2",
        [
            Axis("X", "space", "meters"),
            Axis("Y", "space", "meters"),
        ],
    )
    cs3 = CoordinateSystem(
        "cs3",
        [
            Axis("Z", "space", "micrometers"),
        ],
    )
    assert cs0.merge(cs0, cs1) == CoordinateSystem(
        "cs0_merged_cs1",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
        ],
    )
    with pytest.raises(ValueError):
        CoordinateSystem.merge(cs0, cs2)
    assert CoordinateSystem.merge(cs0, cs3) == CoordinateSystem(
        "cs0_merged_cs3",
        [
            Axis("X", "space", "micrometers"),
            Axis("Y", "space", "meters"),
            Axis("Z", "space", "micrometers"),
        ],
    )
