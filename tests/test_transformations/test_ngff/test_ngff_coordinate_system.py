import copy
import json

import pytest

from spatialdata.transformations.ngff.ngff_coordinate_system import (
    NgffAxis,
    NgffCoordinateSystem,
)

input_dict = {
    "name": "volume_micrometers",
    "axes": [
        {"name": "x", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "z", "type": "space", "unit": "micrometer"},
    ],
}


def test_coordinate_system_instantiation_and_properties():
    coord_sys = NgffCoordinateSystem.from_dict(input_dict)
    assert coord_sys.name == "volume_micrometers"
    assert [ax.name for ax in coord_sys._axes] == ["x", "y", "z"]
    assert coord_sys.axes_names == ("x", "y", "z")

    assert [ax.type for ax in coord_sys._axes] == ["space", "space", "space"]
    assert coord_sys.axes_types == ("space", "space", "space")

    output_dict = coord_sys.to_dict()
    assert input_dict == output_dict

    axes = [
        NgffAxis(name="x", type="space", unit="micrometer"),
        NgffAxis(name="y", type="space", unit="micrometer"),
        NgffAxis(name="z", type="space", unit="micrometer"),
    ]
    coord_manual = NgffCoordinateSystem(
        name="volume_micrometers",
        axes=axes,
    )

    assert coord_manual.to_dict() == coord_sys.to_dict()
    with pytest.raises(ValueError):
        NgffCoordinateSystem(
            name="non unique axes names",
            axes=[
                NgffAxis(name="x", type="space", unit="micrometer"),
                NgffAxis(name="x", type="space", unit="micrometer"),
            ],
        )


def test_coordinate_system_exceptions():
    input_dict1 = copy.deepcopy(input_dict)
    input_dict1["axes"][0].pop("name")
    coord_sys = NgffCoordinateSystem(name="test")
    with pytest.raises(ValueError):
        coord_sys.from_dict(input_dict1)

    input_dict2 = copy.deepcopy(input_dict)
    input_dict2["axes"][0].pop("type")
    coord_sys = NgffCoordinateSystem(name="test")
    with pytest.raises(ValueError):
        coord_sys.from_dict(input_dict2)

    # not testing the cases with axis.type in ['channel', 'array'] as all the more complex checks are handled by the
    # validation schema


def test_coordinate_system_roundtrip():
    input_json = json.dumps(input_dict)
    cs = NgffCoordinateSystem.from_json(input_json)
    output_json = cs.to_json()
    assert input_json == output_json
    cs2 = NgffCoordinateSystem.from_json(output_json)
    assert cs == cs2


def test_repr():
    cs = NgffCoordinateSystem(
        "some coordinate system",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
            NgffAxis("T", "time"),
        ],
    )
    expected = (
        "NgffCoordinateSystem('some coordinate system',"
        + " [NgffAxis('X', 'space', 'micrometers'),"
        + " NgffAxis('Y', 'space', 'meters'), NgffAxis('T', 'time')])"
    )
    as_str = repr(cs)

    assert as_str == expected


def test_equal_up_to_the_units():
    cs1 = NgffCoordinateSystem(
        "some coordinate system",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
            NgffAxis("T", "time"),
        ],
    )
    cs2 = NgffCoordinateSystem(
        "some coordinate systema",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
            NgffAxis("T", "time"),
        ],
    )
    cs3 = NgffCoordinateSystem(
        "some coordinate system",
        [
            NgffAxis("X", "space", "gigameters"),
            NgffAxis("Y", "space", ""),
            NgffAxis("T", "time"),
        ],
    )

    assert cs1.equal_up_to_the_units(cs1)
    assert not cs1.equal_up_to_the_units(cs2)
    assert cs1.equal_up_to_the_units(cs3)


def test_subset_coordinate_system():
    cs = NgffCoordinateSystem(
        "some coordinate system",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
            NgffAxis("Z", "space", "meters"),
            NgffAxis("T", "time"),
        ],
    )
    cs0 = cs.subset(["X", "Z"])
    cs1 = cs.subset(["X", "Y"], new_name="XY")
    assert cs0 == NgffCoordinateSystem(
        "some coordinate system_subset ['X', 'Z']",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Z", "space", "meters"),
        ],
    )
    assert cs1 == NgffCoordinateSystem(
        "XY",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
        ],
    )


def test_merge_coordinate_systems():
    cs0 = NgffCoordinateSystem(
        "cs0",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
        ],
    )
    cs1 = NgffCoordinateSystem(
        "cs1",
        [
            NgffAxis("X", "space", "micrometers"),
        ],
    )
    cs2 = NgffCoordinateSystem(
        "cs2",
        [
            NgffAxis("X", "space", "meters"),
            NgffAxis("Y", "space", "meters"),
        ],
    )
    cs3 = NgffCoordinateSystem(
        "cs3",
        [
            NgffAxis("Z", "space", "micrometers"),
        ],
    )
    assert cs0.merge(cs0, cs1) == NgffCoordinateSystem(
        "cs0_merged_cs1",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
        ],
    )
    with pytest.raises(ValueError):
        NgffCoordinateSystem.merge(cs0, cs2)
    assert NgffCoordinateSystem.merge(cs0, cs3) == NgffCoordinateSystem(
        "cs0_merged_cs3",
        [
            NgffAxis("X", "space", "micrometers"),
            NgffAxis("Y", "space", "meters"),
            NgffAxis("Z", "space", "micrometers"),
        ],
    )
