import pytest

from spatialdata._core.coordinate_system import CoordinateSystem


def test_coordinate_system():
    input = {
        "name": "volume_micrometers",
        "axes": [
            {"name": "x", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "z", "type": "space", "unit": "micrometer"},
        ],
    }

    coord_sys = CoordinateSystem()
    coord_sys.from_dict(input)
    assert coord_sys.name == "volume_micrometers"
    assert coord_sys.axes == ["x", "y", "z"]
    assert coord_sys.types == ["space", "space", "space"]

    output = coord_sys.to_dict()
    assert input == output

    coord_manual = CoordinateSystem(
        name="volume_micrometers",
        axes=["x", "y", "z"],
        types=["space", "space", "space"],
        units=["micrometer", "micrometer", "micrometer"],
    )

    assert coord_manual.to_dict() == coord_sys.to_dict()

    input["axes"][0].pop("name")
    coord_sys = CoordinateSystem()
    with pytest.raises(ValueError):
        coord_sys.from_dict(input)
