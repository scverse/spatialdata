from pathlib import Path

import numpy as np

from spatialdata import SpatialData
from spatialdata._core.coordinate_system import Axis, CoordinateSystem
from spatialdata._core.transformations import Scale

x_axis = Axis(name="x", type="space", unit="micrometer")
y_axis = Axis(name="y", type="space", unit="micrometer")
z_axis = Axis(name="z", type="space", unit="micrometer")
c_axis = Axis(name="c", type="channel")


def test_io_points(tmp_path: str, points: SpatialData):
    tmpdir = Path(tmp_path) / "tmp.zarr"
    transform = Scale(np.array([1, 2, 3]))
    input = CoordinateSystem(name="xy", axes=[x_axis, y_axis])
    output = CoordinateSystem(name="xy", axes=[x_axis, y_axis])
    transform.input_coordinate_system = input
    transform.output_coordinate_system = output
    points.points["points_0"].uns["transform"] = transform

    points.write(tmpdir)
    new_sdata = SpatialData.read(tmpdir)
    print(new_sdata.points["points_0"].uns["transform"].to_dict())
    assert new_sdata.points["points_0"].uns["transform"] == transform
