from pathlib import Path

from spatialdata._core.coordinate_system import Axis, CoordinateSystem

x_axis = Axis(name="x", type="space", unit="micrometer")
y_axis = Axis(name="y", type="space", unit="micrometer")
z_axis = Axis(name="z", type="space", unit="micrometer")
c_axis = Axis(name="c", type="channel")
x_cs = CoordinateSystem(name="x", axes=[x_axis])
y_cs = CoordinateSystem(name="y", axes=[y_axis])
z_cs = CoordinateSystem(name="z", axes=[z_axis])
c_cs = CoordinateSystem(name="c", axes=[c_axis])
xy_cs = CoordinateSystem(name="xy", axes=[x_axis, y_axis])
xyz_cs = CoordinateSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
xyc_cs = CoordinateSystem(name="xyc", axes=[x_axis, y_axis, c_axis])
xyzc_cs = CoordinateSystem(name="xyzc", axes=[x_axis, y_axis, z_axis, c_axis])
yx_cs = CoordinateSystem(name="yx", axes=[y_axis, x_axis])
zyx_cs = CoordinateSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
cyx_cs = CoordinateSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
czyx_cs = CoordinateSystem(name="czyx", axes=[c_axis, z_axis, y_axis, x_axis])


POLYGON_PATH = Path(__file__).parent.parent / "data/polygon.json"
MULTIPOLYGON_PATH = Path(__file__).parent.parent / "data/polygon.json"
POINT_PATH = Path(__file__).parent.parent / "data/points.json"
