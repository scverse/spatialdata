from __future__ import annotations

from spatialdata.transformations.graph.vert import Axis, CoordSystem

x_axis = Axis(name="x", type="space", unit="micrometer")
y_axis = Axis(name="y", type="space", unit="micrometer")
z_axis = Axis(name="z", type="space", unit="micrometer")
c_axis = Axis(name="c", type="channel")

x_cs = CoordSystem(name="x", axes=[x_axis])
y_cs = CoordSystem(name="y", axes=[y_axis])
z_cs = CoordSystem(name="z", axes=[z_axis])
c_cs = CoordSystem(name="c", axes=[c_axis])
xy_cs = CoordSystem(name="xy", axes=[x_axis, y_axis])
yx_cs = CoordSystem(name="yx", axes=[y_axis, x_axis])
xyz_cs = CoordSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
zyx_cs = CoordSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
xyc_cs = CoordSystem(name="xyc", axes=[x_axis, y_axis, c_axis])
cyx_cs = CoordSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
