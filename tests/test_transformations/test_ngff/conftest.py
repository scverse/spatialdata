from spatialdata.transformations.ngff.ngff_coordinate_system import (
    NgffAxis,
    NgffCoordinateSystem,
)

x_axis = NgffAxis(name="x", type="space", unit="micrometer")
y_axis = NgffAxis(name="y", type="space", unit="micrometer")
z_axis = NgffAxis(name="z", type="space", unit="micrometer")
c_axis = NgffAxis(name="c", type="channel")
x_cs = NgffCoordinateSystem(name="x", axes=[x_axis])
y_cs = NgffCoordinateSystem(name="y", axes=[y_axis])
z_cs = NgffCoordinateSystem(name="z", axes=[z_axis])
c_cs = NgffCoordinateSystem(name="c", axes=[c_axis])
xy_cs = NgffCoordinateSystem(name="xy", axes=[x_axis, y_axis])
xyz_cs = NgffCoordinateSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
xyc_cs = NgffCoordinateSystem(name="xyc", axes=[x_axis, y_axis, c_axis])
xyzc_cs = NgffCoordinateSystem(name="xyzc", axes=[x_axis, y_axis, z_axis, c_axis])
yx_cs = NgffCoordinateSystem(name="yx", axes=[y_axis, x_axis])
zyx_cs = NgffCoordinateSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
cyx_cs = NgffCoordinateSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
czyx_cs = NgffCoordinateSystem(name="czyx", axes=[c_axis, z_axis, y_axis, x_axis])
