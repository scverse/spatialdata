from __future__ import annotations

from spatialdata.transformations.graph.vert import Axis, CoordSystem

x_axis = Axis(name="x", type="space", unit="micrometer")
y_axis = Axis(name="y", type="space", unit="micrometer")
z_axis = Axis(name="z", type="space", unit="micrometer")
c_axis = Axis(name="c", type="channel")

a_axis = Axis(name="a", type="space", unit="micrometer")
b_axis = Axis(name="b", type="space", unit="micrometer")
c_axis = Axis(name="c", type="space", unit="micrometer")

x_cs = CoordSystem(name="x", axes=(x_axis,), virtual=False)
y_cs = CoordSystem(name="y", axes=(y_axis,), virtual=False)
z_cs = CoordSystem(name="z", axes=(z_axis,), virtual=False)
c_cs = CoordSystem(name="c", axes=(c_axis,), virtual=False)
xy_cs = CoordSystem(
    name="xy",
    axes=(
        x_axis,
        y_axis,
    ),
    virtual=False,
)
yx_cs = CoordSystem(
    name="yx",
    axes=(
        y_axis,
        x_axis,
    ),
    virtual=False,
)
xyz_cs = CoordSystem(
    name="xyz",
    axes=(
        x_axis,
        y_axis,
        z_axis,
    ),
    virtual=False,
)
zyx_cs = CoordSystem(
    name="zyx",
    axes=(
        z_axis,
        y_axis,
        x_axis,
    ),
    virtual=False,
)
xyc_cs = CoordSystem(
    name="xyc",
    axes=(
        x_axis,
        y_axis,
        c_axis,
    ),
    virtual=False,
)
cyx_cs = CoordSystem(
    name="cyx",
    axes=(
        c_axis,
        y_axis,
        x_axis,
    ),
    virtual=False,
)

abc_cs = CoordSystem(name="abc", axes=(a_axis, b_axis, c_axis), virtual=False)
cba_cs = CoordSystem(
    name="cba",
    axes=(
        c_axis,
        b_axis,
        a_axis,
    ),
    virtual=False,
)
cba_cs = CoordSystem(name="cba", axes=(c_axis, b_axis, a_axis), virtual=False)
