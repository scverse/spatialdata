from __future__ import annotations

import copy

from spatialdata.models import C, X, Y, Z
from spatialdata.transformations.ngff.ngff_coordinate_system import (
    NgffAxis,
    NgffCoordinateSystem,
)

__all__ = "get_default_coordinate_system"

# "unit" is a default placeholder value. This is not suported by NGFF so the user should replace it before saving
x_axis = NgffAxis(name=X, type="space", unit="unit")
y_axis = NgffAxis(name=Y, type="space", unit="unit")
z_axis = NgffAxis(name=Z, type="space", unit="unit")
c_axis = NgffAxis(name=C, type="channel")
x_cs = NgffCoordinateSystem(name="x", axes=[x_axis])
y_cs = NgffCoordinateSystem(name="y", axes=[y_axis])
z_cs = NgffCoordinateSystem(name="z", axes=[z_axis])
c_cs = NgffCoordinateSystem(name="c", axes=[c_axis])
xy_cs = NgffCoordinateSystem(name="xy", axes=[x_axis, y_axis])
xyz_cs = NgffCoordinateSystem(name="xyz", axes=[x_axis, y_axis, z_axis])
yx_cs = NgffCoordinateSystem(name="yx", axes=[y_axis, x_axis])
zyx_cs = NgffCoordinateSystem(name="zyx", axes=[z_axis, y_axis, x_axis])
cyx_cs = NgffCoordinateSystem(name="cyx", axes=[c_axis, y_axis, x_axis])
czyx_cs = NgffCoordinateSystem(name="czyx", axes=[c_axis, z_axis, y_axis, x_axis])
_DEFAULT_COORDINATE_SYSTEM = {
    (X,): x_cs,
    (Y,): y_cs,
    (Z,): z_cs,
    (C,): c_cs,
    (X, Y): xy_cs,
    (X, Y, Z): xyz_cs,
    (Y, X): yx_cs,
    (Z, Y, X): zyx_cs,
    (C, Y, X): cyx_cs,
    (C, Z, Y, X): czyx_cs,
}


def get_default_coordinate_system(dims: tuple[str, ...]) -> NgffCoordinateSystem:
    """
    Get the default coordinate system

    Parameters
    ----------
    dims
        The dimension names to get the corresponding axes of the defeault coordinate system for.
        Names should be in ['x', 'y', 'z', 'c'].

    """
    axes = []
    for c in dims:
        if c == X:
            axes.append(copy.deepcopy(x_axis))
        elif c == Y:
            axes.append(copy.deepcopy(y_axis))
        elif c == Z:
            axes.append(copy.deepcopy(z_axis))
        elif c == C:
            axes.append(copy.deepcopy(c_axis))
        else:
            raise ValueError(f"Invalid dimension: {c}")
    return NgffCoordinateSystem(name="".join(dims), axes=axes)
