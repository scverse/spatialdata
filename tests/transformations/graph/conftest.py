"""
Global pytest fixtures for graph transformations.
"""

from __future__ import annotations

import pytest

from spatialdata.literals import NgffAxisType
from spatialdata.transformations._graph import Axis, CoordSystem


# ----------------------------------------------------------------------
# Simple axes – one per common name / type combination
# ----------------------------------------------------------------------
@pytest.fixture
def axis_x_space() -> Axis:
    """Axis(name='x', type=SPACE)"""
    return Axis(name="x", type=NgffAxisType.SPACE)


@pytest.fixture
def axis_y_space() -> Axis:
    """Axis(name='y', type=SPACE)"""
    return Axis(name="y", type=NgffAxisType.SPACE)


@pytest.fixture
def axis_z_space() -> Axis:
    """Axis(name='z', type=SPACE)"""
    return Axis(name="z", type=NgffAxisType.SPACE)


@pytest.fixture
def axis_c_channel() -> Axis:
    """Axis(name='c', type=CHANNEL)"""
    return Axis(name="c", type=NgffAxisType.CHANNEL)


# ----------------------------------------------------------------------
# Axes with optional metadata
# ----------------------------------------------------------------------
@pytest.fixture
def axis_x_space_with_unit() -> Axis:
    """Axis(name='x', type=SPACE, unit='micrometer')"""
    return Axis(name="x", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_y_space_with_unit() -> Axis:
    """Axis(name='y', type=SPACE, unit='micrometer')"""
    return Axis(name="y", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_z_space_with_unit() -> Axis:
    """Axis(name='z', type=SPACE, unit='micrometer')"""
    return Axis(name="z", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_a_space_with_unit() -> Axis:
    """Axis(name='a', type=SPACE, unit='micrometer')"""
    return Axis(name="a", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_b_space_with_unit() -> Axis:
    """Axis(name='b', type=SPACE, unit='micrometer')"""
    return Axis(name="b", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_c_space_with_unit() -> Axis:
    """Axis(name='x', type=SPACE, unit='micrometer')"""
    return Axis(name="c", type=NgffAxisType.SPACE, unit="micrometer")


@pytest.fixture
def axis_x_space_with_long_name() -> Axis:
    """Axis(name='x', type=SPACE, long_name='X axis')"""
    return Axis(name="x", type=NgffAxisType.SPACE, long_name="X axis")


@pytest.fixture
def axis_x_space_full() -> Axis:
    """Axis(name='x', type=SPACE, unit='micrometer', long_name='X axis')"""
    return Axis(
        name="x",
        type=NgffAxisType.SPACE,
        unit="micrometer",
        long_name="X axis",
    )


# ----------------------------------------------------------------------
# Composite groups that appear frequently in the code base
# ----------------------------------------------------------------------
@pytest.fixture
def axes_xy(axis_x_space, axis_y_space) -> tuple[Axis, ...]:
    """Typical 2‑D spatial system: ('x', 'y')."""
    return axis_x_space, axis_y_space


@pytest.fixture
def axes_zyx(axis_z_space, axis_y_space, axis_x_space) -> tuple[Axis, ...]:
    """Typical 3‑D spatial system in reverse order: ('z', 'y', 'x')."""
    return axis_z_space, axis_y_space, axis_x_space


@pytest.fixture
def axes_czyx(axis_c_channel, axis_z_space, axis_y_space, axis_x_space) -> tuple[Axis, ...]:
    """Channel plus full 3‑D spatial axes: ('c', 'z', 'y', 'x')."""
    return axis_c_channel, axis_z_space, axis_y_space, axis_x_space


@pytest.fixture
def axes_cyx(axis_c_channel, axis_y_space, axis_x_space) -> tuple[Axis, ...]:
    """Channel plus 2‑D spatial axes: ('c', 'y', 'x')."""
    return axis_c_channel, axis_y_space, axis_x_space


# ----------------------------------------------------------------------
# CoordSystem Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def x_cs(axis_x_space_with_unit) -> CoordSystem:
    return CoordSystem(name="x", axes=(axis_x_space_with_unit,), virtual=False)


@pytest.fixture
def y_cs(axis_y_space_with_unit) -> CoordSystem:
    return CoordSystem(name="y", axes=(axis_y_space_with_unit,), virtual=False)


@pytest.fixture
def z_cs(axis_z_space_with_unit) -> CoordSystem:
    return CoordSystem(name="z", axes=(axis_z_space_with_unit,), virtual=False)


@pytest.fixture
def c_cs(axis_c_channel) -> CoordSystem:
    return CoordSystem(name="c", axes=(axis_c_channel,), virtual=False)


@pytest.fixture
def xy_cs(axis_x_space_with_unit, axis_y_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="xy",
        axes=(
            axis_x_space_with_unit,
            axis_y_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def yx_cs(axis_y_space_with_unit, axis_x_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="yx",
        axes=(
            axis_y_space_with_unit,
            axis_x_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def xyz_cs(axis_x_space_with_unit, axis_y_space_with_unit, axis_z_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="xyz",
        axes=(
            axis_x_space_with_unit,
            axis_y_space_with_unit,
            axis_z_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def zyx_cs(axis_z_space_with_unit, axis_y_space_with_unit, axis_x_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="zyx",
        axes=(
            axis_z_space_with_unit,
            axis_y_space_with_unit,
            axis_x_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def czyx_cs(axis_c_channel, axis_z_space_with_unit, axis_y_space_with_unit, axis_x_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="czyx",
        axes=(
            axis_c_channel,
            axis_z_space_with_unit,
            axis_y_space_with_unit,
            axis_x_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def xyc_cs(axis_x_space_with_unit, axis_y_space_with_unit, axis_c_channel) -> CoordSystem:
    return CoordSystem(
        name="xyc",
        axes=(
            axis_x_space_with_unit,
            axis_y_space_with_unit,
            axis_c_channel,
        ),
        virtual=False,
    )


@pytest.fixture
def cyx_cs(axis_c_channel, axis_y_space_with_unit, axis_x_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="cyx",
        axes=(
            axis_c_channel,
            axis_y_space_with_unit,
            axis_x_space_with_unit,
        ),
        virtual=False,
    )


@pytest.fixture
def abc_cs(axis_a_space_with_unit, axis_b_space_with_unit, axis_c_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="abc", axes=(axis_a_space_with_unit, axis_b_space_with_unit, axis_c_space_with_unit), virtual=False
    )


@pytest.fixture
def cba_cs(axis_c_space_with_unit, axis_b_space_with_unit, axis_a_space_with_unit) -> CoordSystem:
    return CoordSystem(
        name="cba",
        axes=(
            axis_c_space_with_unit,
            axis_b_space_with_unit,
            axis_a_space_with_unit,
        ),
        virtual=False,
    )
