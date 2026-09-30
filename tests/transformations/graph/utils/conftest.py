"""
Global pytest fixtures that provide ready‑made ``Axis`` instances.
These fixtures are deliberately small – any test can import the one it needs
and combine them arbitrarily.
"""

from __future__ import annotations

import pytest

from spatialdata.transformations._graph import Axis, NgffAxisType


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
# Axes with optional metadata – useful for “prefer non‑None” tests
# ----------------------------------------------------------------------
@pytest.fixture
def axis_x_space_with_unit() -> Axis:
    """Axis(name='x', type=SPACE, unit='micrometer')"""
    return Axis(name="x", type=NgffAxisType.SPACE, unit="micrometer")


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
