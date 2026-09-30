"""
Test suite for ``get_union_of_axes`` that uses the reusable ``Axis``
fixtures defined in the top‑level ``conftest.py``.
Each logical behavior is expressed as a separate test function; when the
behavior is identical we use ``@pytest.mark.parametrize`` to drive the
different input combinations.
"""

from __future__ import annotations

import pytest

from spatialdata.transformations._graph import Axis, NgffAxisType, get_union_of_axes


# ----------------------------------------------------------------------
# 1. Reverse‑alphabetical ordering (basic union & mixed channel/space)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "axes_fixture_names,expected_names",
    [
        # three pure spatial axes
        (("axis_x_space", "axis_y_space", "axis_z_space"), ("z", "y", "x")),
        # channel + full 3‑D spatial axes
        (("axis_c_channel", "axis_x_space", "axis_y_space", "axis_z_space"), ("z", "y", "x", "c")),
        # mixed groups: xy + full 3‑D (duplicates should be removed)
        (("axes_xy", "axes_zyx"), ("z", "y", "x")),
        # mixed groups: cyx + full 3‑D (duplicates removed, c kept)
        (("axes_cyx", "axes_zyx"), ("z", "y", "x", "c")),
    ],
    ids=[
        "basic_spatial",
        "channel_plus_spatial",
        "xy_plus_zyx",
        "cyx_plus_zyx",
    ],
)
def test_reverse_alphabetical_order(axes_fixture_names, expected_names, request):
    """
    Verify reverse‑alphabetical ordering of the resulting axis names.
    The fixture names are resolved to the actual ``Axis`` objects.
    """
    # Resolve each name (whether a single Axis fixture or a composite tuple fixture)
    resolved = []
    for name in axes_fixture_names:
        obj = request.getfixturevalue(name)
        # Composite fixtures return a tuple; flatten them
        if isinstance(obj, tuple):
            resolved.extend(obj)
        else:
            resolved.append(obj)

    result = get_union_of_axes(tuple(resolved))
    assert tuple(ax.name for ax in result) == expected_names


# ----------------------------------------------------------------------
# 2. Duplicate removal when name and type are identical
# ----------------------------------------------------------------------
def test_duplicate_removal(axis_x_space, axis_y_space):
    """
    Two identical ``x`` axes should be deduplicated, leaving ``y`` untouched.
    """
    result = get_union_of_axes((axis_x_space, axis_x_space, axis_y_space))
    assert len(result) == 2
    assert result[0].name == "y"
    assert result[1].name == "x"


# ----------------------------------------------------------------------
# 3. Same name, different type – both must be kept
# ----------------------------------------------------------------------
def test_same_name_different_type(axis_x_space):
    """
    ``x`` as a spatial axis and ``x`` as a channel axis are distinct.
    """
    axis_x_channel = Axis(name="x", type=NgffAxisType.CHANNEL)  # ad‑hoc, no fixture needed
    result = get_union_of_axes((axis_x_space, axis_x_channel))
    assert len(result) == 2
    # Reverse alphabetical: spatial ``x`` comes before channel ``x``
    assert result == (axis_x_space, axis_x_channel)


# ----------------------------------------------------------------------
# 4. Prefer non‑None ``unit`` over ``None``
# ----------------------------------------------------------------------
def test_prefer_non_none_unit(axis_x_space, axis_x_space_with_unit):
    """
    Between an axis with ``unit=None`` and one with a concrete unit,
    the latter must survive duplicate removal.
    """
    result = get_union_of_axes((axis_x_space, axis_x_space_with_unit))
    assert len(result) == 1
    assert result[0].unit == "micrometer"


# ----------------------------------------------------------------------
# 5. Prefer non‑None ``long_name`` over ``None``
# ----------------------------------------------------------------------
def test_prefer_non_none_long_name(
    axis_x_space,
    axis_x_space_with_long_name,
):
    """
    When only one duplicate defines ``long_name``, that value must be kept.
    """
    result = get_union_of_axes((axis_x_space, axis_x_space_with_long_name))
    assert len(result) == 1
    assert result[0].long_name == "X axis"


# ----------------------------------------------------------------------
# 6. ``unit`` outranks ``long_name`` when they are on different duplicates
# ----------------------------------------------------------------------
def test_unit_beats_long_name(
    axis_x_space_with_unit,
    axis_x_space_with_long_name,
):
    """
    The sorting key places ``unit_not_none`` before ``long_name_not_none``,
    therefore the axis that defines a unit must be chosen even if another
    duplicate defines a long name.
    """
    result = get_union_of_axes((axis_x_space_with_long_name, axis_x_space_with_unit))
    assert len(result) == 1
    assert result[0].unit == "micrometer"
    assert result[0].long_name is None


# ----------------------------------------------------------------------
# 7. Full hierarchy – axis with both optional fields wins
# ----------------------------------------------------------------------
def test_full_hierarchy(
    axis_x_space_full,
    axis_x_space_with_unit,
    axis_x_space,
):
    """
    Three duplicate ``x`` axes:
      * full (unit + long_name)
      * unit only
      * nothing
    The most complete axis must be kept.
    """
    result = get_union_of_axes((axis_x_space, axis_x_space_with_unit, axis_x_space_full))
    assert len(result) == 1
    kept = result[0]
    assert kept.unit == "micrometer"
    assert kept.long_name == "X axis"


# ----------------------------------------------------------------------
# 8. Empty input & single‑axis sanity checks
# ----------------------------------------------------------------------
def test_empty_input():
    """Empty tuple must return an empty tuple."""
    assert get_union_of_axes(()) == ()


def test_single_axis(axis_x_space):
    """A single axis should be returned unchanged."""
    result = get_union_of_axes((axis_x_space,))
    assert len(result) == 1
    assert result[0] is axis_x_space
