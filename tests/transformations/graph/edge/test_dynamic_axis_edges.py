from __future__ import annotations

import numpy as np
import pytest

from spatialdata.transformations._graph import DynamicCreateAxisEdge, DynamicDropAxisEdge, NonZeroDroppedAxisError


# ----------------------------------------------------------------------
#  DynamicCreateAxisEdge – creation logic
# ----------------------------------------------------------------------
def test_dynamic_create_axis_edge_no_created_axes(xyz_cs):
    """When input == output, ``_created_axes`` is empty."""
    edge = DynamicCreateAxisEdge(input=xyz_cs, output=xyz_cs)
    assert edge._created_axes == set()


def test_dynamic_create_axis_edge_single_created_axis(xy_cs, cyx_cs):
    """When ``c`` is missing from the input, it is reported as a created axis."""
    edge = DynamicCreateAxisEdge(input=xy_cs, output=cyx_cs)
    created = {ax.name for ax in edge._created_axes}
    assert created == {"c"}


def test_dynamic_create_axis_edge_multiple_created_axes(xy_cs, czyx_cs):
    """When ``a`` and ``b`` are missing from the input, both are reported."""
    edge = DynamicCreateAxisEdge(input=xy_cs, output=czyx_cs)
    created = {ax.name for ax in edge._created_axes}
    assert created == {"c", "z"}


def test_dynamic_create_axis_edge_inverse(xy_cs, czyx_cs):
    """Inverse of a create edge must be a drop‑edge with swapped coordinate systems."""
    edge = DynamicCreateAxisEdge(input=xy_cs, output=czyx_cs)
    inv = edge.inverse()
    assert isinstance(inv, DynamicDropAxisEdge)
    assert inv.input is edge.output
    assert inv.output is edge.input


def test_dynamic_create_axis_edge_to_affine(xy_cs, xyc_cs):
    """``to_affine`` must delegate to a temporary ``ProjectAxisEdge``."""
    edge = DynamicCreateAxisEdge(input=xy_cs, output=xyc_cs)
    affine = edge.to_affine()
    expected_proj_affine = edge._proj_edge().to_affine()
    assert np.allclose(affine.linear, expected_proj_affine.linear)
    assert np.allclose(affine.translation, expected_proj_affine.translation)
    assert affine.input is edge.input
    assert affine.output is edge.output


def test_dynamic_create_axis_edge_repr_contains_created_axes(xy_cs, xyc_cs):
    """``repr`` must list the created axes and must not contain a drops‑section."""
    edge = DynamicCreateAxisEdge(input=xy_cs, output=xyc_cs)
    rep = repr(edge)
    assert "+ creates axes:" in rep
    assert "c" in rep
    assert "- drops axes:" not in rep


# ----------------------------------------------------------------------
#  DynamicDropAxisEdge – dropping logic
# ----------------------------------------------------------------------
def test_dynamic_drop_axis_edge_no_dropped_axes(xyz_cs):
    """When input == output, ``_dropped_axes`` is empty."""
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xyz_cs)
    assert edge._dropped_axes == set()


def test_dynamic_drop_axis_edge_single_dropped_axis(xyz_cs, xy_cs):
    """When ``z`` is missing from the output, it is reported as a dropped axis."""
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xy_cs)
    dropped = {ax.name for ax in edge._dropped_axes}
    assert dropped == {"z"}


def test_dynamic_drop_axis_edge_multiple_dropped_axes(czyx_cs, xy_cs):
    """When ``a`` and ``b`` are missing from the output, both are reported."""
    edge = DynamicDropAxisEdge(input=czyx_cs, output=xy_cs)
    dropped = {ax.name for ax in edge._dropped_axes}
    assert dropped == {"c", "z"}


def test_dynamic_drop_axis_edge_inverse(zyx_cs, xy_cs):
    """Inverse of a drop‑edge must be a create-edge with swapped coordinate systems."""
    edge = DynamicDropAxisEdge(input=zyx_cs, output=xy_cs)
    inv = edge.inverse()
    assert isinstance(inv, DynamicCreateAxisEdge)
    assert inv.input is edge.output
    assert inv.output is edge.input


def test_dynamic_drop_axis_edge_to_affine(xyz_cs, xy_cs):
    """``to_affine`` of a drop‑edge must produce an affine with linear part whose shape
    reflects the output‑by‑input dimensionality (2×3 in this case)."""
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xy_cs)
    affine = edge.to_affine()
    assert affine.linear.shape == (2, 3)  # 2 output axes, 3 input axes


def test_dynamic_drop_axis_edge_pre_transform_check_passes(xyz_cs, xy_cs):
    """Zero‑check passes when the dropped axis (z) contains only zeros."""
    points = np.array([[1, 2, 0], [3, 4, 0]], dtype=float)
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xy_cs)  # drops ``z``
    edge.pre_transforming_check(points)  # should not raise


def test_dynamic_drop_axis_edge_pre_transform_check_fails(xyz_cs, xy_cs):
    """Zero‑check raises ``NonZeroDroppedAxisError`` when a dropped axis is non‑zero."""
    points = np.array([[1, 2, 5], [3, 4, 0]], dtype=float)
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xy_cs)  # drops ``z``
    with pytest.raises(NonZeroDroppedAxisError):
        edge.pre_transforming_check(points)


def test_dynamic_drop_axis_edge_repr_contains_dropped_axes(xyz_cs, xy_cs):
    """``repr`` must list the dropped axes and must not contain a creates‑section."""
    edge = DynamicDropAxisEdge(input=xyz_cs, output=xy_cs)
    rep = repr(edge)
    assert "- drops axes:" in rep
    assert "z" in rep
    assert "+ creates axes:" not in rep
