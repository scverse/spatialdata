# pyright: strict

from __future__ import annotations

import numpy as np
import pytest

from spatialdata._core.transformation_manager.exceptions import (
    AxisRedefinitionError,
    EmptyTransformSequenceError,
    IncompatibleCoordSystemsError,
    MissingAxisError,
    NotUnimodularError,
    UnexpectedShapeError,
    UnmappedAxisError,
)
from spatialdata.transformations.graph.edge import (
    AffineEdge,
    ByDimensionEdge,
    IdentityEdge,
    MapAxisEdge,
    ProjectAxisEdge,
    RotationEdge,
    ScaleEdge,
    SequenceEdge,
    TranslationEdge,
)
from tests.transformations.edge.conftest import (
    x_cs,
    xy_cs,
    xyc_cs,
    xyz_cs,
    y_axis,
    y_cs,
    yx_cs,
    z_axis,
    zyx_cs,
)

POINTS_2D = np.array([[1.0, 2.0], [3.0, 4.0]])
POINTS_3D = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


class TestAffineEdge:
    def test_constructor_rejects_wrong_linear_shape(self):
        with pytest.raises(UnexpectedShapeError):
            AffineEdge(linear=np.eye(3), input=xy_cs, output=xy_cs)

    def test_constructor_rejects_wrong_translation_shape(self):
        with pytest.raises(UnexpectedShapeError):
            AffineEdge(linear=np.eye(2), translation=np.zeros(3), input=xy_cs, output=xy_cs)

    def test_from_affine_matrix(self):
        # fmt: off
        affine_matrix = np.array([
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 5.0],
            [0.0, 0.0, 1.0]
        ])
        linear = [
            [2.0, 0.0],
            [0.0, 3.0],
        ]
        translation = [
            1.0,
            5.0
        ]
        # fmt: on
        edge = AffineEdge.from_affine_matrix(name=None, affine_matrix=affine_matrix, input=xy_cs, output=xy_cs)
        np.testing.assert_equal(edge.linear, linear)
        np.testing.assert_equal(edge.translation, translation)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[3.0, 11.0], [7.0, 17.0]]))

    def test_mapping_classmethod_builds_permutation_matrix(self):
        edge = AffineEdge.mapping(input=xy_cs, output=yx_cs)
        np.testing.assert_allclose(edge.linear, np.array([[0.0, 1.0], [1.0, 0.0]]))
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 1.0], [4.0, 3.0]]))

    def test_transform_points_scale_and_translate(self):
        edge = AffineEdge(
            linear=np.array([[2.0, 0.0], [0.0, 3.0]]), translation=np.array([1.0, 1.0]), input=xy_cs, output=xy_cs
        )
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[3.0, 7.0], [7.0, 13.0]]))

    def test_transform_points_2d_to_3d(self):
        linear = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        edge = AffineEdge(linear=linear, input=xy_cs, output=xyz_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]]))

    def test_transform_points_rejects_wrong_shape(self):
        edge = AffineEdge(linear=np.eye(2), input=xy_cs, output=xy_cs)
        with pytest.raises(UnexpectedShapeError):
            # expecting (n, 2), gets (2,)
            edge.transform_points(np.asarray([1.0, 2.0]))
        with pytest.raises(UnexpectedShapeError):
            # expecting (n, 2), gets (n, 3)
            edge.transform_points(POINTS_3D)

    def test_inverse_roundtrips(self):
        edge = AffineEdge(
            linear=np.array([[2.0, 0.0], [0.0, 4.0]]), translation=np.array([1.0, -1.0]), input=xy_cs, output=xy_cs
        )
        inv = edge.inverse()
        assert inv is not None
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)

    def test_inverse_returns_none_for_singular_matrix(self):
        edge = AffineEdge(linear=np.zeros((2, 2)), input=xy_cs, output=xy_cs)
        assert edge.inverse() is None


class TestIdentityEdge:
    def test_constructor_rejects_mismatched_number_of_axes(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            IdentityEdge(None, input=xy_cs, output=xyz_cs)

    def test_transform_points_is_a_noop(self):
        edge = IdentityEdge(None, input=xy_cs, output=xy_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), POINTS_2D)

    def test_to_affine_matches_transform_points(self):
        edge = IdentityEdge(None, input=xy_cs, output=xy_cs)
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.linear, np.eye(2))
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), POINTS_2D)

    def test_inverse_is_still_the_identity(self):
        edge = IdentityEdge(None, input=xy_cs, output=xy_cs)
        inv = edge.inverse()
        np.testing.assert_allclose(inv.transform_points(POINTS_2D), POINTS_2D)


class TestMapAxisEdge:
    def test_constructor_rejects_different_sets_of_axes(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            MapAxisEdge(input=xy_cs, output=xyz_cs)

    def test_transform_points_swaps_axes(self):
        edge = MapAxisEdge(input=xy_cs, output=yx_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 1.0], [4.0, 3.0]]))

    def test_transform_points_permutation_of_three_axes(self):
        edge = MapAxisEdge(input=xyz_cs, output=zyx_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_3D), np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]))

    def test_to_affine_matches_transform_points(self):
        edge = MapAxisEdge(input=xy_cs, output=yx_cs)
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        edge = MapAxisEdge(input=xyz_cs, output=zyx_cs)
        inv = edge.inverse()
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_3D)), POINTS_3D)


class TestProjectAxisEdge:
    def test_constructor_rejects_dropped_axis_missing_from_input(self):
        with pytest.raises(MissingAxisError):
            ProjectAxisEdge(input=xy_cs, output=xy_cs, dropped_inputs={z_axis}, created_outputs=set())

    def test_constructor_rejects_created_axis_missing_from_output(self):
        with pytest.raises(MissingAxisError):
            ProjectAxisEdge(input=xy_cs, output=xy_cs, dropped_inputs=set(), created_outputs={z_axis})

    def test_general_coord_system_incompatibility(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            ProjectAxisEdge(input=xy_cs, output=xyz_cs, dropped_inputs=set(), created_outputs=set())
        with pytest.raises(IncompatibleCoordSystemsError):
            ProjectAxisEdge(input=xyz_cs, output=xy_cs, dropped_inputs=set(), created_outputs=set())

    def test_to_affine_for_identity_case(self):
        edge = ProjectAxisEdge(input=xy_cs, output=xy_cs, dropped_inputs=set(), created_outputs=set())
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.linear, np.eye(2))
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), POINTS_2D)

    def test_dropping_one_axis(self):
        edge = ProjectAxisEdge(input=xyz_cs, output=xy_cs, dropped_inputs={z_axis}, created_outputs=set())
        affine = edge.to_affine()
        assert affine is not None
        np.testing.assert_allclose(edge.transform_points(POINTS_3D), np.array([[1.0, 2.0], [4.0, 5.0]]))

    def test_creating_one_axis(self):
        edge = ProjectAxisEdge(input=xy_cs, output=xyz_cs, dropped_inputs=set(), created_outputs={z_axis})
        affine = edge.to_affine()
        assert affine is not None
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]))

    def test_inverse_none_when_axes_are_dropped_or_created(self):
        dropped = ProjectAxisEdge(input=xyz_cs, output=xy_cs, dropped_inputs={z_axis}, created_outputs=set())
        assert dropped.inverse() is None
        created = ProjectAxisEdge(input=xy_cs, output=xyz_cs, dropped_inputs=set(), created_outputs={z_axis})
        assert created.inverse() is None

    def test_inverse_roundtrip(self):
        edge = ProjectAxisEdge(input=xy_cs, output=yx_cs, dropped_inputs=set(), created_outputs=set())
        inv = edge.inverse()
        assert inv is not None
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)


class TestTranslationEdge:
    def test_constructor_rejects_mismatched_number_of_axes(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xyz_cs)

    def test_transform_points_adds_translation(self):
        edge = TranslationEdge(translation=np.array([10.0, 20.0]), input=xy_cs, output=xy_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[11.0, 22.0], [13.0, 24.0]]))

    def test_to_affine_matches_transform_points(self):
        edge = TranslationEdge(translation=np.array([10.0, 20.0]), input=xy_cs, output=xy_cs)
        affine = edge.to_affine()
        assert affine is not None
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        edge = TranslationEdge(translation=np.array([10.0, 20.0]), input=xy_cs, output=xy_cs)
        inv = edge.inverse()
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)


class TestScaleEdge:
    def test_constructor_rejects_wrong_scale_shape(self):
        with pytest.raises(UnexpectedShapeError):
            ScaleEdge(scale=np.array([1.0, 2.0, 3.0]), input=xy_cs, output=xy_cs)

    def test_constructor_rejects_mismatched_number_of_axes(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            ScaleEdge(scale=np.array([1.0, 2.0]), input=xy_cs, output=xyz_cs)

    def test_transform_points(self):
        edge = ScaleEdge(scale=np.array([2.0, 4.0]), input=xy_cs, output=xy_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 8.0], [6.0, 16.0]]))

    def test_to_affine_matches_transform_points(self):
        edge = ScaleEdge(scale=np.array([2.0, 4.0]), input=xy_cs, output=xy_cs)
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        edge = ScaleEdge(scale=np.array([2.0, 4.0]), input=xy_cs, output=xy_cs)
        inv = edge.inverse()
        assert inv is not None
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)

    def test_inverse_returns_none_when_scale_is_zero(self):
        edge = ScaleEdge(scale=np.array([0.0, 4.0]), input=xy_cs, output=xy_cs)
        assert edge.inverse() is None


class TestRotationEdge:
    def test_constructor_rejects_mismatched_number_of_axes(self):
        with pytest.raises(IncompatibleCoordSystemsError):
            RotationEdge(linear_matrix=np.eye(2), input=xy_cs, output=xyz_cs)

    def test_constructor_rejects_wrong_shape(self):
        with pytest.raises(UnexpectedShapeError):
            RotationEdge(linear_matrix=np.eye(3), input=xy_cs, output=xy_cs)

    def test_constructor_rejects_non_unimodular_matrix(self):
        with pytest.raises(NotUnimodularError):
            RotationEdge(linear_matrix=np.array([[1.0, 0.0], [0.0, -1.0]]), input=xy_cs, output=xy_cs)

    def test_transform_points_rotates_90_degrees(self):
        edge = RotationEdge(linear_matrix=np.array([[0.0, -1.0], [1.0, 0.0]]), input=xy_cs, output=xy_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[-2.0, 1.0], [-4.0, 3.0]]))

    def test_to_affine_matches_transform_points(self):
        edge = RotationEdge(linear_matrix=np.array([[0.0, -1.0], [1.0, 0.0]]), input=xy_cs, output=xy_cs)
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        edge = RotationEdge(linear_matrix=np.array([[0.0, -1.0], [1.0, 0.0]]), input=xy_cs, output=xy_cs)
        inv = edge.inverse()
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)


class TestSequenceEdge:
    def test_constructor_rejects_empty_sequence(self):
        with pytest.raises(EmptyTransformSequenceError):
            SequenceEdge(transformations=[])

    def test_constructor_rejects_incompatible_neighbors(self):
        first = TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xy_cs)
        second = TranslationEdge(translation=np.array([1.0, 2.0, 3.0]), input=xyz_cs, output=xyz_cs)
        with pytest.raises(IncompatibleCoordSystemsError):
            SequenceEdge(transformations=[first, second])

    def test_transform_points_composes_in_order(self):
        translate = TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xy_cs)
        scale = ScaleEdge(scale=np.array([3.0, 4.0]), input=xy_cs, output=xy_cs)
        edge = SequenceEdge(transformations=[translate, scale])
        expected = (POINTS_2D + np.array([1.0, 2.0])) * np.array([3.0, 4.0])
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), expected)

    def test_to_affine_matches_transform_points(self):
        translate = TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xy_cs)
        scale = ScaleEdge(scale=np.array([3.0, 4.0]), input=xy_cs, output=xy_cs)
        edge = SequenceEdge(transformations=[translate, scale])
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        translate = TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xy_cs)
        scale = ScaleEdge(scale=np.array([3.0, 4.0]), input=xy_cs, output=xy_cs)
        edge = SequenceEdge(transformations=[translate, scale])
        inv = edge.inverse()
        assert inv is not None
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)

    def test_inverse_returns_none_if_any_component_is_not_invertible(self):
        scale = ScaleEdge(scale=np.array([0.0, 4.0]), input=xy_cs, output=xy_cs)
        translate = TranslationEdge(translation=np.array([1.0, 2.0]), input=xy_cs, output=xy_cs)
        edge = SequenceEdge(transformations=[scale, translate])
        assert edge.inverse() is None


class TestByDimensionEdge:
    def test_constructor_rejects_input_axis_missing_from_overall_input(self):
        sub = IdentityEdge(None, input=xyc_cs, output=xyc_cs)
        with pytest.raises(MissingAxisError):
            ByDimensionEdge(transformations=[sub], input=xy_cs, output=xyc_cs)

    def test_constructor_rejects_output_axis_missing_from_overall_output(self):
        sub = IdentityEdge(None, input=xy_cs, output=xy_cs)
        with pytest.raises(MissingAxisError):
            ByDimensionEdge(transformations=[sub], input=xy_cs, output=x_cs)

    def test_constructor_rejects_output_axis_defined_more_than_once(self):
        first = IdentityEdge(None, input=x_cs, output=x_cs)
        second = IdentityEdge(None, input=x_cs, output=x_cs)
        with pytest.raises(AxisRedefinitionError):
            ByDimensionEdge(transformations=[first, second], input=x_cs, output=x_cs)

    def test_constructor_rejects_unmapped_output_axis(self):
        sub = IdentityEdge(None, input=x_cs, output=x_cs)
        try:
            ByDimensionEdge(transformations=[sub], input=xy_cs, output=xy_cs)
            raise AssertionError(f"Expected {UnmappedAxisError.__name__} to be raised")
        except UnmappedAxisError as e:
            assert e.axis == y_axis

    def test_transform_points_applies_each_transformation_per_axis(self):
        scale_x = ScaleEdge(scale=np.array([2.0]), input=x_cs, output=x_cs)
        translate_y = TranslationEdge(translation=np.array([5.0]), input=y_cs, output=y_cs)
        edge = ByDimensionEdge(transformations=[scale_x, translate_y], input=xy_cs, output=xy_cs)
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 7.0], [6.0, 9.0]]))

    def test_to_affine_matches_transform_points(self):
        scale_x = ScaleEdge(scale=np.array([2.0]), input=x_cs, output=x_cs)
        translate_y = TranslationEdge(translation=np.array([5.0]), input=y_cs, output=y_cs)
        edge = ByDimensionEdge(transformations=[scale_x, translate_y], input=xy_cs, output=xy_cs)
        affine = edge.to_affine()
        np.testing.assert_allclose(affine.transform_points(POINTS_2D), edge.transform_points(POINTS_2D))

    def test_inverse_roundtrips(self):
        scale_x = ScaleEdge(scale=np.array([2.0]), input=x_cs, output=x_cs)
        translate_y = TranslationEdge(translation=np.array([5.0]), input=y_cs, output=y_cs)
        edge = ByDimensionEdge(transformations=[scale_x, translate_y], input=xy_cs, output=xy_cs)
        inv = edge.inverse()
        assert inv is not None
        np.testing.assert_allclose(inv.transform_points(edge.transform_points(POINTS_2D)), POINTS_2D)
