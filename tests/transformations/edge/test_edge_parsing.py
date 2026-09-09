# pyright: strict

from __future__ import annotations

import numpy as np
import ome_zarr_models.v06.coordinate_transforms as ozm06trans

from spatialdata.transformations.graph.edge import (
    AffineEdge,
    ByDimensionEdge,
    CsGen,
    MapAxisEdge,
    RotationEdge,
    ScaleEdge,
    SequenceEdge,
    TranslationEdge,
    parse_affine,
    parse_by_dimension,
    parse_identity,
    parse_map_axis,
    parse_project_axis,
    parse_rotation,
    parse_scale,
    parse_sequence,
    parse_translation,
)
from tests.transformations.edge.conftest import (
    xy_cs,
    xyz_cs,
    yx_cs,
    z_axis,
)

# The ome-zarr-models transform models used below all leave `input`/`output` at their
# default of None: those fields identify coordinate systems by name/path in the NGFF
# metadata and aren't consumed by the parse_* functions, which instead take the already
# resolved `CoordSystem` (or a `CsGen`) as separate arguments. Only the transform-specific
# fields (e.g. `translation`, `scale`, `name`) are exercised here.

POINTS_2D = np.array([[1.0, 2.0], [3.0, 4.0]])


class TestParseIdentity:
    def test_parses_name(self):
        model = ozm06trans.Identity(name="my-identity")
        edge = parse_identity(model, input=xy_cs, out=xy_cs)
        assert edge.name == "my-identity"
        assert edge.input is xy_cs
        assert edge.output is xy_cs

    def test_csgen_output_mirrors_input(self):
        model = ozm06trans.Identity(name="my-identity")
        edge = parse_identity(model, input=xy_cs, out=CsGen(base_name="gen"))
        assert edge.output.axes_names == xy_cs.axes_names
        assert edge.output.virtual is True


class TestParseTranslation:
    def test_parses_translation_vector(self):
        model = ozm06trans.Translation(name="my-translation", translation=(1.0, 2.0))
        edge = parse_translation(model, input=xy_cs, out=xy_cs)
        np.testing.assert_allclose(edge.translation, [1.0, 2.0])
        assert edge.name == model.name
        assert edge.input is xy_cs
        assert edge.output is xy_cs

    def test_csgen_output_mirrors_input(self):
        model = ozm06trans.Translation(translation=(1.0, 2.0))
        edge = parse_translation(model, input=xy_cs, out=CsGen(base_name="gen"))
        assert edge.output.axes_names == xy_cs.axes_names
        assert edge.output.virtual is True


class TestParseProjectAxis:
    def test_parses_dropped_inputs(self):
        model = ozm06trans.ProjectAxis(name="drop-z", droppedInputs=(2,))
        edge = parse_project_axis(model, input=xyz_cs, output=xy_cs)
        assert edge.dropped_inputs == {z_axis}
        assert edge.created_outputs == set()
        assert edge.name == "drop-z"

    def test_parses_created_outputs(self):
        model = ozm06trans.ProjectAxis(name="create-z", createdOutputs=(2,))
        edge = parse_project_axis(model, input=xy_cs, output=xyz_cs)
        assert edge.created_outputs == {z_axis}
        assert edge.dropped_inputs == set()

    def test_csgen_output_computes_dimensionality(self):
        model = ozm06trans.ProjectAxis(droppedInputs=(2,))
        edge = parse_project_axis(model, input=xyz_cs, output=CsGen(base_name="gen"))
        assert edge.output.num_axes == 2
        assert edge.dropped_inputs == {z_axis}


class TestParseScale:
    def test_parses_scale_vector_and_name(self):
        model = ozm06trans.Scale(name="my-scale", scale=(2.0, 3.0))
        edge = parse_scale(model, input=xy_cs, out=xy_cs)
        assert isinstance(edge, ScaleEdge)
        np.testing.assert_allclose(edge.scale, [2.0, 3.0])
        assert edge.name == "my-scale"

    def test_csgen_output_mirrors_input(self):
        model = ozm06trans.Scale(scale=(2.0, 3.0))
        edge = parse_scale(model, input=xy_cs, out=CsGen(base_name="gen"))
        assert edge.output.axes_names == xy_cs.axes_names
        assert edge.output.virtual is True


class TestParseMapAxis:
    def test_parses_with_explicit_output(self):
        model = ozm06trans.MapAxis(name="my-map-axis", mapAxis=(1, 0))
        edge = parse_map_axis(model, input=xy_cs, out=yx_cs)
        assert isinstance(edge, MapAxisEdge)
        assert edge.name == "my-map-axis"
        assert edge.output is yx_cs
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 1.0], [4.0, 3.0]]))

    def test_csgen_output_is_permuted_input_axes(self):
        model = ozm06trans.MapAxis(mapAxis=(1, 0))
        edge = parse_map_axis(model, input=xy_cs, out=CsGen(base_name="gen"))
        assert edge.output.axes_names == yx_cs.axes_names
        assert edge.output.virtual is True


class TestParseAffine:
    def test_parses_linear_translation_and_name(self):
        model = ozm06trans.Affine(name="my-affine", affine=((2.0, 0.0, 1.0), (0.0, 3.0, 5.0)))
        edge = parse_affine(model, input=xy_cs, output=xy_cs)
        assert isinstance(edge, AffineEdge)
        np.testing.assert_allclose(edge.linear, [[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_allclose(edge.translation, [1.0, 5.0])
        assert edge.name == "my-affine"

    def test_csgen_output_derived_from_matrix_row_count(self):
        model = ozm06trans.Affine(affine=((2.0, 0.0, 1.0), (0.0, 3.0, 5.0)))
        edge = parse_affine(model, input=xy_cs, output=CsGen(base_name="gen"))
        assert edge.output.num_axes == 2
        assert edge.output.virtual is True


class TestParseRotation:
    def test_parses_rotation_matrix_and_name(self):
        model = ozm06trans.Rotation(name="my-rotation", rotation=((0.0, -1.0), (1.0, 0.0)))
        edge = parse_rotation(model, input=xy_cs, out=xy_cs)
        assert isinstance(edge, RotationEdge)
        np.testing.assert_allclose(edge.rotation, [[0.0, -1.0], [1.0, 0.0]])
        assert edge.name == "my-rotation"

    def test_csgen_output_derived_from_matrix_row_count(self):
        model = ozm06trans.Rotation(rotation=((0.0, -1.0), (1.0, 0.0)))
        edge = parse_rotation(model, input=xy_cs, out=CsGen(base_name="gen"))
        assert edge.output.num_axes == 2
        assert edge.output.virtual is True


class TestParseSequence:
    def test_parses_name_and_composes_inner_transformations_in_order(self):
        inner_translation = ozm06trans.Translation(translation=(1.0, 2.0))
        inner_scale = ozm06trans.Scale(scale=(2.0, 3.0))
        model = ozm06trans.Sequence(name="my-sequence", transformations=(inner_translation, inner_scale))
        edge = parse_sequence(model, input=xy_cs, output=xy_cs)
        assert isinstance(edge, SequenceEdge)
        assert edge.name == "my-sequence"
        assert edge.input is xy_cs
        assert edge.output is xy_cs
        assert [type(t) for t in edge.transformations] == [TranslationEdge, ScaleEdge]
        expected = (POINTS_2D + np.array([1.0, 2.0])) * np.array([2.0, 3.0])
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), expected)

    def test_single_inner_transformation_uses_given_output_directly(self):
        inner_translation = ozm06trans.Translation(translation=(1.0, 2.0))
        model = ozm06trans.Sequence(transformations=(inner_translation,))
        edge = parse_sequence(model, input=xy_cs, output=xy_cs)
        assert len(edge.transformations) == 1
        assert edge.transformations[0].output is xy_cs

    def test_multiple_inner_transformations_use_virtual_intermediate_coord_systems(self):
        inner_translation = ozm06trans.Translation(translation=(1.0, 2.0))
        inner_scale = ozm06trans.Scale(scale=(2.0, 3.0))
        model = ozm06trans.Sequence(transformations=(inner_translation, inner_scale))
        edge = parse_sequence(model, input=xy_cs, output=xy_cs)
        intermediate = edge.transformations[0].output
        assert intermediate is not xy_cs
        assert intermediate.virtual is True
        assert intermediate.axes_names == xy_cs.axes_names
        assert edge.transformations[1].output is xy_cs

    def test_csgen_output_mirrors_input(self):
        inner_translation = ozm06trans.Translation(translation=(1.0, 2.0))
        inner_scale = ozm06trans.Scale(scale=(2.0, 3.0))
        model = ozm06trans.Sequence(transformations=(inner_translation, inner_scale))
        edge = parse_sequence(model, input=xy_cs, output=CsGen(base_name="gen"))
        assert edge.output.axes_names == xy_cs.axes_names
        assert edge.output.virtual is True


class TestParseByDimension:
    def test_parses_name_and_splits_axes_between_inner_transformations(self):
        scale_x = ozm06trans.Scale(scale=(2.0,))
        translate_y = ozm06trans.Translation(translation=(5.0,))
        model = ozm06trans.ByDimension(
            name="my-by-dim",
            transformations=(
                ozm06trans.ByDimensionTransform(transformation=scale_x, input_axes=(0,), output_axes=(0,)),
                ozm06trans.ByDimensionTransform(transformation=translate_y, input_axes=(1,), output_axes=(1,)),
            ),
        )
        edge = parse_by_dimension(model, input=xy_cs, output=xy_cs)
        assert isinstance(edge, ByDimensionEdge)
        assert edge.name == "my-by-dim"
        assert [type(t) for t in edge.transformations] == [ScaleEdge, TranslationEdge]
        np.testing.assert_allclose(edge.transform_points(POINTS_2D), np.array([[2.0, 7.0], [6.0, 9.0]]))

    def test_csgen_output_derived_from_max_output_axis_index(self):
        scale_x = ozm06trans.Scale(scale=(2.0,))
        translate_y = ozm06trans.Translation(translation=(5.0,))
        model = ozm06trans.ByDimension(
            transformations=(
                ozm06trans.ByDimensionTransform(transformation=scale_x, input_axes=(0,), output_axes=(0,)),
                ozm06trans.ByDimensionTransform(transformation=translate_y, input_axes=(1,), output_axes=(1,)),
            ),
        )
        edge = parse_by_dimension(model, input=xy_cs, output=CsGen(base_name="gen"))
        assert edge.output.num_axes == 2
        assert edge.output.virtual is True
