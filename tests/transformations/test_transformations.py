from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import xarray.testing
from xarray import DataArray

from spatialdata import transform
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM, ValidAxis_t, get_channels
from spatialdata.transformations.ngff._utils import get_default_coordinate_system
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata.transformations.ngff.ngff_transformations import (
    NgffAffine,
    NgffBaseTransformation,
    NgffByDimension,
    NgffIdentity,
    NgffMapAxis,
    NgffScale,
    NgffSequence,
    NgffTranslation,
)
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    _decompose_affine_into_linear_and_translation,
    _decompose_transformation,
    _get_affine_for_element,
)


def test_identity():
    assert np.allclose(Identity().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3))
    assert np.allclose(Identity().inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3))
    assert np.allclose(
        Identity().to_affine_matrix(input_axes=("x", "y", "z"), output_axes=("y", "x", "z")),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        Identity().to_affine_matrix(input_axes=("x", "y"), output_axes=("c", "y", "x")),
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )
    with pytest.raises(ValueError):
        Identity().to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("x", "y"))


def test_map_axis():
    # map_axis0 behaves like an identity
    map_axis0 = MapAxis({"x": "x", "y": "y"})
    # second validation logic
    with pytest.raises(ValueError):
        map_axis0.to_affine_matrix(input_axes=("x", "y", "z"), output_axes=("x", "y"))

    # first validation logic
    with pytest.raises(ValueError):
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("z",), output_axes=("z",))
    assert np.allclose(
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("x",), output_axes=("x",)),
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        ),
    )
    # adding new axes with MapAxis (something that the Ngff MapAxis can't do)
    assert np.allclose(
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("x",), output_axes=("x", "z")),
        np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
            ]
        ),
    )

    map_axis0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.allclose(map_axis0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3))

    # map_axis1 is an example of invertible MapAxis; here it swaps x and y
    map_axis1 = MapAxis({"x": "y", "y": "x"})
    map_axis1_inverse = map_axis1.inverse()
    assert np.allclose(
        map_axis1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        map_axis1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        map_axis1_inverse.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
    assert np.allclose(
        map_axis1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y", "z")),
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        map_axis1.to_affine_matrix(input_axes=("x", "y", "z"), output_axes=("x", "y", "z")),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )
    # map_axis2 is an example of non-invertible MapAxis
    map_axis2 = MapAxis({"x": "z", "y": "z", "c": "x"})
    with pytest.raises(ValueError):
        map_axis2.inverse()
    with pytest.raises(ValueError):
        map_axis2.to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("x", "y", "c"))
    assert np.allclose(
        map_axis2.to_affine_matrix(input_axes=("x", "y", "z", "c"), output_axes=("x", "y", "z", "c")),
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        map_axis2.to_affine_matrix(input_axes=("x", "y", "z", "c"), output_axes=("x", "y", "c", "z")),
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )


def test_translation():
    with pytest.raises(TypeError):
        Translation(translation=(1, 2, 3))
    t0 = Translation([1, 2], axes=("x", "y"))
    t1 = Translation(np.array([2, 1]), axes=("y", "x"))
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        t1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("y", "x", "z", "c")),
        np.array([[0, 1, 0, 2], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
    assert np.allclose(
        t0.inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [1, 0, -1],
                [0, 1, -2],
                [0, 0, 1],
            ]
        ),
    )


def test_scale():
    with pytest.raises(TypeError):
        Scale(scale=(1, 2, 3))
    t0 = Scale([3, 2], axes=("x", "y"))
    t1 = Scale(np.array([2, 3]), axes=("y", "x"))
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        t1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("y", "x", "z", "c")),
        np.array([[0, 2, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    )
    assert np.allclose(
        t0.inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [1 / 3.0, 0, 0],
                [0, 1 / 2.0, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_affine():
    with pytest.raises(TypeError):
        Affine(affine=(1, 2, 3))
    with pytest.raises(ValueError):
        # wrong shape
        Affine([1, 2, 3, 4, 5, 6, 0, 0, 1], input_axes=("x", "y"), output_axes=("x", "y"))
    t0 = Affine(
        np.array(
            [
                [4, 5, 6],
                [1, 2, 3],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("y", "x"),
    )
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [0, 0, 1],
            ]
        ),
    )
    # checking that permuting the axes of an affine matrix and
    # inverting it are operations that commute (the order doesn't matter)
    inverse0 = t0.inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    t1 = Affine(
        t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    inverse1 = t1.inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.allclose(inverse0, inverse1)
    # check that the inversion works
    m0 = t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    m0_inverse = t0.inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.allclose(np.dot(m0, m0_inverse), np.eye(3))

    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("x", "y", "z", "c")),
        np.array(
            [
                [1, 2, 0, 3],
                [4, 5, 0, 6],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )

    # adding new axes
    assert np.allclose(
        Affine(
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                ]
            ),
            input_axes=("x"),
            output_axes=("x", "y"),
        ).to_affine_matrix(input_axes=("x"), output_axes=("x", "y")),
        np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        ),
    )
    # validation logic: adding an axes via the matrix but also having it as input
    with pytest.raises(ValueError):
        Affine(
            np.array(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                ]
            ),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ).to_affine_matrix(input_axes=("x"), output_axes=("x", "y"))

    # removing axes
    assert np.allclose(
        Affine(
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                ]
            ),
            input_axes=("x", "y"),
            output_axes=("x"),
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x")),
        np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_sequence():
    translation = Translation([1, 2], axes=("x", "y"))
    scale = Scale([3, 2, 1], axes=("y", "x", "z"))
    affine = Affine(
        np.array(
            [
                [4, 5, 6],
                [1, 2, 3],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("y", "x"),
    )
    sequence = Sequence([translation, scale, affine])
    manual = (
        # affine
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # scale
        @ np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        # translation
        @ np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    computed = sequence.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.allclose(manual, computed)

    larger_space0 = sequence.to_affine_matrix(input_axes=("x", "y", "c"), output_axes=("x", "y", "z", "c"))
    larger_space1 = Affine(manual, input_axes=("x", "y"), output_axes=("x", "y")).to_affine_matrix(
        input_axes=("x", "y", "c"), output_axes=("x", "y", "z", "c")
    )
    assert np.allclose(larger_space0, larger_space1)
    assert np.allclose(
        larger_space0,
        (
            # affine
            np.array(
                [
                    [1.0, 2.0, 0.0, 3.0],
                    [4.0, 5.0, 0.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # scale
            @ np.array(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # translation
            @ np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ),
    )
    # test sequence with MapAxis
    map_axis = MapAxis({"x": "y", "y": "x"})
    assert np.allclose(
        Sequence([map_axis, map_axis]).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3)
    )
    assert np.allclose(
        Sequence([map_axis, map_axis, map_axis]).to_affine_matrix(input_axes=("x", "y"), output_axes=("y", "x")),
        np.eye(3),
    )
    # test nested sequence
    affine_2d_to_3d = Affine(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y", "z"),
    )
    # the function _get_current_output_axes() doesn't get called for the last transformation in a sequence,
    # that's why we add Identity()
    sequence0 = Sequence([translation, map_axis, affine_2d_to_3d, Identity()])
    sequence1 = Sequence([Sequence([translation, map_axis]), affine_2d_to_3d, Identity()])
    sequence2 = Sequence([translation, Sequence([map_axis, affine_2d_to_3d, Identity()]), Identity()])
    matrix0 = sequence0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y", "z"))
    matrix1 = sequence1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y", "z"))
    matrix2 = sequence2.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y", "z"))
    assert np.allclose(matrix0, matrix1)
    assert np.allclose(matrix0, matrix2)
    assert np.allclose(
        matrix0,
        np.array(
            [
                [0, 1, 2],
                [1, 0, 1],
                [2, 0, 2],
                [0, 0, 1],
            ]
        ),
    )


def test_sequence_reorder_axes():
    # The order of the axes of the sequence is arbitrary, so it may not match the one that the user could ask when
    # calling to_affine_matrix(). This is why we need to reorder the axes in to_affine_matrix().
    # This test triggers this case
    affine = Affine(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y", "c"),
    )
    sequence = Sequence([affine])
    assert np.allclose(
        sequence.to_affine_matrix(input_axes=("x", "y"), output_axes=("c", "y", "x")),
        np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_sequence_reduce_dimensionality_of_last_transformation():
    # from a bug that I found, this was raising an expection in
    # Identity when calling to_affine_matrix() since the input_axes
    # contained 'c' but the output_axes didn't
    affine = Affine(
        [
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [0, 0, 0, 1],
        ],
        input_axes=("x", "y", "c"),
        output_axes=("x", "y", "c"),
    )
    Sequence([Identity(), affine, Identity()])
    matrix = affine.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.allclose(
        matrix,
        np.array(
            [
                [1, 2, 4],
                [4, 5, 7],
                [0, 0, 1],
            ]
        ),
    )


def test_transform_coordinates():
    map_axis = MapAxis({"x": "y", "y": "x"})
    translation = Translation([1, 2, 3], axes=("x", "y", "z"))
    scale = Scale([2, 3, 4], axes=("x", "y", "z"))
    affine = Affine(
        [
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y", "c"),
    )
    transformaions = [
        Identity(),
        map_axis,
        translation,
        scale,
        affine,
        Sequence([translation, scale, affine]),
    ]
    affine_matrix_manual = np.array(
        [
            [1, 2, 0, 3],
            [4, 5, 0, 6],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    coords = DataArray([[0, 0, 0], [1, 2, 3]], coords={"points": range(2), "dim": ["x", "y", "z"]})
    manual0 = (affine_matrix_manual @ np.vstack((coords.data.T, np.ones((1, 2)))))[:-2].T
    coords_manual = np.array([[2, 6, 12], [4, 12, 24]])
    manual1 = (affine_matrix_manual @ np.vstack((coords_manual.T, np.ones((1, 2)))))[:-2].T
    expected = [
        DataArray([[0, 0, 0], [1, 2, 3]], coords={"points": range(2), "dim": ["x", "y", "z"]}),
        DataArray([[0, 0, 0], [2, 1, 3]], coords={"points": range(2), "dim": ["x", "y", "z"]}),
        DataArray([[1, 2, 3], [2, 4, 6]], coords={"points": range(2), "dim": ["x", "y", "z"]}),
        DataArray([[0, 0, 0], [2, 6, 12]], coords={"points": range(2), "dim": ["x", "y", "z"]}),
        DataArray(manual0, coords={"points": range(2), "dim": ["x", "y", "z"]}),
        DataArray(manual1, coords={"points": range(2), "dim": ["x", "y", "z"]}),
    ]
    for t, e in zip(transformaions, expected, strict=True):
        transformed = t._transform_coordinates(coords)
        xarray.testing.assert_allclose(transformed, e)


def _make_cs(axes: tuple[ValidAxis_t, ...]) -> NgffCoordinateSystem:
    cs = get_default_coordinate_system(axes)
    for ax in axes:
        cs.get_axis(ax).unit = "micrometer"
    return cs


def _assert_sequence_transformations_equal_up_to_intermediate_coordinate_systems_names_and_units(
    t0: NgffSequence, t1: NgffSequence, outer_sequence: bool = True
):
    # in a sequence it is irrelevant which are the intermediate coordinate system names or unit. During conversion we
    # don't keep them (the code would be unnecessarily complex), therefore here we ignore them
    if outer_sequence:
        assert t0.input_coordinate_system.name == t1.input_coordinate_system.name
        assert t0.output_coordinate_system.name == t1.output_coordinate_system.name
    for sub0, sub1 in zip(t0.transformations, t1.transformations, strict=True):
        if isinstance(sub0, NgffSequence):
            assert isinstance(sub1, NgffSequence)
            _assert_sequence_transformations_equal_up_to_intermediate_coordinate_systems_names_and_units(
                sub0, sub1, outer_sequence=False
            )
        else:
            sub0_copy = deepcopy(sub0)
            sub1_copy = deepcopy(sub1)
            css = [
                sub0_copy.input_coordinate_system,
                sub0_copy.output_coordinate_system,
                sub1_copy.input_coordinate_system,
                sub1_copy.output_coordinate_system,
            ]
            for cs in css:
                cs.name = ""
                for ax in cs.axes_names:
                    cs.set_unit(ax, "")
            if sub0_copy != sub1_copy:
                raise AssertionError(f"{sub0_copy} != {sub1_copy}")


def _convert_and_compare(t0: NgffBaseTransformation, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem):
    t1 = BaseTransformation.from_ngff(t0)
    t2 = t1.to_ngff(
        input_axes=input_cs.axes_names,
        output_axes=output_cs.axes_names,
        unit="micrometer",
        output_coordinate_system_name=output_cs.name,
    )
    t3 = BaseTransformation.from_ngff(t2)
    if not isinstance(t0, NgffSequence):
        assert t0 == t2
    else:
        assert isinstance(t2, NgffSequence)
        _assert_sequence_transformations_equal_up_to_intermediate_coordinate_systems_names_and_units(t0, t2)
    assert t1 == t3


# conversion back and forth the NGFF transformations
def test_ngff_conversion_identity():
    # matching axes
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffIdentity(input_coordinate_system=input_cs, output_coordinate_system=output_cs)
    _convert_and_compare(t0, input_cs, output_cs)

    # TODO: add tests like this to all the transformations (https://github.com/scverse/spatialdata/issues/114)
    # # mismatching axes
    # input_cs, output_cs = _get_input_output_coordinate_systems(input_axes=("x", "y"), output_axes=("x", "y", "z"))
    # t0 = NgffIdentity(input_coordinate_system=input_cs, output_coordinate_system=output_cs)
    # _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_map_axis():
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffMapAxis(
        input_coordinate_system=input_cs, output_coordinate_system=output_cs, map_axis={"x": "y", "y": "x", "z": "z"}
    )
    _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_map_axis_creating_new_axes():
    # this is a case that is supported by the MapAxis class but not by the NgffMapAxis class, since in NGFF the
    # MapAxis can't create new axes

    # TODO: the conversion should raise an error in the NgffMapAxis class and should require adjusted input/output when
    # converting to fix it (see https://github.com/scverse/spatialdata/issues/114)
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffMapAxis(
        input_coordinate_system=input_cs,
        output_coordinate_system=output_cs,
        map_axis={"x": "y", "y": "x", "z": "z", "c": "x"},
    )
    _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_translation():
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffTranslation(
        input_coordinate_system=input_cs, output_coordinate_system=output_cs, translation=[1.0, 2.0, 3.0]
    )
    _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_scale():
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffScale(input_coordinate_system=input_cs, output_coordinate_system=output_cs, scale=[1.0, 2.0, 3.0])
    _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_affine():
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y"))
    t0 = NgffAffine(
        input_coordinate_system=input_cs,
        output_coordinate_system=output_cs,
        affine=[
            [1.0, 2.0, 3.0, 10.0],
            [4.0, 5.0, 6.0, 11.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    _convert_and_compare(t0, input_cs, output_cs)


def test_ngff_conversion_sequence():
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y"))
    affine0 = NgffAffine(
        input_coordinate_system=_make_cs(("x", "y", "z")),
        output_coordinate_system=_make_cs(("x", "y")),
        affine=[
            [1.0, 2.0, 3.0, 10.0],
            [4.0, 5.0, 6.0, 11.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    affine1 = NgffAffine(
        input_coordinate_system=_make_cs(("x", "y")),
        output_coordinate_system=_make_cs(("x", "y", "z")),
        affine=[
            [1.0, 2.0, 10.0],
            [4.0, 5.0, 11.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    )
    sequence = NgffSequence(
        input_coordinate_system=input_cs,
        output_coordinate_system=output_cs,
        transformations=[
            NgffIdentity(input_coordinate_system=input_cs, output_coordinate_system=input_cs),
            NgffSequence(
                input_coordinate_system=input_cs,
                output_coordinate_system=input_cs,
                transformations=[affine0, affine1],
            ),
        ],
    )
    _convert_and_compare(sequence, input_cs, output_cs)


def test_ngff_conversion_not_supported():
    # NgffByDimension is not supported in the new transformations classes
    # we may add converters in the future to create an Affine out of a NgffByDimension class
    input_cs = _make_cs(("x", "y", "z"))
    output_cs = _make_cs(("x", "y", "z"))
    t0 = NgffByDimension(
        input_coordinate_system=input_cs,
        output_coordinate_system=output_cs,
        transformations=[NgffIdentity(input_coordinate_system=input_cs, output_coordinate_system=output_cs)],
    )
    with pytest.raises(ValueError):
        _convert_and_compare(t0, input_cs, output_cs)


def test_get_affine_for_element(images):
    """This is testing the ability to predict the axis of a transformation given the transformation and the element
    it will be applied to. It is also testing the embedding of a 2d image with channel into the 3d space."""
    image = images.images["image2d"]
    t = Affine(
        np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y", "z"),
    )
    translation = Translation(np.array([1, 2, 3]), axes=("x", "y", "z"))
    sequence = Sequence([t, translation])
    real = _get_affine_for_element(image, sequence)
    assert real.input_axes == ("c", "y", "x")
    assert real.output_axes == ("c", "x", "y", "z")
    assert np.allclose(
        real.matrix,
        np.array(
            [
                # fmt: off
                #c  y  x       # noqa: E265
                [1, 0, 0, 0],  # c
                [0, 0, 1, 1],  # x
                [0, 1, 0, 2],  # y
                [0, 0, 0, 4],  # z
                [0, 0, 0, 1],
                # fmt: on
            ]
        ),
    )


def test_decompose_affine_into_linear_and_translation():
    matrix = np.array([[1, 2, 3, 10], [4, 5, 6, 11], [0, 0, 0, 1]])
    affine = Affine(matrix, input_axes=("x", "y", "z"), output_axes=("x", "y"))
    linear, translation = _decompose_affine_into_linear_and_translation(affine)
    assert np.allclose(linear.matrix, np.array([[1, 2, 3, 0], [4, 5, 6, 0], [0, 0, 0, 1]]))
    assert np.allclose(translation.translation, np.array([10, 11]))


@pytest.mark.parametrize(
    "matrix,input_axes,output_axes,valid",
    [
        # non-square matrix are not supported
        (
            np.array(
                [
                    [1, 2, 3, 10],
                    [4, 5, 6, 11],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "z"),
            ("x", "y"),
            False,
        ),
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [0, 0, 1],
                ]
            ),
            ("x", "y"),
            ("x", "y", "z"),
            False,
        ),
        # z axis should not be present
        (
            np.array(
                [
                    [1, 2, 3, 10],
                    [4, 5, 6, 11],
                    [7, 8, 9, 12],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "z"),
            ("x", "y", "z"),
            False,
        ),
        # c channel is modified
        (
            np.array(
                [
                    [1, 2, 0, 4],
                    [4, 5, 0, 7],
                    [8, 9, 1, 10],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "c"),
            ("x", "y", "c"),
            False,
        ),
        (
            np.array(
                [
                    [1, 2, 0, 4],
                    [4, 5, 0, 7],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "c"),
            ("x", "y", "c"),
            False,
        ),
        (
            np.array(
                [
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "c"),
            ("x", "y", "c"),
            False,
        ),
        # valid, no c channel
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [0, 0, 1],
                ]
            ),
            ("x", "y"),
            ("x", "y"),
            True,
        ),
        # valid, c channel
        (
            np.array(
                [
                    [1, 2, 0, 4],
                    [4, 5, 0, 7],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            ("x", "y", "c"),
            ("x", "y", "c"),
            True,
        ),
    ],
)
@pytest.mark.parametrize("simple_decomposition", [True, False])
def test_decompose_transformation(matrix, input_axes, output_axes, valid, simple_decomposition):
    affine = Affine(matrix, input_axes=input_axes, output_axes=output_axes)
    context = nullcontext() if valid else pytest.raises(ValueError)
    with context:
        _ = _decompose_transformation(affine, input_axes=input_axes, simple_decomposition=simple_decomposition)


def test_assign_xy_scale_to_cyx_image():
    scale = Scale(np.array([2, 3]), axes=("x", "y"))
    image = Image2DModel.parse(np.zeros((10, 10, 10)), dims=("c", "y", "x"))

    affine = _get_affine_for_element(image, scale)
    assert np.allclose(
        affine.matrix,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_assign_xyz_scale_to_cyx_image():
    scale = Scale(np.array([2, 3, 4]), axes=("x", "y", "z"))
    image = Image2DModel.parse(np.zeros((10, 10, 10)), dims=("c", "y", "x"))

    affine = _get_affine_for_element(image, scale)
    assert np.allclose(
        affine.matrix,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_assign_cyx_scale_to_xyz_points():
    scale = Scale(np.array([1, 3, 2]), axes=("c", "y", "x"))
    points = PointsModel.parse(np.zeros((10, 3)))

    affine = _get_affine_for_element(points, scale)
    assert np.allclose(
        affine.matrix,
        np.array(
            [
                [2, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
    )


def test_compose_in_xy_and_operate_in_cyx():
    k = 0.5
    scale = Scale([k, k], axes=("x", "y"))
    theta = np.pi / 6
    rotation = Affine(
        np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    sequence = Sequence([rotation, scale])
    affine = sequence.to_affine_matrix(input_axes=("c", "y", "x"), output_axes=("c", "y", "x"))
    assert np.allclose(
        affine,
        np.array(
            [
                [1, 0, 0, 0],
                [0, k * np.cos(theta), k * np.sin(theta), 0],
                [0, k * -np.sin(theta), k * np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        ),
    )


@pytest.mark.parametrize("image_name", ["blobs_image", "blobs_multiscale_image"])
def test_keep_numerical_coordinates_c(image_name):
    c_coords = range(3)
    sdata = blobs(n_channels=len(c_coords))
    t_blobs = transform(sdata.images[image_name], to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)
    assert np.array_equal(get_channels(t_blobs), c_coords)


@pytest.mark.parametrize("image_name", ["blobs_image", "blobs_multiscale_image"])
def test_keep_string_coordinates_c(image_name):
    c_coords = ["a", "b", "c"]
    # n_channels will be ignored, testing also that this works
    sdata = blobs(c_coords=c_coords, n_channels=4)
    t_blobs = transform(sdata.images[image_name], to_coordinate_system=DEFAULT_COORDINATE_SYSTEM)
    assert np.array_equal(get_channels(t_blobs), c_coords)
