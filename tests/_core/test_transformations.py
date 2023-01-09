import json

import numpy as np
import pytest

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transformations import (
    Affine,
    BaseTransformation,
    ByDimension,
    Identity,
    MapAxis,
    Rotation,
    Scale,
    Sequence,
    Translation,
)
from spatialdata._types import ArrayLike
from tests._core.conftest import (
    c_cs,
    cyx_cs,
    x_cs,
    xy_cs,
    xyc_cs,
    xyz_cs,
    y_cs,
    yx_cs,
    z_cs,
    zyx_cs,
)


def _test_transformation(
    transformation: BaseTransformation,
    original: np.ndarray,
    transformed: np.ndarray,
    input_cs: CoordinateSystem,
    output_cs: CoordinateSystem,
    wrong_output_cs: CoordinateSystem,
    test_affine: bool = True,
    test_affine_inverse: bool = True,
    test_inverse: bool = True,
):
    # missing input and output coordinate systems.
    # If the transformation is a Sequence, it can have the input_coordinate system specified (inherited from the
    # first component). In this case the test is skipped
    if transformation.input_coordinate_system is None:
        with pytest.raises(ValueError):
            assert np.allclose(transformation.transform_points(original), transformed)

    # missing output coordinate system
    # If the transformation is a Sequence, it can have the output_coordinate system specified (inherited from the
    # last component). In this case the test is skipped
    transformation.input_coordinate_system = input_cs
    if transformation.output_coordinate_system is None:
        with pytest.raises(ValueError):
            assert np.allclose(transformation.transform_points(original), transformed)

    # wrong output coordinate system
    transformation.output_coordinate_system = wrong_output_cs
    try:
        # if the output coordinate system still allows to compute the transformation, it will give points different
        # from the one we expect
        assert not np.allclose(transformation.transform_points(original), transformed)
    except ValueError:
        # covers the case in which the tranformation failed because of an incompatible output coordinate system
        pass

    # wrong points shapes
    transformation.output_coordinate_system = output_cs
    with pytest.raises(ValueError):
        assert transformation.transform_points(original.ravel())
    with pytest.raises(ValueError):
        assert transformation.transform_points(original.transpose())
    with pytest.raises(ValueError):
        assert transformation.transform_points(np.expand_dims(original, 0))

    # correct
    assert np.allclose(transformation.transform_points(original), transformed)

    if test_affine:
        affine = transformation.to_affine()
        assert np.allclose(affine.transform_points(original), transformed)
        if test_inverse:
            affine = transformation.to_affine()
            assert np.allclose(affine.inverse().transform_points(transformed), original)

    if test_inverse:
        inverse = transformation.inverse()
        assert np.allclose(inverse.transform_points(transformed), original)
    else:
        try:
            transformation.inverse()
        except ValueError:
            pass
        except np.linalg.LinAlgError:
            pass

    # test to_dict roundtrip
    assert transformation.to_dict() == BaseTransformation.from_dict(transformation.to_dict()).to_dict()

    # test to_json roundtrip
    assert json.dumps(transformation.to_dict()) == json.dumps(
        BaseTransformation.from_dict(json.loads(json.dumps(transformation.to_dict()))).to_dict()
    )

    # test repr
    as_str = repr(transformation)
    assert repr(transformation.input_coordinate_system) in as_str
    assert repr(transformation.output_coordinate_system) in as_str
    assert type(transformation).__name__ in as_str


def test_identity():
    _test_transformation(
        transformation=Identity(),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[1, 2, 3], [1, 1, 1]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_map_axis():
    _test_transformation(
        transformation=MapAxis({"x": "x", "y": "y", "z": "z"}),
        original=np.array([[1, 2, 3], [2, 3, 4]]),
        transformed=np.array([[3, 2, 1], [4, 3, 2]]),
        input_cs=xyz_cs,
        output_cs=zyx_cs,
        wrong_output_cs=xyz_cs,
    )
    _test_transformation(
        transformation=MapAxis({"x": "x", "y": "y", "z": "y"}),
        original=np.array([[1, 2]]),
        transformed=np.array([[2, 2, 1]]),
        input_cs=xy_cs,
        output_cs=zyx_cs,
        wrong_output_cs=xyz_cs,
        test_inverse=False,
    )
    _test_transformation(
        transformation=MapAxis({"x": "y", "y": "x", "z": "z"}),
        original=np.array([[1, 2, 3]]),
        transformed=np.array([[2, 1, 3]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_translations():
    _test_transformation(
        transformation=Translation(np.array([1, 2, 3])),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[2, 4, 6], [2, 3, 4]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_scale():
    _test_transformation(
        transformation=Scale(np.array([1, 2, 3])),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[1, 4, 9], [1, 2, 3]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_affine_2d():
    _test_transformation(
        transformation=Affine(np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[8, 20], [14, 38], [20, 56]]),
        input_cs=xy_cs,
        output_cs=xy_cs,
        # this would give the same result as above, because affine doesn't check the axes
        # wrong_output_cs=yx_cs,
        # instead, this is wrong, since the affine matrix is not compatible with the output coordinate system
        wrong_output_cs=zyx_cs,
    )


def test_affine_2d_to_3d():
    # embedding a space into a larger one
    _test_transformation(
        transformation=Affine(np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [0, 0, 1]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[8, 8, 20], [14, 14, 38], [20, 20, 56]]),
        input_cs=yx_cs,
        output_cs=cyx_cs,
        wrong_output_cs=yx_cs,
        test_inverse=False,
    )


def test_affine_3d_to_2d():
    # projecting a space into a smaller one
    _test_transformation(
        transformation=Affine(np.array([[4, 5, 6], [0, 0, 1]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[20], [38], [56]]),
        input_cs=xy_cs,
        output_cs=y_cs,
        wrong_output_cs=xy_cs,
        test_inverse=False,
    )


def test_rotations():
    _test_transformation(
        transformation=Rotation(np.array([[0, -1], [1, 0]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[-2, 1], [-4, 3], [-6, 5]]),
        input_cs=xy_cs,
        output_cs=xy_cs,
        # this would give the same result as above, because affine doesn't check the axes
        # wrong_output_cs=yx_cs,
        # instead, this is wrong, since the affine matrix is not compatible with the output coordinate system
        wrong_output_cs=zyx_cs,
    )


def _test_sequence_helper() -> tuple[ArrayLike, Affine, ArrayLike]:
    original = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    affine = Affine(np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]))
    transformed = np.matmul(
        np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]),
        np.vstack([np.transpose((original + np.array([1, 2])) * np.array([3, 4])), [1] * len(original)]),
    )[:-1, :].T
    return original, affine, transformed


def test_sequence_ambiguous_coordinate_systems():
    original, affine, transformed = _test_sequence_helper()
    # ambiguous 2d case (no input/output coordinate system specified for the affine transformation composing the
    # sequence)
    with pytest.raises(ValueError):
        _test_transformation(
            transformation=Sequence(
                [
                    Translation(np.array([1, 2])),
                    Scale(np.array([3, 4])),
                    affine,
                ]
            ),
            original=original,
            transformed=transformed,
            input_cs=xy_cs,
            output_cs=xy_cs,
            wrong_output_cs=yx_cs,
        )


def test_sequence_2d():
    original, affine, transformed = _test_sequence_helper()
    # 2d case
    affine.input_coordinate_system = xy_cs
    affine.output_coordinate_system = xy_cs
    _test_transformation(
        transformation=Sequence(
            [
                Translation(np.array([1, 2])),
                Scale(np.array([3, 4])),
                affine,
            ]
        ),
        original=original,
        transformed=transformed,
        input_cs=xy_cs,
        output_cs=xy_cs,
        wrong_output_cs=yx_cs,
    )


def test_sequence_3d():
    original, affine, transformed = _test_sequence_helper()
    # 3d case
    _test_transformation(
        transformation=Sequence(
            [
                Translation(np.array([1, 2, 3])),
                Scale(np.array([4, 5, 6])),
                Translation(np.array([7, 8, 9])),
            ]
        ),
        original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        transformed=np.array([[15, 28, 45], [27, 43, 63], [39, 58, 81], [51, 73, 99]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_sequence_2d_to_2d_with_c():
    original, affine, transformed = _test_sequence_helper()
    affine.input_coordinate_system = xy_cs
    affine.output_coordinate_system = xy_cs
    # 2d case, extending a xy->xy transformation to a cyx->cyx transformation using additional affine transformations
    cyx_to_xy = Affine(
        np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        input_coordinate_system=cyx_cs,
        output_coordinate_system=xy_cs,
    )
    xy_to_cyx = Affine(
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        input_coordinate_system=xy_cs,
        output_coordinate_system=cyx_cs,
    )

    def _manual_xy_to_cyx(x: np.ndarray) -> np.ndarray:
        return np.hstack((np.zeros(len(x)).reshape((len(x), 1)), np.fliplr(x)))

    _test_transformation(
        transformation=Sequence(
            [
                cyx_to_xy,
                # some alternative ways to go back and forth between xy and cyx
                # xy -> cyx
                ByDimension(
                    transformations=[
                        MapAxis({"x": "x", "y": "y"}, input_coordinate_system=xy_cs, output_coordinate_system=yx_cs),
                        Affine(
                            np.array([[0, 0], [0, 1]]),
                            input_coordinate_system=x_cs,
                            output_coordinate_system=c_cs,
                        ),
                    ],
                    input_coordinate_system=xy_cs,
                    output_coordinate_system=cyx_cs,
                ),
                # cyx -> xy
                MapAxis({"x": "x", "y": "y"}, input_coordinate_system=cyx_cs, output_coordinate_system=xy_cs),
                Translation(np.array([1, 2])),
                Scale(np.array([3, 4])),
                affine,
                xy_to_cyx,
            ]
        ),
        original=_manual_xy_to_cyx(original),
        transformed=_manual_xy_to_cyx(transformed),
        input_cs=cyx_cs,
        output_cs=cyx_cs,
        wrong_output_cs=xyc_cs,
        test_inverse=False,
    )


def test_sequence_nested():
    original, affine, transformed = _test_sequence_helper()
    # test sequence inside sequence, with full inference of the intermediate coordinate systems
    # two nested should be enought, let's test even three!
    _test_transformation(
        transformation=Sequence(
            [
                Scale(np.array([2, 3])),
                Sequence(
                    [
                        Scale(np.array([4, 5])),
                        Sequence(
                            [Scale(np.array([6, 7]))],
                        ),
                    ],
                ),
            ]
        ),
        original=original,
        transformed=original * np.array([2 * 4 * 6, 3 * 5 * 7]),
        input_cs=yx_cs,
        output_cs=yx_cs,
        wrong_output_cs=xy_cs,
        test_inverse=True,
    )


def test_sequence_mismatching_cs_inference():
    original, affine, transformed = _test_sequence_helper()
    _test_transformation(
        transformation=Sequence(
            [
                Scale(np.array([2, 3]), input_coordinate_system=yx_cs, output_coordinate_system=yx_cs),
                Scale(np.array([4, 5]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
            ]
        ),
        original=original,
        transformed=original * np.array([2 * 5, 3 * 4]),
        input_cs=yx_cs,
        output_cs=yx_cs,
        wrong_output_cs=xy_cs,
        test_inverse=True,
    )


@pytest.mark.skip()
def test_displacements():
    raise NotImplementedError()


@pytest.mark.skip()
def test_coordinates():
    raise NotImplementedError()


@pytest.mark.skip()
def test_vector_field():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_inverse_of():
    raise NotImplementedError()


@pytest.mark.skip()
def test_bijection():
    raise NotImplementedError()


#
def test_by_dimension():
    _test_transformation(
        transformation=ByDimension(
            [
                Translation(np.array([1, 2]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
                Scale(np.array([3]), input_coordinate_system=z_cs, output_coordinate_system=z_cs),
            ]
        ),
        original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        transformed=np.array([[2, 4, 9], [5, 7, 18], [8, 10, 27], [11, 13, 36]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_get_affine_form_input_output_coordinate_systems():
    from spatialdata._core.core_utils import C, X, Y, Z, get_default_coordinate_system

    data = {
        X: 1.0,
        Y: 2.0,
        Z: 3.0,
        C: 4.0,
    }
    input_css = [
        get_default_coordinate_system(t) for t in [(X, Y), (Y, X), (C, Y, X), (X, Y, Z), (Z, Y, X), (C, Z, Y, X)]
    ]
    output_css = input_css.copy()
    for input_cs in input_css:
        for output_cs in output_css:
            a = Affine.from_input_output_coordinate_systems(input_cs, output_cs)

            input_axes = input_cs.axes_names
            output_axes = output_cs.axes_names
            input_data = np.atleast_2d([data[a] for a in input_axes])
            output_data = np.atleast_2d([data[a] if a in input_axes else 0.0 for a in output_axes])

            transformed_data = a.transform_points(input_data)
            assert np.allclose(transformed_data, output_data)
            print(a.affine)
