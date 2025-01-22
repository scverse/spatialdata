import contextlib
import copy
import json

import numpy as np
import pytest

from spatialdata._types import ArrayLike
from spatialdata.models import C, X, Y, Z
from spatialdata.transformations.ngff._utils import get_default_coordinate_system
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata.transformations.ngff.ngff_transformations import (
    NgffAffine,
    NgffBaseTransformation,
    NgffByDimension,
    NgffIdentity,
    NgffMapAxis,
    NgffRotation,
    NgffScale,
    NgffSequence,
    NgffTranslation,
)
from tests.transformations.ngff.conftest import (
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
    transformation: NgffBaseTransformation,
    original: np.ndarray,
    transformed: np.ndarray,
    input_cs: NgffCoordinateSystem,
    output_cs: NgffCoordinateSystem,
    wrong_output_cs: NgffCoordinateSystem,
    test_affine: bool = True,
    test_affine_inverse: bool = True,
    test_inverse: bool = True,
):
    # missing input and output coordinate systems.
    # If the transformation is a NgffSequence, it can have the input_coordinate system specified (inherited from the
    # first component). In this case the test is skipped
    if transformation.input_coordinate_system is None:
        with pytest.raises(ValueError):
            # the function check_and_infer_coordinate_systems() can modfiy the attribute "self.transformations" and
            # corrupt the object, so we need to create a copy every time we want to modify the object in the current
            # function
            # it's not a problem when the object is not corrupt, but here we are modifying it on purpose to be corrupt
            assert np.allclose(copy.deepcopy(transformation).transform_points(original), transformed)

    # missing output coordinate system
    # If the transformation is a NgffSequence, it can have the output_coordinate system specified (inherited from the
    # last component). In this case the test is skipped
    transformation.input_coordinate_system = input_cs
    if transformation.output_coordinate_system is None:
        with pytest.raises(ValueError):
            assert np.allclose(copy.deepcopy(transformation).transform_points(original), transformed)

    # wrong output coordinate system
    transformation.output_coordinate_system = wrong_output_cs
    with contextlib.suppress(ValueError):
        # covers the case in which the transformation failed because of an incompatible output coordinate system
        # if the output coordinate system still allows to compute the transformation, it will give points different
        # from the one we expect
        assert not np.allclose(copy.deepcopy(transformation).transform_points(original), transformed)

    # wrong points shapes
    transformation.output_coordinate_system = output_cs
    with pytest.raises(ValueError):
        assert copy.deepcopy(transformation).transform_points(original.ravel())
    with pytest.raises(ValueError):
        assert copy.deepcopy(transformation).transform_points(original.transpose())
    with pytest.raises(ValueError):
        assert copy.deepcopy(transformation).transform_points(np.expand_dims(original, 0))

    # correct
    assert np.allclose(copy.deepcopy(transformation).transform_points(original), transformed)

    if test_affine:
        affine = copy.deepcopy(transformation).to_affine()
        assert np.allclose(affine.transform_points(original), transformed)
        if test_inverse:
            affine = copy.deepcopy(transformation).to_affine()
            assert np.allclose(affine.inverse().transform_points(transformed), original)

    if test_inverse:
        try:
            inverse = copy.deepcopy(transformation).inverse()
            assert np.allclose(inverse.transform_points(transformed), original)
        except ValueError:
            pass
    else:
        try:
            copy.deepcopy(transformation).inverse()
        except ValueError:
            pass
        except np.linalg.LinAlgError:
            pass

    # test to_dict roundtrip
    assert transformation.to_dict() == NgffBaseTransformation.from_dict(transformation.to_dict()).to_dict()

    # test to_json roundtrip
    assert json.dumps(transformation.to_dict()) == json.dumps(
        NgffBaseTransformation.from_dict(json.loads(json.dumps(transformation.to_dict()))).to_dict()
    )

    # test repr
    as_str = repr(transformation)
    assert type(transformation).__name__ in as_str


def test_identity():
    _test_transformation(
        transformation=NgffIdentity(),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[1, 2, 3], [1, 1, 1]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_map_axis():
    _test_transformation(
        transformation=NgffMapAxis({"x": "x", "y": "y", "z": "z"}),
        original=np.array([[1, 2, 3], [2, 3, 4]]),
        transformed=np.array([[3, 2, 1], [4, 3, 2]]),
        input_cs=xyz_cs,
        output_cs=zyx_cs,
        wrong_output_cs=xyz_cs,
    )
    _test_transformation(
        transformation=NgffMapAxis({"x": "x", "y": "y", "z": "y"}),
        original=np.array([[1, 2]]),
        transformed=np.array([[2, 2, 1]]),
        input_cs=xy_cs,
        output_cs=zyx_cs,
        wrong_output_cs=xyz_cs,
        test_inverse=False,
    )
    _test_transformation(
        transformation=NgffMapAxis({"x": "y", "y": "x", "z": "z"}),
        original=np.array([[1, 2, 3]]),
        transformed=np.array([[2, 1, 3]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_translations():
    _test_transformation(
        transformation=NgffTranslation(np.array([1, 2, 3])),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[2, 4, 6], [2, 3, 4]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_scale():
    _test_transformation(
        transformation=NgffScale(np.array([1, 2, 3])),
        original=np.array([[1, 2, 3], [1, 1, 1]]),
        transformed=np.array([[1, 4, 9], [1, 2, 3]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_affine_2d():
    _test_transformation(
        transformation=NgffAffine(np.array([[1, 2, 3], [4, 5, 6], [0, 0, 1]])),
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
        transformation=NgffAffine(np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [0, 0, 1]])),
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
        transformation=NgffAffine(np.array([[4, 5, 6], [0, 0, 1]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[20], [38], [56]]),
        input_cs=xy_cs,
        output_cs=y_cs,
        wrong_output_cs=xy_cs,
        test_inverse=False,
    )


def test_rotations():
    _test_transformation(
        transformation=NgffRotation(np.array([[0, -1], [1, 0]])),
        original=np.array([[1, 2], [3, 4], [5, 6]]),
        transformed=np.array([[-2, 1], [-4, 3], [-6, 5]]),
        input_cs=xy_cs,
        output_cs=xy_cs,
        # this would give the same result as above, because affine doesn't check the axes
        # wrong_output_cs=yx_cs,
        # instead, this is wrong, since the affine matrix is not compatible with the output coordinate system
        wrong_output_cs=zyx_cs,
    )


def _test_sequence_helper() -> tuple[ArrayLike, NgffAffine, ArrayLike]:
    original = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    affine = NgffAffine(np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]))
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
            transformation=NgffSequence(
                [
                    NgffTranslation(np.array([1, 2])),
                    NgffScale(np.array([3, 4])),
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
        transformation=NgffSequence(
            [
                NgffTranslation(np.array([1, 2])),
                NgffScale(np.array([3, 4])),
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
        transformation=NgffSequence(
            [
                NgffTranslation(np.array([1, 2, 3])),
                NgffScale(np.array([4, 5, 6])),
                NgffTranslation(np.array([7, 8, 9])),
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
    cyx_to_xy = NgffAffine(
        np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        input_coordinate_system=cyx_cs,
        output_coordinate_system=xy_cs,
    )
    xy_to_cyx = NgffAffine(
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        input_coordinate_system=xy_cs,
        output_coordinate_system=cyx_cs,
    )

    def _manual_xy_to_cyx(x: np.ndarray) -> np.ndarray:
        return np.hstack((np.zeros(len(x)).reshape((len(x), 1)), np.fliplr(x)))

    _test_transformation(
        transformation=NgffSequence(
            [
                cyx_to_xy,
                # some alternative ways to go back and forth between xy and cyx
                # xy -> cyx
                NgffByDimension(
                    transformations=[
                        NgffMapAxis(
                            {"x": "x", "y": "y"}, input_coordinate_system=xy_cs, output_coordinate_system=yx_cs
                        ),
                        NgffAffine(
                            np.array([[0, 0], [0, 1]]),
                            input_coordinate_system=x_cs,
                            output_coordinate_system=c_cs,
                        ),
                    ],
                    input_coordinate_system=xy_cs,
                    output_coordinate_system=cyx_cs,
                ),
                # cyx -> xy
                NgffMapAxis({"x": "x", "y": "y"}, input_coordinate_system=cyx_cs, output_coordinate_system=xy_cs),
                NgffTranslation(np.array([1, 2])),
                NgffScale(np.array([3, 4])),
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
        transformation=NgffSequence(
            [
                NgffScale(np.array([2, 3])),
                NgffSequence(
                    [
                        NgffScale(np.array([4, 5])),
                        NgffSequence(
                            [NgffScale(np.array([6, 7]))],
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


# @pytest.mark.skip()
# def test_displacements():
#     raise NotImplementedError()
#
#
# @pytest.mark.skip()
# def test_coordinates():
#     raise NotImplementedError()
#
# @pytest.mark.skip()
# def test_inverse_of_inverse_of():
#     raise NotImplementedError()
#
#
# @pytest.mark.skip()
# def test_bijection():
#     raise NotImplementedError()


#
def test_by_dimension():
    _test_transformation(
        transformation=NgffByDimension(
            [
                NgffTranslation(np.array([1, 2]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
                NgffScale(np.array([3]), input_coordinate_system=z_cs, output_coordinate_system=z_cs),
            ]
        ),
        original=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        transformed=np.array([[2, 4, 9], [5, 7, 18], [8, 10, 27], [11, 13, 36]]),
        input_cs=xyz_cs,
        output_cs=xyz_cs,
        wrong_output_cs=zyx_cs,
    )


def test_get_affine_form_input_output_coordinate_systems():
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
            a = NgffAffine.from_input_output_coordinate_systems(input_cs, output_cs)

            input_axes = input_cs.axes_names
            output_axes = output_cs.axes_names
            input_data = np.atleast_2d([data[a] for a in input_axes])
            output_data = np.atleast_2d([data[a] if a in input_axes else 0.0 for a in output_axes])

            transformed_data = a.transform_points(input_data)
            assert np.allclose(transformed_data, output_data)
