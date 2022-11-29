import numpy as np
import pytest

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transformations import (Affine, BaseTransformation,
                                               ByDimension, Identity, MapAxis,
                                               Rotation, Scale, Sequence,
                                               Translation)
from tests._core.conftest import (c_cs, cyx_cs, x_cs, xy_cs, xyc_cs, xyz_cs,
                                  y_cs, yx_cs, z_cs, zyx_cs)


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
    # missing input and output coordinate systems
    with pytest.raises(ValueError):
        assert np.allclose(transformation.transform_points(original), transformed)

    # missing output coordinate system
    transformation.input_coordinate_system = input_cs
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
        assert np.allclose(transformation.transform_points(original.ravel()), transformed.ravel())
    with pytest.raises(ValueError):
        assert np.allclose(transformation.transform_points(original.transpose()), transformed.transpose())
    with pytest.raises(ValueError):
        assert np.allclose(transformation.transform_points(np.expand_dims(original, 0)), np.expand_dims(transformed, 0))

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
    assert transformation.to_json() == BaseTransformation.from_json(transformation.to_json()).to_json()


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


def test_affine():
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


def test_sequence():
    original = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    affine = Affine(np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]))
    transformed = np.matmul(
        np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]),
        np.vstack([np.transpose((original + np.array([1, 2])) * np.array([3, 4])), [1] * len(original)]),
    )[:-1, :].T

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

    def _manual_xy_to_cyz(x: np.ndarray) -> np.ndarray:
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
        original=_manual_xy_to_cyz(original),
        transformed=_manual_xy_to_cyz(transformed),
        input_cs=cyx_cs,
        output_cs=cyx_cs,
        wrong_output_cs=xyc_cs,
        test_inverse=False,
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
