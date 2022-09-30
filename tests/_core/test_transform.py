import numpy as np
import pytest

from spatialdata._core.transform import get_transformation_from_json


def test_identity():
    assert np.allclose(
        act('{"coordinateTransformations": {"type": "identity"}}', ndim=2),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
    )


@pytest.mark.skip()
def test_map_index():
    raise NotImplementedError()


@pytest.mark.skip()
def test_map_axis():
    raise NotImplementedError()


def test_translation_3d():
    assert np.allclose(
        act('{"coordinateTransformations": {"type": "translation", "translation": [1, 2, 3]}}', ndim=3),
        [[2, 4, 6], [5, 7, 9], [8, 10, 12], [11, 13, 15]],
    )


def test_scale_3d():
    assert np.allclose(
        act('{"coordinateTransformations": {"type": "scale", "scale": [1, 2, 3]}}', ndim=3),
        [[1, 4, 9], [4, 10, 18], [7, 16, 27], [10, 22, 36]],
    )


def test_affine_2d():
    assert np.allclose(
        act('{"coordinateTransformations": {"type": "affine", "affine": [1, 2, 3, 4, 5, 6]}}', ndim=2),
        [[8, 20], [14, 38], [20, 56]],
    )


def test_rotation_2d():
    assert np.allclose(
        act('{"coordinateTransformations": {"type": "rotation", "affine": [0, -1, 1, 0]}}', ndim=2),
        [[-2, 1], [-4, 3], [-6, 5]],
    )


# output from np.matmul(np.array([[5, 6, 7], [8, 9, 10], [0, 0, 1]]), np.vstack([np.transpose((xyz + np.array([1, 2])) * np.array([3, 4])), [1, 1, 1]]))[:-1, :].T
def test_sequence_2d():
    assert np.allclose(
        act(
            '{"coordinateTransformations": {"type": "sequence", "transformations": [{"type": "translation", '
            '"translation": [1, 2]}, {"type": "scale", "scale": [3, 4]}, {"type": "affine", '
            '"affine": [5, 6, 7, 8, 9, 10]}}',
            ndim=2,
        ),
        [[133, 202], [211, 322], [289, 442]],
    )


def test_sequence_3d():
    assert np.allclose(
        act(
            '{"coordinateTransformations": {"type": "sequence", "transformations": [{"type": "translation", '
            '"translation": [1, 2, 3]}, {"type": "scale", "scale": [4, 5, 6]}, {"type": "translation", '
            '"translation": [7, 8, 9]}}',
            ndim=3,
        ),
        [[15, 28, 45], [27, 43, 63], [39, 58, 81], [51, 73, 99]],
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
def test_inverse_of_inverse():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_translation():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_scale():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_affine_2d():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_rotation_2d():
    raise NotImplementedError()


@pytest.mark.skip()
def test_inverse_of_sequence_2d():
    raise NotImplementedError()


@pytest.mark.skip()
def test_bijection():
    raise NotImplementedError()


@pytest.mark.skip()
def test_by_dimension():
    raise NotImplementedError()


def act(s: str, ndim: int) -> np.array:
    if ndim == 2:
        points = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    elif ndim == 3:
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=float)
    else:
        raise ValueError(f"Invalid ndim: {ndim}")
    return get_transformation_from_json(s).transform_points(points)


# TODO: test that the scale, translation and rotation as above gives the same as the rotation as an affine matrices
# TODO: test also affine transformations to embed 2D points in a 3D space
