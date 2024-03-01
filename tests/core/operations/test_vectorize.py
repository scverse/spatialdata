import numpy as np
import pytest
from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata._core.operations.vectorize import to_circles
from spatialdata.datasets import blobs
from spatialdata.testing import assert_elements_are_identical
from spatialdata.transformations.operations import set_transformation

from tests.core.operations.test_transform import _get_affine

# the tests operate on different elements, hence we can initialize the data once without conflicts
sdata = blobs()
affine = _get_affine()
matrix = affine.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))


@pytest.mark.parametrize("is_3d", [False, True])
def test_labels_to_circles(is_3d: bool) -> None:
    pass


def test_circles_to_circles() -> None:
    element = sdata["blobs_circles"].iloc[:2]
    set_transformation(element, affine, "aligned")
    new_circles = to_circles(element, target_coordinate_system="aligned")

    # compare the returned circles
    old_centroids = get_centroids(element)
    set_transformation(old_centroids, affine, "aligned")
    old_centroids_transformed = transform(old_centroids, to_coordinate_system="aligned")

    new_centroids = get_centroids(new_circles, coordinate_system="aligned")
    assert_elements_are_identical(new_centroids, old_centroids_transformed)

    np.allclose(new_circles.radius, 2 * element.radius)


def test_polygons_to_circles() -> None:
    pass


def test_multipolygons_to_circles() -> None:
    pass


def test_points_images_to_circles() -> None:
    with pytest.raises(RuntimeError, match=r"Cannot apply to_circles\(\) to images."):
        to_circles(sdata["blobs_image"], target_coordinate_system="global")
    with pytest.raises(RuntimeError, match="Unsupported type"):
        to_circles(sdata["blobs_points"], target_coordinate_system="global")
