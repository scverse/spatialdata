import math

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from numpy.random import default_rng

from spatialdata._core.centroids import get_centroids
from spatialdata._core.query.relational_query import get_element_instances
from spatialdata.models import Labels2DModel, Labels3DModel, PointsModel, TableModel, get_axes_names
from spatialdata.transformations import Affine, Identity, get_transformation, set_transformation

RNG = default_rng(42)


def _get_affine() -> Affine:
    theta: float = math.pi / 18
    k = 10.0
    return Affine(
        [
            [2 * math.cos(theta), 2 * math.sin(-theta), -1000 / k],
            [2 * math.sin(theta), 2 * math.cos(theta), 300 / k],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


affine = _get_affine()


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("is_3d", [False, True])
def test_get_centroids_points(points, coordinate_system: str, is_3d: bool):
    element = points["points_0"].compute()
    element.index = np.arange(len(element)) + 10
    element = PointsModel.parse(element)

    # by default, the coordinate system is global and the points are 2D; let's modify the points as instructed by the
    # test arguments
    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    if is_3d:
        element["z"] = element["x"]

    axes = get_axes_names(element)
    centroids = get_centroids(element, coordinate_system=coordinate_system)

    # the axes of the centroids should be the same as the axes of the element
    assert centroids.columns.tolist() == list(axes)

    # check the index is preserved
    assert np.array_equal(centroids.index.values, element.index.values)

    # the centroids should not contain extra columns
    assert "genes" in element.columns and "genes" not in centroids.columns

    # the centroids transformation to the target coordinate system should be an Identity because the transformation has
    # already been applied
    assert get_transformation(centroids, to_coordinate_system=coordinate_system) == Identity()

    # let's check the values
    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, element[list(axes)].compute().values)
    else:
        matrix = affine.to_affine_matrix(input_axes=axes, output_axes=axes)
        centroids_untransformed = element[list(axes)].compute().values
        n = len(axes)
        centroids_transformed = np.dot(centroids_untransformed, matrix[:n, :n].T) + matrix[:n, n]
        assert np.allclose(centroids.compute().values, centroids_transformed)


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("shapes_name", ["circles", "poly", "multipoly"])
def test_get_centroids_shapes(shapes, coordinate_system: str, shapes_name: str):
    element = shapes[shapes_name]
    element.index = np.arange(len(element)) + 10

    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    centroids = get_centroids(element, coordinate_system=coordinate_system)

    assert np.array_equal(centroids.index.values, element.index.values)

    if shapes_name == "circles":
        xy = element.geometry.get_coordinates().values
    else:
        assert shapes_name in ["poly", "multipoly"]
        xy = element.geometry.centroid.get_coordinates().values

    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, xy)
    else:
        matrix = affine.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        centroids_transformed = np.dot(xy, matrix[:2, :2].T) + matrix[:2, 2]
        assert np.allclose(centroids.compute().values, centroids_transformed)


@pytest.mark.parametrize("coordinate_system", ["global", "aligned"])
@pytest.mark.parametrize("is_multiscale", [False, True])
@pytest.mark.parametrize("is_3d", [False, True])
@pytest.mark.parametrize("return_background", [False, True])
def test_get_centroids_labels(
    labels, coordinate_system: str, is_multiscale: bool, is_3d: bool, return_background: bool
):
    scale_factors = [2] if is_multiscale else None
    if is_3d:
        model = Labels3DModel
        array = np.array(
            [
                [
                    [0, 0, 10, 10],
                    [0, 0, 10, 10],
                ],
                [
                    [20, 20, 10, 10],
                    [20, 20, 10, 10],
                ],
            ]
        )
        expected_centroids = pd.DataFrame(
            {
                "x": [1, 3, 1],
                "y": [1, 1.0, 1],
                "z": [0.5, 1, 1.5],
            },
            index=[0, 1, 2],
        )
        if not return_background:
            expected_centroids = expected_centroids.drop(index=0)
    else:
        array = np.array(
            [
                [10, 10, 10, 10],
                [20, 20, 20, 20],
                [20, 20, 20, 20],
                [20, 20, 20, 20],
            ]
        )
        model = Labels2DModel
        expected_centroids = pd.DataFrame(
            {
                "x": [2, 2],
                "y": [0.5, 2.5],
            },
            index=[1, 2],
        )
    element = model.parse(array, scale_factors=scale_factors)

    if coordinate_system == "aligned":
        set_transformation(element, transformation=affine, to_coordinate_system=coordinate_system)
    centroids = get_centroids(element, coordinate_system=coordinate_system, return_background=return_background)

    labels_indices = get_element_instances(element, return_background=return_background)
    assert np.array_equal(centroids.index.values, labels_indices)

    if not return_background:
        assert 0 not in centroids.index

    if coordinate_system == "global":
        assert np.array_equal(centroids.compute().values, expected_centroids.values)
    else:
        axes = get_axes_names(element)
        n = len(axes)
        # the axes from the labels have 'x' last, but we want it first to manually transform the points, so we sort
        matrix = affine.to_affine_matrix(input_axes=sorted(axes), output_axes=sorted(axes))
        centroids_transformed = np.dot(expected_centroids.values, matrix[:n, :n].T) + matrix[:n, n]
        assert np.allclose(centroids.compute().values, centroids_transformed)


def test_get_centroids_invalid_element(images):
    # cannot compute centroids for images
    with pytest.raises(ValueError, match="Expected a `Labels` element. Found an `Image` instead."):
        get_centroids(images["image2d"])

    # cannot compute centroids for tables
    N = 10
    adata = TableModel.parse(
        AnnData(X=RNG.random((N, N)), obs={"region": ["dummy" for _ in range(N)], "instance_id": np.arange(N)}),
        region="dummy",
        region_key="region",
        instance_key="instance_id",
    )
    with pytest.raises(ValueError, match="The object type <class 'anndata._core.anndata.AnnData'> is not supported."):
        get_centroids(adata)


def test_get_centroids_invalid_coordinate_system(points):
    with pytest.raises(AssertionError, match="No transformation to coordinate system"):
        get_centroids(points["points_0"], coordinate_system="invalid")
