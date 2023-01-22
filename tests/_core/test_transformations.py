import math

import numpy as np
import pytest
import scipy.misc
import xarray.testing
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import SpatialData
from spatialdata._core.core_utils import get_dims
from spatialdata._core.models import Image2DModel
from spatialdata._core.transformations import (
    Affine,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
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
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("z"), output_axes=("z"))
    assert np.allclose(
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("x"), output_axes=("x")),
        np.array(
            [
                [1, 0],
                [0, 1],
            ]
        ),
    )
    # adding new axes with MapAxis (something that the Ngff MapAxis can't do)
    assert np.allclose(
        MapAxis({"z": "x"}).to_affine_matrix(input_axes=("x"), output_axes=("x", "z")),
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
    # checking that permuting the axes of an affine matrix and inverting it are operations that commute (the order doesn't matter)
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
    print("test with error:")
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
    print(sequence0)


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
    for t, e in zip(transformaions, expected):
        transformed = t._transform_coordinates(coords)
        xarray.testing.assert_allclose(transformed, e)


def _get_affine(small_translation: bool = True) -> Affine:
    theta = math.pi / 18
    k = 10.0 if small_translation else 1.0
    return Affine(
        [
            [2 * math.cos(theta), 2 * math.sin(-theta), -1000 / k],
            [2 * math.sin(theta), 2 * math.cos(theta), 300 / k],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


# TODO: when the io for 3D images and 3D labels work, add those tests
def test_transform_image_spatial_image(images: SpatialData):
    sdata = SpatialData(images={k: v for k, v in images.images.items() if isinstance(v, SpatialImage)})

    im = scipy.misc.face()
    im_element = Image2DModel.parse(im, dims=["y", "x", "c"])
    # TODO: remove this del later on when we are ready to test 3D images
    del sdata.images["image2d"]
    sdata.images["face"] = im_element

    affine = _get_affine(small_translation=False)
    affine.inverse().transform(affine.transform(sdata))
    # TODO: unpad the image
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_image_spatial_multiscale_spatial_image(images: SpatialData):
    sdata = SpatialData(images={k: v for k, v in images.images.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    affine.inverse().transform(affine.transform(sdata))
    # TODO: unpad the image
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, SpatialImage)})
    affine = _get_affine()
    affine.inverse().transform(affine.transform(sdata))
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_multiscale_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    affine.inverse().transform(affine.transform(sdata))
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


# TODO: maybe add methods for comparing the coordinates of elements so the below code gets less verbose
def test_transform_points(points: SpatialData):
    affine = _get_affine()
    new_points = affine.inverse().transform(affine.transform(points))
    keys0 = list(points.points.keys())
    keys1 = list(new_points.points.keys())
    assert keys0 == keys1
    for k in keys0:
        p0 = points.points[k]
        p1 = new_points.points[k]
        axes0 = get_dims(p0)
        axes1 = get_dims(p1)
        assert axes0 == axes1
        for ax in axes0:
            x0 = p0[ax].to_numpy()
            x1 = p1[ax].to_numpy()
            assert np.allclose(x0, x1)


def test_transform_polygons(polygons: SpatialData):
    affine = _get_affine()
    new_polygons = affine.inverse().transform(affine.transform(polygons))
    keys0 = list(polygons.polygons.keys())
    keys1 = list(new_polygons.polygons.keys())
    assert keys0 == keys1
    for k in keys0:
        p0 = polygons.polygons[k]
        p1 = new_polygons.polygons[k]
        for i in range(len(p0.geometry)):
            assert p0.geometry.iloc[i].almost_equals(p1.geometry.iloc[i])


def test_transform_shapes(shapes: SpatialData):
    affine = _get_affine()
    new_shapes = affine.inverse().transform(affine.transform(shapes))
    keys0 = list(shapes.shapes.keys())
    keys1 = list(new_shapes.shapes.keys())
    assert keys0 == keys1
    for k in keys0:
        p0 = shapes.shapes[k]
        p1 = new_shapes.shapes[k]
        assert np.allclose(p0.obsm["spatial"], p1.obsm["spatial"])


def test_set_transform_with_mismatching_cs(sdata: SpatialData):
    pass
    # input_css = [
    #     get_default_coordinate_system(t) for t in [(X, Y), (Y, X), (C, Y, X), (X, Y, Z), (Z, Y, X), (C, Z, Y, X)]
    # ]
    # for element_type in sdata._non_empty_elements():
    #     if element_type == "table":
    #         continue
    #     for v in getattr(sdata, element_type).values():
    #         for input_cs in input_css:
    #             affine = NgffAffine.from_input_output_coordinate_systems(input_cs, input_cs)
    #             set_transform(v, affine)


def test_assign_xy_scale_to_cyx_image():
    pass
    # xy_cs = get_default_coordinate_system(("x", "y"))
    # scale = NgffScale(np.array([2, 3]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs)
    # image = Image2DModel.parse(np.zeros((10, 10, 10)), dims=("c", "y", "x"))
    #
    # set_transform(image, scale)
    # t = get_transform(image)
    # pprint(t.to_dict())
    # print(t.to_affine())
    #
    # set_transform(image, scale.to_affine())
    # t = get_transform(image)
    # pprint(t.to_dict())
    # print(t.to_affine())


def test_assign_xyz_scale_to_cyx_image():
    pass
    # xyz_cs = get_default_coordinate_system(("x", "y", "z"))
    # scale = NgffScale(np.array([2, 3, 4]), input_coordinate_system=xyz_cs, output_coordinate_system=xyz_cs)
    # image = Image2DModel.parse(np.zeros((10, 10, 10)), dims=("c", "y", "x"))
    #
    # set_transform(image, scale)
    # t = get_transform(image)
    # pprint(t.to_dict())
    # print(t.to_affine())
    # pprint(t.to_affine().to_dict())
    #
    # set_transform(image, scale.to_affine())
    # t = get_transform(image)
    # pprint(t.to_dict())
    # print(t.to_affine())


def test_assign_cyx_scale_to_xyz_points():
    pass
    # cyx_cs = get_default_coordinate_system(("c", "y", "x"))
    # scale = NgffScale(np.array([1, 3, 2]), input_coordinate_system=cyx_cs, output_coordinate_system=cyx_cs)
    # points = PointsModel.parse(coords=np.zeros((10, 3)))
    #
    # set_transform(points, scale)
    # t = get_transform(points)
    # pprint(t.to_dict())
    # print(t.to_affine())
    #
    # set_transform(points, scale.to_affine())
    # t = get_transform(points)
    # pprint(t.to_dict())
    # print(t.to_affine())


def test_compose_in_xy_and_operate_in_cyx():
    pass
    # xy_cs = get_default_coordinate_system(("x", "y"))
    # cyx_cs = get_default_coordinate_system(("c", "y", "x"))
    # k = 0.5
    # scale = NgffScale([k, k], input_coordinate_system=xy_cs, output_coordinate_system=xy_cs)
    # theta = np.pi / 6
    # rotation = NgffAffine(
    #     np.array(
    #         [
    #             [np.cos(theta), -np.sin(theta), 0],
    #             [np.sin(theta), np.cos(theta), 0],
    #             [0, 0, 1],
    #         ]
    #     ),
    #     input_coordinate_system=xy_cs,
    #     output_coordinate_system=xy_cs,
    # )
    # sequence = NgffSequence([rotation, scale], input_coordinate_system=cyx_cs, output_coordinate_system=cyx_cs)
    # affine = sequence.to_affine()
    # print(affine)
    # assert affine.affine[0, 0] == 1.0
