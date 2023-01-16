import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.transformations import Identity, MapAxis


def test_identity():
    assert np.array_equal(Identity().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3))
    assert np.array_equal(
        Identity().inverse().to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3)
    )
    assert np.array_equal(
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
    assert np.array_equal(
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
    with pytest.raises(ValueError):
        map_axis0.to_affine_matrix(input_axes=("x", "y", "z"), output_axes=("x", "y"))

    map_axis0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    assert np.array_equal(map_axis0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), np.eye(3))

    # map_axis1 is an example of invertible MapAxis; here it swaps x and y
    map_axis1 = MapAxis({"x": "y", "y": "x"})
    map_axis1_inverse = map_axis1.inverse()
    assert np.array_equal(
        map_axis1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.array_equal(
        map_axis1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        map_axis1_inverse.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
    assert np.array_equal(
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
    assert np.array_equal(
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
    assert np.array_equal(
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
    assert np.array_equal(
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


def test_sequence_mismatching_cs_inference():
    pass
    # original, affine, transformed = _test_sequence_helper()
    # _test_transformation(
    #     transformation=NgffSequence(
    #         [
    #             NgffScale(np.array([2, 3]), input_coordinate_system=yx_cs, output_coordinate_system=yx_cs),
    #             NgffScale(np.array([4, 5]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
    #         ]
    #     ),
    #     original=original,
    #     transformed=original * np.array([2 * 5, 3 * 4]),
    #     input_cs=yx_cs,
    #     output_cs=yx_cs,
    #     wrong_output_cs=xy_cs,
    #     test_inverse=True,
    # )


def test_sequence_2d_to_2d_with_c_with_mismatching_cs():
    pass


#     original, affine, transformed = _test_sequence_helper()
#     affine.input_coordinate_system = xy_cs
#     affine.output_coordinate_system = xy_cs
#
#     def _manual_xy_to_cyx(x: np.ndarray) -> np.ndarray:
#         return np.hstack((np.zeros(len(x)).reshape((len(x), 1)), np.fliplr(x)))
#
#     _test_transformation(
#         transformation=NgffSequence(
#             [
#                 NgffTranslation(np.array([1, 2]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
#                 NgffScale(np.array([3, 4]), input_coordinate_system=xy_cs, output_coordinate_system=xy_cs),
#                 affine,
#             ]
#         ),
#         original=_manual_xy_to_cyx(original),
#         transformed=_manual_xy_to_cyx(transformed),
#         input_cs=cyx_cs,
#         output_cs=cyx_cs,
#         wrong_output_cs=xyc_cs,
#         test_inverse=False,
#     )


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
