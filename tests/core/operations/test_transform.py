import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from geopandas.testing import geom_almost_equals
from xarray import DataArray, DataTree

from spatialdata import transform
from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.spatialdata import SpatialData
from spatialdata._utils import unpad_raster
from spatialdata.models import PointsModel, ShapesModel, get_axes_names
from spatialdata.transformations.operations import (
    align_elements_using_landmarks,
    get_transformation,
    get_transformation_between_coordinate_systems,
    get_transformation_between_landmarks,
    remove_transformation,
    remove_transformations_to_coordinate_system,
    set_transformation,
)
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Identity,
    Scale,
    Sequence,
    Translation,
)


class TestElementsTransform:
    @pytest.mark.parametrize(
        "transform", [Scale(np.array([1, 2, 3]), axes=("x", "y", "z")), Scale(np.array([2]), axes=("x",))]
    )
    def test_points(
        self,
        tmp_path: str,
        points: SpatialData,
        transform: Scale,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        set_transformation(points.points["points_0"], transform)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)

        # when the points are 2d and we have a scale 3d, the 3rd dimension is not saved to disk, so we have to remove
        # it from the assertion
        assert isinstance(transform, Scale)
        axes = get_axes_names(points.points["points_0"])
        expected_scale = Scale(transform.to_scale_vector(axes), axes)
        assert get_transformation(new_sdata.points["points_0"]) == expected_scale

    @pytest.mark.parametrize(
        "transform", [Scale(np.array([1, 2, 3]), axes=("x", "y", "z")), Scale(np.array([2]), axes=("x",))]
    )
    def test_shapes(
        self,
        tmp_path: str,
        shapes: SpatialData,
        transform: Scale,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        set_transformation(shapes.shapes["circles"], transform, "my_coordinate_system1")
        set_transformation(shapes.shapes["circles"], transform, "my_coordinate_system2")

        shapes.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        loaded_transform1 = get_transformation(new_sdata.shapes["circles"], "my_coordinate_system1")
        loaded_transform2 = get_transformation(new_sdata.shapes["circles"], get_all=True)["my_coordinate_system2"]

        # when the points are 2d and we have a scale 3d, the 3rd dimension is not saved to disk, so we have to remove
        # it from the assertion
        assert isinstance(transform, Scale)
        axes = get_axes_names(new_sdata.shapes["circles"])
        expected_scale = Scale(transform.to_scale_vector(axes), axes)
        assert loaded_transform1 == expected_scale
        assert loaded_transform2 == expected_scale

    def test_coordinate_systems(self, shapes: SpatialData) -> None:
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        set_transformation(shapes.shapes["circles"], ct, "test")
        assert set(shapes.coordinate_systems) == {"global", "test"}

    @pytest.mark.skip("Physical units are not supported for now with the new implementation for transformations")
    def test_physical_units(self, tmp_path: str, shapes: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        shapes.write(tmpdir)
        set_transformation(shapes.shapes["circles"], ct, "test", shapes)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.coordinate_systems["test"]._axes[0].unit == "micrometers"


def _get_affine(small_translation: bool = True, theta: float = math.pi / 18) -> Affine:
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


def _unpad_rasters(sdata: SpatialData) -> SpatialData:
    new_images = {}
    new_labels = {}
    for name, image in sdata.images.items():
        unpadded = unpad_raster(image)
        new_images[name] = unpadded
    for name, label in sdata.labels.items():
        unpadded = unpad_raster(label)
        new_labels[name] = unpadded
    return SpatialData(images=new_images, labels=new_labels)


def _postpone_transformation(
    sdata: SpatialData, from_coordinate_system: str, to_coordinate_system: str, transformation: BaseTransformation
):
    for element in sdata._gen_spatial_element_values():
        d = get_transformation(element, get_all=True)
        assert isinstance(d, dict)
        assert len(d) == 1
        t = d[from_coordinate_system]
        sequence = Sequence([t, transformation])
        set_transformation(element, sequence, to_coordinate_system)


@pytest.mark.parametrize("element_type", ["image", "labels"])
@pytest.mark.parametrize("multiscale", [False, True])
def test_transform_raster(full_sdata: SpatialData, element_type: str, multiscale: bool):
    datatype = DataTree if multiscale else DataArray

    if element_type == "image":
        sdata = SpatialData(images={k: v for k, v in full_sdata.images.items() if isinstance(v, datatype)})
    else:
        assert element_type == "labels"
        sdata = SpatialData(labels={k: v for k, v in full_sdata.labels.items() if isinstance(v, datatype)})

    affine = _get_affine(small_translation=False)

    _postpone_transformation(
        sdata, from_coordinate_system="global", to_coordinate_system="transformed", transformation=affine
    )
    sdata_transformed = transform(sdata, to_coordinate_system="transformed")

    _postpone_transformation(
        sdata_transformed,
        from_coordinate_system="transformed",
        to_coordinate_system="transformed_back",
        transformation=affine.inverse(),
    )
    padded = transform(sdata_transformed, to_coordinate_system="transformed_back")

    # cleanup to make the napari visualization less cluttered
    remove_transformations_to_coordinate_system(sdata, "transformed")
    remove_transformations_to_coordinate_system(sdata_transformed, "transformed_back")

    unpadded = _unpad_rasters(padded)

    e0 = get_extent(sdata)
    e1 = get_extent(unpadded, coordinate_system="transformed_back")
    assert are_extents_equal(e0, e1)

    # Interactive([sdata, unpadded])
    # TODO: above we compared the alignment; compare also the data (this need to be tolerant to the interporalation and
    #  should be done after https://github.com/scverse/spatialdata/issues/165 is fixed to have better results


# TODO: maybe add methods for comparing the coordinates of elements so the below code gets less verbose
def test_transform_points(points: SpatialData):
    affine = _get_affine()

    _postpone_transformation(
        points, from_coordinate_system="global", to_coordinate_system="global", transformation=affine
    )
    sdata_transformed = transform(points, to_coordinate_system="global")

    _postpone_transformation(
        sdata_transformed,
        from_coordinate_system="global",
        to_coordinate_system="global",
        transformation=affine.inverse(),
    )
    new_points = transform(sdata_transformed, to_coordinate_system="global")

    keys0 = list(points.points.keys())
    keys1 = list(new_points.points.keys())
    assert keys0 == keys1
    for k in keys0:
        p0 = points.points[k]
        p1 = new_points.points[k]
        axes0 = get_axes_names(p0)
        axes1 = get_axes_names(p1)
        assert axes0 == axes1
        for ax in axes0:
            x0 = p0[ax].to_dask_array().compute()
            x1 = p1[ax].to_dask_array().compute()
            assert np.allclose(x0, x1)


def test_transform_shapes(shapes: SpatialData):
    affine = _get_affine()

    _postpone_transformation(
        shapes, from_coordinate_system="global", to_coordinate_system="global", transformation=affine
    )
    sdata_transformed = transform(shapes, to_coordinate_system="global")

    _postpone_transformation(
        sdata_transformed,
        from_coordinate_system="global",
        to_coordinate_system="global",
        transformation=affine.inverse(),
    )
    new_shapes = transform(sdata_transformed, to_coordinate_system="global")

    keys0 = list(shapes.shapes.keys())
    keys1 = list(new_shapes.shapes.keys())
    assert keys0 == keys1
    for k in keys0:
        p0 = shapes.shapes[k]
        p1 = new_shapes.shapes[k]
        assert geom_almost_equals(p0["geometry"], p1["geometry"])


def test_map_coordinate_systems_single_path(full_sdata: SpatialData):
    scale = Scale([2], axes=("x",))
    translation = Translation([100], axes=("x",))

    im = full_sdata.images["image2d_multiscale"]
    la = full_sdata.labels["labels2d"]
    po = full_sdata.shapes["multipoly"]

    set_transformation(im, scale)
    set_transformation(po, translation)
    set_transformation(po, translation, "my_space")
    set_transformation(po, scale)
    # identity
    assert (
        get_transformation_between_coordinate_systems(
            full_sdata, source_coordinate_system="global", target_coordinate_system="global"
        )
        == Identity()
    )
    assert (
        get_transformation_between_coordinate_systems(
            full_sdata, source_coordinate_system=la, target_coordinate_system=la
        )
        == Identity()
    )

    # intrinsic coordinate system (element) to extrinsic coordinate system and back
    t0 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system=im, target_coordinate_system="global"
    )
    t1 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system="global", target_coordinate_system=im
    )
    t2 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system=po, target_coordinate_system="my_space"
    )
    t3 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system="my_space", target_coordinate_system=po
    )
    assert np.allclose(
        t0.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [2, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        t1.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [0.5, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        t2.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [1, 0, 100],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
    assert np.allclose(
        t3.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [1, 0, -100],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )

    # intrinsic to intrinsic (element to element)
    t4 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system=im, target_coordinate_system=la
    )
    assert np.allclose(
        t4.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [2, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )

    # extrinsic to extrinsic
    t5 = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system="global", target_coordinate_system="my_space"
    )
    assert np.allclose(
        t5.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [0.5, 0, 100],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_coordinate_systems_with_shortest_paths(full_sdata: SpatialData):
    scale = Scale([2], axes=("x",))
    translation = Translation([100], axes=("x",))
    cs1_to_cs2 = Sequence([scale.inverse(), translation])

    im = full_sdata.images["image2d_multiscale"]
    la = full_sdata.labels["labels2d"]
    po = full_sdata.shapes["multipoly"]
    po2 = full_sdata.shapes["circles"]

    set_transformation(im, {"cs1": Identity()}, set_all=True)
    set_transformation(la, {"cs2": Identity()}, set_all=True)

    with pytest.raises(RuntimeError):  # error 0
        get_transformation_between_coordinate_systems(full_sdata, im, la)

    set_transformation(po, {"cs1": scale, "cs2": translation}, set_all=True)

    t = get_transformation_between_coordinate_systems(full_sdata, im, la, shortest_path=True)
    assert len(t.transformations) == 4
    t = get_transformation_between_coordinate_systems(full_sdata, im, la, shortest_path=False)
    assert len(t.transformations) == 4

    set_transformation(im, cs1_to_cs2, "cs2")

    with pytest.raises(RuntimeError):  # error 4
        get_transformation_between_coordinate_systems(full_sdata, im, la, shortest_path=False)

    t = get_transformation_between_coordinate_systems(full_sdata, im, la, shortest_path=True)

    assert len(t.transformations) == 2

    set_transformation(po2, {"cs1": scale, "cs2": translation}, set_all=True)

    get_transformation_between_coordinate_systems(full_sdata, im, la, shortest_path=True)


def test_map_coordinate_systems_zero_or_multiple_paths(full_sdata):
    scale = Scale([2], axes=("x",))

    im = full_sdata.images["image2d_multiscale"]
    la = full_sdata.labels["labels2d"]

    set_transformation(im, scale, "my_space0")
    set_transformation(la, scale, "my_space0")

    # error 0
    with pytest.raises(RuntimeError):
        get_transformation_between_coordinate_systems(
            full_sdata, source_coordinate_system="my_space0", target_coordinate_system="globalE"
        )

    # error 2
    with pytest.raises(RuntimeError):
        t = get_transformation_between_coordinate_systems(
            full_sdata, source_coordinate_system="my_space0", target_coordinate_system="global"
        )

    t = get_transformation_between_coordinate_systems(
        full_sdata,
        source_coordinate_system="my_space0",
        target_coordinate_system="global",
        intermediate_coordinate_systems=im,
    )
    assert np.allclose(
        t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [0.5, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
    # error 3
    with pytest.raises(RuntimeError):
        get_transformation_between_coordinate_systems(
            full_sdata,
            source_coordinate_system="my_space0",
            target_coordinate_system="global",
            intermediate_coordinate_systems="globalE",
        )
    # error 5
    with pytest.raises(RuntimeError):
        get_transformation_between_coordinate_systems(
            full_sdata,
            source_coordinate_system="my_space0",
            target_coordinate_system="global",
            intermediate_coordinate_systems="global",
        )


def test_map_coordinate_systems_non_invertible_transformations(full_sdata):
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
    im = full_sdata.images["image2d_multiscale"]
    set_transformation(im, affine)
    t = get_transformation_between_coordinate_systems(
        full_sdata, source_coordinate_system=im, target_coordinate_system="global"
    )
    assert np.allclose(
        t.to_affine_matrix(input_axes=("x", "y"), output_axes=("c", "y", "x")),
        np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]
        ),
    )
    with pytest.raises(RuntimeError):
        # error 0 (no path between source and target because the affine matrix is not invertible)
        try:
            get_transformation_between_coordinate_systems(
                full_sdata, source_coordinate_system="global", target_coordinate_system=im
            )
        except RuntimeError as e:
            assert str(e) == "No path found between the two coordinate systems"
            raise e


def test_map_coordinate_systems_long_path(full_sdata):
    im = full_sdata.images["image2d_multiscale"]
    la0 = full_sdata.labels["labels2d"]
    la1 = full_sdata.labels["labels2d_multiscale"]
    po = full_sdata.shapes["multipoly"]

    scale = Scale([2], axes=("x",))

    remove_transformation(im, remove_all=True)
    set_transformation(im, scale.inverse(), "my_space0")
    set_transformation(im, scale, "my_space1")

    remove_transformation(la0, remove_all=True)
    set_transformation(la0, scale.inverse(), "my_space1")
    set_transformation(la0, scale, "my_space2")

    remove_transformation(la1, remove_all=True)
    set_transformation(la1, scale.inverse(), "my_space1")
    set_transformation(la1, scale, "my_space2")

    remove_transformation(po, remove_all=True)
    set_transformation(po, scale.inverse(), "my_space2")
    set_transformation(po, scale, "my_space3")

    with pytest.raises(RuntimeError):
        # error 1
        get_transformation_between_coordinate_systems(
            full_sdata, source_coordinate_system="my_space0", target_coordinate_system="my_space3"
        )

    t = get_transformation_between_coordinate_systems(
        full_sdata,
        source_coordinate_system="my_space0",
        target_coordinate_system="my_space3",
        intermediate_coordinate_systems=la1,
    )
    assert np.allclose(
        t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        np.array(
            [
                [64.0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )


@pytest.mark.parametrize("maintain_positioning", [True, False])
def test_transform_elements_and_entire_spatial_data_object(full_sdata: SpatialData, maintain_positioning: bool):
    k = 10.0
    scale = Scale([k], axes=("x",))
    translation = Translation([k], axes=("x",))
    sequence = Sequence([scale, translation])
    for _, element_name, _ in full_sdata.gen_spatial_elements():
        element = full_sdata[element_name]
        set_transformation(element, sequence, "my_space")
        transformed_element = full_sdata.transform_element_to_coordinate_system(
            element_name, "my_space", maintain_positioning=maintain_positioning
        )
        t = get_transformation(transformed_element, to_coordinate_system="my_space")
        a = t.to_affine_matrix(input_axes=("x",), output_axes=("x",))
        d = get_transformation(transformed_element, get_all=True)
        assert isinstance(d, dict)
        if maintain_positioning:
            assert set(d.keys()) == {"global", "my_space"}
            a2 = d["global"].to_affine_matrix(input_axes=("x",), output_axes=("x",))
            assert np.allclose(a, a2)
            if isinstance(element, DataArray | DataTree):
                assert np.allclose(a, np.array([[1 / k, 0], [0, 1]]))
            else:
                assert np.allclose(a, np.array([[1 / k, -k / k], [0, 1]]))
        else:
            assert set(d.keys()) == {"my_space"}
            if isinstance(element, DataArray | DataTree):
                assert np.allclose(a, np.array([[1, k], [0, 1]]))
            else:
                assert np.allclose(a, np.array([[1, 0], [0, 1]]))

    # this calls transform_element_to_coordinate_system() internally()
    _ = full_sdata.transform_to_coordinate_system("my_space", maintain_positioning=maintain_positioning)


@pytest.mark.parametrize("maintain_positioning", [True, False])
def test_transform_elements_and_entire_spatial_data_object_multi_hop(
    full_sdata: SpatialData, maintain_positioning: bool
):
    k = 10.0
    scale = Scale([k], axes=("x",))
    for element in full_sdata._gen_spatial_element_values():
        set_transformation(element, scale, "my_space")

    # testing the scenario "element1 -> cs1 <- element2 -> cs2" and transforming element1 to cs2
    translation = Translation([k], axes=("x",))
    full_sdata["proxy_element"] = full_sdata.shapes["multipoly"].copy()
    set_transformation(
        full_sdata["proxy_element"], {"multi_hop_space": translation, "my_space": Identity()}, set_all=True
    )

    # otherwise we have multiple paths to go from my_space to multi_hop_space
    for element in full_sdata._gen_spatial_element_values():
        d = get_transformation(element, get_all=True)
        assert isinstance(d, dict)
        if "global" in d:
            remove_transformation(element, "global")

    for element in full_sdata._gen_spatial_element_values():
        transformed_element = full_sdata.transform_element_to_coordinate_system(
            element, "multi_hop_space", maintain_positioning=maintain_positioning
        )
        temp = SpatialData(
            images=dict(full_sdata.images),
            labels=dict(full_sdata.labels),
            points=dict(full_sdata.points),
            shapes=dict(full_sdata.shapes),
            table=full_sdata["table"],
        )
        temp["transformed_element"] = transformed_element
        transformation = get_transformation_between_coordinate_systems(
            temp, temp["transformed_element"], "multi_hop_space"
        )
        affine = transformation.to_affine_matrix(input_axes=("x",), output_axes=("x",))
        d = get_transformation(transformed_element, get_all=True)
        assert isinstance(d, dict)
        if maintain_positioning:
            if full_sdata.locate_element(element) == ["shapes/proxy_element"]:
                # non multi-hop case, since there is a direct transformation
                assert set(d.keys()) == {"multi_hop_space", "my_space"}
                affine2 = d["my_space"].to_affine_matrix(input_axes=("x",), output_axes=("x",))
                # I'd say that in the general case maybe they are not necessarily identical, but in this case they are
                assert np.allclose(affine, affine2)
                assert np.allclose(affine, np.array([[1, -k], [0, 1]]))
            elif isinstance(element, DataArray | DataTree):
                assert set(d.keys()) == {"my_space"}
                assert np.allclose(affine, np.array([[1, k], [0, 1]]))
            else:
                assert set(d.keys()) == {"my_space"}
                assert np.allclose(affine, np.array([[1, 0], [0, 1]]))
        else:
            assert set(d.keys()) == {"multi_hop_space"}
            if full_sdata.locate_element(element) == ["shapes/proxy_element"]:
                # non multi-hop case, since there is a direct transformation
                assert np.allclose(affine, np.array([[1, 0], [0, 1]]))
            elif isinstance(element, DataArray | DataTree):
                assert np.allclose(affine, np.array([[1, k], [0, 1]]))
            else:
                assert np.allclose(affine, np.array([[1, 0], [0, 1]]))


def test_transformations_between_coordinate_systems(images):
    # just a test that all the code is executed without errors and a quick test that the affine matrix is correct.
    # For a full test the notebooks are more exhaustive
    with tempfile.TemporaryDirectory() as tmpdir:
        images.write(Path(tmpdir) / "sdata.zarr")
        el0 = images.images["image2d"]
        el1 = images.images["image2d_multiscale"]
        set_transformation(el0, {"global0": Identity()}, set_all=True, write_to_sdata=images)
        set_transformation(el1, {"global1": Identity()}, set_all=True, write_to_sdata=images)
        for positive_determinant in [True, False]:
            reference_landmarks_coords = np.array([[0, 0], [0, 1], [1, 1], [3, 3]])
            if positive_determinant:
                moving_landmarks_coords = np.array([[0, 0], [0, 2], [2, 2], [6, 6]])
            else:
                moving_landmarks_coords = np.array([[0, 0], [0, -2], [2, -2], [6, -6]])

            reference_landmarks_shapes = ShapesModel.parse(reference_landmarks_coords, geometry=0, radius=10)
            moving_landmarks_shapes = ShapesModel.parse(np.array(moving_landmarks_coords), geometry=0, radius=10)
            reference_landmarks_points = PointsModel.parse(reference_landmarks_coords)
            moving_landmarks_points = PointsModel.parse(moving_landmarks_coords)

            for reference_landmarks, moving_landmarks in [
                (reference_landmarks_shapes, moving_landmarks_shapes),
                (reference_landmarks_points, moving_landmarks_points),
            ]:
                affine = get_transformation_between_landmarks(reference_landmarks, moving_landmarks)
                # testing a transformation with determinant > 0 for shapes
                # and a transformation with determinant < 0 for points
                if positive_determinant:
                    assert np.allclose(
                        affine.matrix,
                        np.array(
                            [
                                [0.5, 0, 0],
                                [0, 0.5, 0],
                                [0, 0, 1],
                            ]
                        ),
                    )
                else:
                    assert np.allclose(
                        affine.matrix,
                        np.array(
                            [
                                [0.5, 0, 0],
                                [0, -0.5, 0],
                                [0, 0, 1],
                            ]
                        ),
                    )
                for sdata in [images, None]:
                    align_elements_using_landmarks(
                        references_coords=reference_landmarks,
                        moving_coords=moving_landmarks,
                        reference_element=el0,
                        moving_element=el1,
                        reference_coordinate_system="global0",
                        moving_coordinate_system="global1",
                        new_coordinate_system="global2",
                        write_to_sdata=sdata,
                    )
                assert "global2" in images.coordinate_systems


def test_transform_until_0_0_15(points):
    from spatialdata._core.operations.transform import ERROR_MSG_AFTER_0_0_15

    t0 = Identity()
    t1 = Translation([10], axes=("x",))
    # only one between `transformation` and `to_coordinate_system` can be passed
    with pytest.raises(RuntimeError, match=ERROR_MSG_AFTER_0_0_15[:10]):
        transform(points, transformation=t0, to_coordinate_system="t0")

    # and need to pass at least one
    with pytest.raises(RuntimeError, match=ERROR_MSG_AFTER_0_0_15[:10]):
        transform(points)

    # need to use `to_coordinate_system`, not transformation`
    with pytest.raises(RuntimeError, match=ERROR_MSG_AFTER_0_0_15[:10]):
        transform(points["points_0_1"], transformation=t1)

    # except, for convenience to the user, when there is only a transformation in the element, and it coincides to the
    # one passed as argument to `transformation`
    transform(points["points_0"], transformation=t0)

    # but not for spatialdata objects, here we need to use `to_coordinate_system`
    with pytest.raises(RuntimeError, match=ERROR_MSG_AFTER_0_0_15[:10]):
        transform(points, transformation=t0)

    # correct way to use it
    transform(points, to_coordinate_system="global")

    # finally, when `maintain_positioning` is True, we can use either `transformation` or `to_coordinate_system`, as
    # long as excatly one of them is passed
    with pytest.raises(AssertionError, match="When maintain_positioning is True, only one "):
        transform(points, maintain_positioning=True)

    with pytest.raises(AssertionError, match="When maintain_positioning is True, only one "):
        transform(points, transformation=t0, to_coordinate_system="global", maintain_positioning=True)

    transform(points, transformation=t0, maintain_positioning=True)
    transform(points, to_coordinate_system="global", maintain_positioning=True)
