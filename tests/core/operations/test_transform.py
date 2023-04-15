import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.misc
from geopandas.testing import geom_almost_equals
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata import SpatialData, transform
from spatialdata._utils import unpad_raster
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, get_axes_names
from spatialdata.transformations.operations import (
    align_elements_using_landmarks,
    get_transformation,
    get_transformation_between_coordinate_systems,
    get_transformation_between_landmarks,
    remove_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import (
    Affine,
    Identity,
    Scale,
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


# TODO: when the io for 3D images and 3D labels work, add those tests
def test_transform_image_spatial_image(images: SpatialData):
    sdata = SpatialData(images={k: v for k, v in images.images.items() if isinstance(v, SpatialImage)})

    VISUAL_DEBUG = False
    if VISUAL_DEBUG:
        im = scipy.misc.face()
        im_element = Image2DModel.parse(im, dims=["y", "x", "c"])
        del sdata.images["image2d"]
        sdata.images["face"] = im_element

    affine = _get_affine(small_translation=False)
    padded = transform(
        transform(sdata, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
    _unpad_rasters(padded)
    # raise NotImplementedError("TODO: plot the images")
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_image_spatial_multiscale_spatial_image(images: SpatialData):
    sdata = SpatialData(images={k: v for k, v in images.images.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    padded = transform(
        transform(sdata, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
    _unpad_rasters(padded)
    # TODO: unpad the image
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, SpatialImage)})
    affine = _get_affine()
    padded = transform(
        transform(sdata, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
    _unpad_rasters(padded)
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_multiscale_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    padded = transform(
        transform(sdata, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
    _unpad_rasters(padded)
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


# TODO: maybe add methods for comparing the coordinates of elements so the below code gets less verbose
def test_transform_points(points: SpatialData):
    affine = _get_affine()
    new_points = transform(
        transform(points, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
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
    new_shapes = transform(
        transform(shapes, affine, maintain_positioning=False), affine.inverse(), maintain_positioning=False
    )
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

    # error 1
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
    # error 2
    with pytest.raises(RuntimeError):
        get_transformation_between_coordinate_systems(
            full_sdata,
            source_coordinate_system="my_space0",
            target_coordinate_system="global",
            intermediate_coordinate_systems="globalE",
        )
    # error 3
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


def test_transform_elements_and_entire_spatial_data_object(sdata: SpatialData):
    # TODO: we are just applying the transformation,
    #  we are not checking it is correct. We could improve this test
    scale = Scale([2], axes=("x",))
    for element in sdata._gen_elements_values():
        set_transformation(element, scale, "my_space")
        sdata.transform_element_to_coordinate_system(element, "my_space")
    sdata.transform_to_coordinate_system("my_space")


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
