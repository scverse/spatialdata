import math
from pathlib import Path

import numpy as np
import pytest
import scipy.misc
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core.core_utils import _get_transform, _set_transform, get_dims
from spatialdata._core.models import Image2DModel
from spatialdata._core.transformations import Affine, Identity, Scale
from spatialdata.utils import unpad_raster


class TestElementsTransform:
    @pytest.mark.skip("Waiting for the new points implementation")
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
        points.points["points_0"] = _set_transform(points.points["points_0"], transform)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert _get_transform(new_sdata.points["points_0"]) == transform

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
        _set_transform(shapes.shapes["shapes_0"], transform)
        shapes.write(tmpdir)
        SpatialData.read(tmpdir)
        assert _get_transform(shapes.shapes["shapes_0"]) == transform

    @pytest.mark.skip("Coordinate systems not yet ported to the new transformation implementation")
    def test_coordinate_systems(self, shapes: SpatialData) -> None:
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        shapes.shapes["shapes_0"] = _set_transform(shapes.shapes["shapes_0"], ct)
        assert list(shapes.coordinate_systems.keys()) == ["cyx", "test"]

    @pytest.mark.skip("Coordinate systems not yet ported to the new transformation implementation")
    def test_physical_units(self, tmp_path: str, shapes: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        shapes.shapes["shapes_0"] = _set_transform(shapes.shapes["shapes_0"], ct)
        shapes.write(tmpdir)
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
    padded = affine.inverse().transform(affine.transform(sdata))
    _unpad_rasters(padded)
    # raise NotImplementedError("TODO: plot the images")
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_image_spatial_multiscale_spatial_image(images: SpatialData):
    sdata = SpatialData(images={k: v for k, v in images.images.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    padded = affine.inverse().transform(affine.transform(sdata))
    _unpad_rasters(padded)
    # TODO: unpad the image
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, SpatialImage)})
    affine = _get_affine()
    padded = affine.inverse().transform(affine.transform(sdata))
    _unpad_rasters(padded)
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


def test_transform_labels_spatial_multiscale_spatial_image(labels: SpatialData):
    sdata = SpatialData(labels={k: v for k, v in labels.labels.items() if isinstance(v, MultiscaleSpatialImage)})
    affine = _get_affine()
    padded = affine.inverse().transform(affine.transform(sdata))
    _unpad_rasters(padded)
    # TODO: unpad the labels
    # raise NotImplementedError("TODO: compare the transformed images with the original ones")


# TODO: maybe add methods for comparing the coordinates of elements so the below code gets less verbose
@pytest.mark.skip("waiting for the new points implementation")
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


def test_map_coordinate_systems(full_sdata):
    scale = Scale([2], axes=("x",))
    im = full_sdata.images["image2d_multiscale"]
    la = full_sdata.labels["labels2d"]
    full_sdata.set_transformation(im, scale)
    assert (
        full_sdata.map_coordinate_systems(source_coordinate_system="global", target_coordinate_system="global")
        == Identity()
    )
    t0 = full_sdata.map_coordinate_systems(source_coordinate_system=im, target_coordinate_system="global")
    t1 = full_sdata.map_coordinate_systems(source_coordinate_system="global", target_coordinate_system=im)
    t2 = full_sdata.map_coordinate_systems(source_coordinate_system=im, target_coordinate_system=la)

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
                [0.5, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )
