import itertools
import os
import tempfile

import dask.dataframe as dd
import dask_image.ndinterp
import xarray
import xarray.testing
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import read_zarr
from spatialdata._core.models import get_schema
from spatialdata._core.transformations import Affine
from spatialdata.utils import (
    get_backing_files,
    multiscale_spatial_image_from_data_tree,
    unpad_raster,
)


def _pad_raster(data: DataArray, axes: tuple[str, ...]) -> DataArray:
    new_shape = tuple([data.shape[i] * (2 if axes[i] != "c" else 1) for i in range(len(data.shape))])
    x = data.shape[axes.index("x")]
    y = data.shape[axes.index("y")]
    affine = Affine(
        [
            [1, 0, -x / 2.0],
            [0, 1, -y / 2.0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    matrix = affine.to_affine_matrix(input_axes=axes, output_axes=axes)
    transformed = dask_image.ndinterp.affine_transform(data, matrix, output_shape=new_shape)
    return transformed


def test_unpad_raster(images, labels) -> None:
    for raster in itertools.chain(images.images.values(), labels.labels.values()):
        schema = get_schema(raster)
        if isinstance(raster, SpatialImage):
            data = raster
        elif isinstance(raster, MultiscaleSpatialImage):
            d = dict(raster["scale0"])
            assert len(d) == 1
            data = d.values().__iter__().__next__()
        else:
            raise ValueError(f"Unknown type: {type(raster)}")
        padded = _pad_raster(data.data, data.dims)
        if isinstance(raster, SpatialImage):
            padded = schema.parse(padded, dims=data.dims)
        elif isinstance(raster, MultiscaleSpatialImage):
            # some arbitrary scaling factors
            padded = schema.parse(padded, dims=data.dims, multiscale_factors=[2, 2])
        else:
            raise ValueError(f"Unknown type: {type(raster)}")
        unpadded = unpad_raster(padded)
        if isinstance(raster, SpatialImage):
            xarray.testing.assert_equal(raster, unpadded)
        elif isinstance(raster, MultiscaleSpatialImage):
            d0 = dict(raster["scale0"])
            assert len(d0) == 1
            d1 = dict(unpadded["scale0"])
            assert len(d1) == 1
            xarray.testing.assert_equal(d0.values().__iter__().__next__(), d1.values().__iter__().__next__())
        else:
            raise ValueError(f"Unknown type: {type(raster)}")


def test_backing_files_points(points):
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "points0.zarr")
        f1 = os.path.join(tmp_dir, "points1.zarr")
        points.write(f0)
        points.write(f1)
        points0 = read_zarr(f0)
        points1 = read_zarr(f1)
        p0 = points0.points["points_0"]
        p1 = points1.points["points_0"]
        p2 = dd.concat([p0, p1], axis=0)
        files = get_backing_files(p2)
        expected_zarr_locations = [os.path.join(f, "points/points_0/points.parquet") for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)


def test_backing_files_images(images):
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "images0.zarr")
        f1 = os.path.join(tmp_dir, "images1.zarr")
        images.write(f0)
        images.write(f1)
        images0 = read_zarr(f0)
        images1 = read_zarr(f1)

        # single scale
        im0 = images0.images["image2d"]
        im1 = images1.images["image2d"]
        im2 = im0 + im1
        files = get_backing_files(im2)
        expected_zarr_locations = [os.path.join(f, "images/image2d") for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # multiscale
        im3 = images0.images["image2d_multiscale"]
        im4 = images1.images["image2d_multiscale"]
        im5 = multiscale_spatial_image_from_data_tree(im3 + im4)
        files = get_backing_files(im5)
        expected_zarr_locations = [os.path.join(f, "images/image2d_multiscale") for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)


# TODO: this function here below is very similar to the above, unify the test with the above or delete this todo
def test_backing_files_labels(labels):
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "labels0.zarr")
        f1 = os.path.join(tmp_dir, "labels1.zarr")
        labels.write(f0)
        labels.write(f1)
        labels0 = read_zarr(f0)
        labels1 = read_zarr(f1)

        # single scale
        im0 = labels0.labels["labels2d"]
        im1 = labels1.labels["labels2d"]
        im2 = im0 + im1
        files = get_backing_files(im2)
        expected_zarr_locations = [os.path.join(f, "labels/labels2d") for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # multiscale
        im3 = labels0.labels["labels2d_multiscale"]
        im4 = labels1.labels["labels2d_multiscale"]
        im5 = multiscale_spatial_image_from_data_tree(im3 + im4)
        files = get_backing_files(im5)
        expected_zarr_locations = [os.path.join(f, "labels/labels2d_multiscale") for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)
