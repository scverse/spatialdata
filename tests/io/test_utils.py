import os
import tempfile

import dask.dataframe as dd
import numpy as np
import pytest
from spatialdata import read_zarr, save_transformations
from spatialdata._io._utils import get_dask_backing_files
from spatialdata._utils import multiscale_spatial_image_from_data_tree
from spatialdata.transformations import Scale, get_transformation, set_transformation


def test_backing_files_points(points):
    """Test the ability to identify the backing files of a dask dataframe from examining its computational graph"""
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
        files = get_dask_backing_files(p2)
        expected_zarr_locations = [
            os.path.realpath(os.path.join(f, "points/points_0/points.parquet")) for f in [f0, f1]
        ]
        assert set(files) == set(expected_zarr_locations)


def test_backing_files_images(images):
    """
    Test the ability to identify the backing files of single scale and multiscale images from examining their
    computational graph
    """
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
        files = get_dask_backing_files(im2)
        expected_zarr_locations = [os.path.realpath(os.path.join(f, "images/image2d")) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # multiscale
        im3 = images0.images["image2d_multiscale"]
        im4 = images1.images["image2d_multiscale"]
        im5 = multiscale_spatial_image_from_data_tree(im3 + im4)
        files = get_dask_backing_files(im5)
        expected_zarr_locations = [os.path.realpath(os.path.join(f, "images/image2d_multiscale")) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)


# TODO: this function here below is very similar to the above, unify the test with the above or delete this todo
def test_backing_files_labels(labels):
    """
    Test the ability to identify the backing files of single scale and multiscale labels from examining their
    computational graph
    """
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
        files = get_dask_backing_files(im2)
        expected_zarr_locations = [os.path.realpath(os.path.join(f, "labels/labels2d")) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # multiscale
        im3 = labels0.labels["labels2d_multiscale"]
        im4 = labels1.labels["labels2d_multiscale"]
        im5 = multiscale_spatial_image_from_data_tree(im3 + im4)
        files = get_dask_backing_files(im5)
        expected_zarr_locations = [os.path.realpath(os.path.join(f, "labels/labels2d_multiscale")) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)


def test_backing_files_combining_points_and_images(points, images):
    """
    Test the ability to identify the backing files of an object that depends both on dask dataframes and dask arrays
    from examining its computational graph
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "points0.zarr")
        f1 = os.path.join(tmp_dir, "images1.zarr")
        points.write(f0)
        images.write(f1)
        points0 = read_zarr(f0)
        images1 = read_zarr(f1)

        p0 = points0.points["points_0"]
        im1 = images1.images["image2d"]
        v = p0["x"].loc[0].values
        v.compute_chunk_sizes()
        im2 = v + im1
        files = get_dask_backing_files(im2)
        expected_zarr_locations = [
            os.path.realpath(os.path.join(f0, "points/points_0/points.parquet")),
            os.path.realpath(os.path.join(f1, "images/image2d")),
        ]
        assert set(files) == set(expected_zarr_locations)


def test_save_transformations(labels):
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "labels0.zarr")
        scale = Scale([2, 2], axes=("x", "y"))
        set_transformation(labels.labels["labels2d"], scale)
        with pytest.raises(ValueError):
            save_transformations(labels)
        labels.write(f0)
        save_transformations(labels)
        labels0 = read_zarr(f0)
        scale0 = get_transformation(labels0.labels["labels2d"])
        assert isinstance(scale0, Scale)
        assert np.array_equal(scale.scale, scale0.scale)
