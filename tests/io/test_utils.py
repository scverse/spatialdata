import os
import pickle
import tempfile

import dask.dataframe as dd
import pytest
from spatialdata import read_zarr
from spatialdata._io._utils import _search_for_backing_files_recursively, get_dask_backing_files


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
        expected_zarr_locations_legacy = [
            os.path.realpath(os.path.join(f, "points/points_0/points.parquet")) for f in [f0, f1]
        ]
        expected_zarr_locations_new = [
            os.path.realpath(os.path.join(f, "points/points_0/points.parquet/part.0.parquet")) for f in [f0, f1]
        ]
        assert set(files) == set(expected_zarr_locations_legacy) or set(files) == set(expected_zarr_locations_new)

        # to prepare the test data to commit to the repository
        # to generate with dask-expr disabled
        # with open('data/read-parquet_piece.pkl', 'wb') as outfile:
        #     pickle.dump(p2.dask.to_dict(), outfile)
        # to generate with dask-expr enabled
        # with open('data/read_parquet_piece.pkl', 'wb') as outfile:
        #     pickle.dump(p2.dask, outfile)


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
        im5 = im3 + im4
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
        im5 = im3 + im4
        files = get_dask_backing_files(im5)
        expected_zarr_locations = [os.path.realpath(os.path.join(f, "labels/labels2d_multiscale")) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # to prepare the test data to commit to the repository
        # with open('data/original-from-zarr.pkl', 'wb') as outfile:
        #     graph = im5['scale0']['image'].data.dask
        #     pickle.dump(graph.to_dict(), outfile)


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
        expected_zarr_locations_old = [
            os.path.realpath(os.path.join(f0, "points/points_0/points.parquet")),
            os.path.realpath(os.path.join(f1, "images/image2d")),
        ]
        expected_zarr_locations_new = [
            os.path.realpath(os.path.join(f0, "points/points_0/points.parquet/part.0.parquet")),
            os.path.realpath(os.path.join(f1, "images/image2d")),
        ]
        assert set(files) == set(expected_zarr_locations_old) or set(files) == set(expected_zarr_locations_new)


@pytest.mark.parametrize(
    "dask_graph_file", ["data/read-parquet_piece.pkl", "data/read_parquet_piece.pkl", "data/original-from-zarr.pkl"]
)
def test_search_for_backing_files_recursively(dask_graph_file: str):
    with open(dask_graph_file, "rb") as infile:
        dask_graph = pickle.load(infile)
        files = []
        _search_for_backing_files_recursively(dask_graph, files)
        assert len(files) == 2
