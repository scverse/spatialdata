from __future__ import annotations

import os
import tempfile
from contextlib import nullcontext

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from spatialdata import SpatialData, read_zarr
from spatialdata._io._utils import get_dask_backing_files, handle_read_errors, join_fsspec_store_path
from spatialdata.models import PointsModel


@pytest.mark.parametrize(
    ("store_path", "relative_path", "expected"),
    [
        # empty / slash-only relative paths return the store root unchanged
        ("bucket/store.zarr", "", "bucket/store.zarr"),
        ("bucket/store.zarr", "/", "bucket/store.zarr"),
        ("store.zarr", "", "store.zarr"),
        # leading slashes on the relative path are stripped before joining
        ("bucket/store.zarr", "/images/img", "bucket/store.zarr/images/img"),
        ("bucket/store.zarr", "//images/img", "bucket/store.zarr/images/img"),
        # nested and single-segment relative paths
        ("bucket/store.zarr", "images/img", "bucket/store.zarr/images/img"),
        ("bucket/store.zarr", "points", "bucket/store.zarr/points"),
        ("bucket", "key", "bucket/key"),
    ],
)
def test_join_fsspec_store_path(store_path: str, relative_path: str, expected: str) -> None:
    """`join_fsspec_store_path` joins a relative zarr-group path onto an fsspec store root."""
    assert join_fsspec_store_path(store_path, relative_path) == expected


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


def test_backing_files_images(images):
    """
    Test the ability to identify the backing files of single scale and multiscale images from examining their
    computational graph
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = UPath(tmp_dir) / "images0.zarr"
        f1 = UPath(tmp_dir) / "images1.zarr"
        images.write(f0)
        images.write(f1)
        images0 = read_zarr(f0)
        images1 = read_zarr(f1)

        # single scale
        im0 = images0.images["image2d"]
        im1 = images1.images["image2d"]
        im2 = im0 + im1
        files = get_dask_backing_files(im2)
        expected_zarr_locations = [str((f / "images" / "image2d").resolve()) for f in [f0, f1]]
        assert set(files) == set(expected_zarr_locations)

        # multiscale
        im3 = images0.images["image2d_multiscale"]
        im4 = images1.images["image2d_multiscale"]
        im5 = im3 + im4
        files = get_dask_backing_files(im5)
        expected_zarr_locations = [str((f / "images" / "image2d_multiscale").resolve()) for f in [f0, f1]]
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


def test_backing_files_combining_points_and_images(points, images):
    """
    Test the ability to identify the backing files of an object that depends both on dask dataframes and dask arrays
    from examining its computational graph
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = UPath(tmp_dir) / "points0.zarr"
        f1 = UPath(tmp_dir) / "images1.zarr"
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
            str((f0 / "points" / "points_0" / "points.parquet").resolve()),
            str((f1 / "images" / "image2d").resolve()),
        ]
        expected_zarr_locations_new = [
            str((f0 / "points" / "points_0" / "points.parquet" / "part.0.parquet").resolve()),
            str((f1 / "images" / "image2d").resolve()),
        ]
        assert set(files) == set(expected_zarr_locations_old) or set(files) == set(expected_zarr_locations_new)


@pytest.mark.parametrize(
    ("on_bad_files", "actual_error", "expectation"),
    [
        ("error", None, nullcontext()),
        ("error", KeyError("key"), pytest.raises(KeyError)),
        ("warn", None, nullcontext()),
        ("warn", KeyError("key"), pytest.warns(UserWarning, match="location: KeyError")),
        ("warn", RuntimeError("unhandled"), pytest.raises(RuntimeError)),
    ],
)
def test_handle_read_errors(on_bad_files: str, actual_error: Exception, expectation):
    with expectation:  # noqa: SIM117
        with handle_read_errors(on_bad_files=on_bad_files, location="location", exc_types=KeyError):
            if actual_error is not None:
                raise actual_error


def test_repr_points_shows_row_count():
    """repr() must show the concrete row count, not <Delayed>, for backed points."""
    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = os.path.join(tmp, "points.parquet")
        n_rows = 400
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.random(n_rows), "y": rng.random(n_rows)})
        # aggregate_files=True produces a list-of-piece-dicts graph, the case reported in #1084
        dd.from_pandas(df, npartitions=4).to_parquet(parquet_path, write_index=False)
        ddf = dd.read_parquet(parquet_path, aggregate_files=True)

        points = PointsModel.parse(ddf)
        sdata = SpatialData(points={"pts": points})
        sdata.write(os.path.join(tmp, "example.zarr"))

        r = repr(sdata)
        assert f"({n_rows}," in r, f"expected row count {n_rows} in repr, got: {r}"
        assert "<Delayed>" not in r, f"repr still contains <Delayed>: {r}"
