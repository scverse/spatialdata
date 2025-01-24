import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import fsspec
import pytest
from fsspec.implementations.local import LocalFileSystem, make_path_posix
from fsspec.registry import _registry, register_implementation
from fsspec.utils import stringify_path
from upath import UPath
from upath.implementations.cloud import S3Path

from spatialdata import SpatialData
from spatialdata.testing import assert_spatial_data_objects_are_identical


## Mock setup from https://github.com/fsspec/universal_pathlib/blob/main/upath/tests/conftest.py
def posixify(path):
    return str(path).replace("\\", "/")


class DummyTestFS(LocalFileSystem):
    protocol = "mock"
    root_marker = "/"

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith("mock://"):
            path = path[7:]
        elif path.startswith("mock:"):
            path = path[5:]
        return make_path_posix(path).rstrip("/") or cls.root_marker


@pytest.fixture(scope="session")
def clear_registry():
    register_implementation("mock", DummyTestFS)
    try:
        yield
    finally:
        _registry.clear()


@pytest.fixture(scope="session")
def s3_server():
    # writable local S3 system
    if "BOTO_CONFIG" not in os.environ:  # pragma: no cover
        os.environ["BOTO_CONFIG"] = "/dev/null"
    if "AWS_ACCESS_KEY_ID" not in os.environ:  # pragma: no cover
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    if "AWS_SECURITY_TOKEN" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
    if "AWS_SESSION_TOKEN" not in os.environ:  # pragma: no cover
        os.environ["AWS_SESSION_TOKEN"] = "testing"
    if "AWS_DEFAULT_REGION" not in os.environ:  # pragma: no cover
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    requests = pytest.importorskip("requests")

    pytest.importorskip("moto")

    port = 5555
    endpoint_uri = f"http://127.0.0.1:{port}/"
    proc = subprocess.Popen(
        shlex.split(f"moto_server -p {port}"),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    try:
        timeout = 5
        while timeout > 0:
            try:
                r = requests.get(endpoint_uri, timeout=10)
                if r.ok:
                    break
            except requests.exceptions.RequestException:  # pragma: no cover
                pass
            timeout -= 0.1  # pragma: no cover
            time.sleep(0.1)  # pragma: no cover
        anon = False
        s3so = {
            "client_kwargs": {"endpoint_url": endpoint_uri},
            "use_listings_cache": True,
        }
        yield anon, s3so
    finally:
        proc.terminate()
        proc.wait()


@pytest.fixture(scope="function")
def s3_fixture(s3_server):
    pytest.importorskip("s3fs")
    anon, s3so = s3_server
    s3 = fsspec.filesystem("s3", anon=False, **s3so)
    random_name = uuid.uuid4().hex
    bucket_name = f"test_{random_name}"
    if s3.exists(bucket_name):
        for dir, _, keys in s3.walk(bucket_name):
            for key in keys:
                s3.rm(f"{dir}/{key}")
    else:
        s3.mkdir(bucket_name)
    # for x in Path(local_testdir).glob("**/*"):
    #     target_path = f"{bucket_name}/{posixify(x.relative_to(local_testdir))}"
    #     if x.is_file():
    #         s3.upload(str(x), target_path)
    s3.invalidate_cache()
    yield f"s3://{bucket_name}", anon, s3so


@pytest.fixture(scope="session")
def http_server(tmp_path_factory):
    http_tempdir = tmp_path_factory.mktemp("http")

    requests = pytest.importorskip("requests")
    pytest.importorskip("http.server")
    proc = subprocess.Popen(shlex.split(f"python -m http.server --directory {http_tempdir} 8080"))
    try:
        url = "http://127.0.0.1:8080/folder"
        path = Path(http_tempdir) / "folder"
        path.mkdir()
        timeout = 10
        while True:
            try:
                r = requests.get(url, timeout=10)
                if r.ok:
                    yield path, url
                    break
            except requests.exceptions.RequestException as e:  # noqa: E722
                timeout -= 1
                if timeout < 0:
                    raise SystemError from e
                time.sleep(1)
    finally:
        proc.terminate()
        proc.wait()


@pytest.fixture
def http_fixture(local_testdir, http_server):
    http_path, http_url = http_server
    shutil.rmtree(http_path)
    shutil.copytree(local_testdir, http_path)
    yield http_url


class TestRemoteMock:
    @pytest.fixture(scope="function")
    def upath(self, s3_fixture):
        path, anon, s3so = s3_fixture
        return UPath(path, anon=anon, **s3so)

    def test_is_S3Path(self, upath):
        assert isinstance(upath, S3Path)

    # # Test UPath with Moto Mocking
    def test_creating_file(self, upath):
        file_name = "file1"
        p1 = upath / file_name
        p1.touch()
        contents = [p.name for p in upath.iterdir()]
        assert file_name in contents

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for ImageElement not yet implemented")
    def test_images(self, upath: UPath, images: SpatialData) -> None:
        tmpdir = upath / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(images, sdata)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for LabelsElement not yet implemented")
    def test_labels(self, upath: UPath, labels: SpatialData) -> None:
        tmpdir = upath / "tmp.zarr"
        labels.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(labels, sdata)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for ShapesElement not yet implemented")
    def test_shapes(self, upath: UPath, shapes: SpatialData) -> None:
        import numpy as np

        tmpdir = upath / "tmp.zarr"

        # check the index is correctly written and then read
        shapes["circles"].index = np.arange(1, len(shapes["circles"]) + 1)

        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(shapes, sdata)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for PointsElement not yet implemented")
    def test_points(self, upath: UPath, points: SpatialData) -> None:
        import dask.dataframe as dd
        import numpy as np

        tmpdir = upath / "tmp.zarr"

        # check the index is correctly written and then read
        new_index = dd.from_array(np.arange(1, len(points["points_0"]) + 1))
        points["points_0"] = points["points_0"].set_index(new_index)

        points.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(points, sdata)

    def _test_table(self, upath: UPath, table: SpatialData) -> None:
        tmpdir = upath / "tmp.zarr"
        table.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(table, sdata)

    def _test_read_elem(self, upath: UPath, table: SpatialData) -> None:
        tmpdir = upath / "tmp.zarr"
        store = zarr.open()
        table.write(tmpdir)
        # location of table

        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(elem, sdata)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for TableElement not yet implemented")
    def test_single_table_single_annotation(self, upath: UPath, table_single_annotation: SpatialData) -> None:
        self._test_table(upath, table_single_annotation)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for TableElement not yet implemented")
    def test_single_table_multiple_annotations(self, upath: UPath, table_multiple_annotations: SpatialData) -> None:
        self._test_table(upath, table_multiple_annotations)

    # TODO: fix this test
    @pytest.mark.xfail(reason="Fails because remote support for SpatialData not yet implemented")
    def test_full_sdata(self, upath: UPath, full_sdata: SpatialData) -> None:
        tmpdir = upath / "tmp.zarr"
        full_sdata.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(full_sdata, sdata)
