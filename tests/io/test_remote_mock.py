import os
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import fsspec
import pytest
from upath import UPath
from upath.implementations.cloud import S3Path

from spatialdata import SpatialData
from spatialdata.testing import assert_spatial_data_objects_are_identical

# This mock setup was inspired by https://github.com/fsspec/universal_pathlib/blob/main/upath/tests/conftest.py


@pytest.fixture(scope="session")
def s3_server():
    # create a writable local S3 system via moto
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


def clear_s3(s3_server, location=None):
    # clear an s3 bucket of all contents
    anon, s3so = s3_server
    s3 = fsspec.filesystem("s3", anon=anon, **s3so)
    if location and s3.exists(location):
        for d, _, keys in s3.walk(location):
            for key in keys:
                s3.rm(f"{d}/{key}")
    s3.invalidate_cache()


def upload_to_upath(upath, sdata):
    # write the object to disk via a regular path, then copy it to the UPath byte-by-byte
    # useful for testing the read and write functionality separately
    with tempfile.TemporaryDirectory() as tempdir:
        sdata_path = Path(tempdir) / "temp.zarr"
        sdata.write(sdata_path)
        # for every file in the sdata_path, copy it to the upath
        for x in sdata_path.glob("**/*"):
            if x.is_file():
                data = x.read_bytes()
                destination = upath / x.relative_to(sdata_path)
                destination.write_bytes(data)


@pytest.fixture(scope="function")
def s3_fixture(s3_server):
    # make a mock bucket available for testing
    pytest.importorskip("s3fs")
    anon, s3so = s3_server
    s3 = fsspec.filesystem("s3", anon=anon, **s3so)
    random_name = uuid.uuid4().hex
    bucket_name = f"test_{random_name}"
    clear_s3(s3_server, bucket_name)
    s3.mkdir(bucket_name)
    # here you could write existing test files to s3.upload if needed
    s3.invalidate_cache()
    yield f"s3://{bucket_name}", anon, s3so


class TestRemoteMock:
    @pytest.fixture(scope="function")
    def upath(self, s3_fixture):
        # create a UPath object for the mock s3 bucket
        path, anon, s3so = s3_fixture
        return UPath(path, anon=anon, **s3so)

    def test_is_S3Path(self, upath):
        assert isinstance(upath, S3Path)

    def test_upload_sdata(self, upath, full_sdata):
        tmpdir = upath / "tmp.zarr"
        upload_to_upath(tmpdir, full_sdata)
        assert tmpdir.exists()
        assert len(list(tmpdir.glob("*"))) == 8

    def test_creating_file(self, upath) -> None:
        file_name = "file1"
        p1 = upath / file_name
        p1.touch()
        contents = [p.name for p in upath.iterdir()]
        assert file_name in contents

    @pytest.mark.parametrize(
        "sdata_type",
        [
            "images",
            "labels",
            "points",
            "shapes",
            "table_single_annotation",
            "table_multiple_annotations",
        ],
    )
    def test_reading_mocked_elements(self, upath: UPath, sdata_type: str, request) -> None:
        sdata = request.getfixturevalue(sdata_type)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "tmp.zarr"
            sdata.write(local_path)
            local_sdata = SpatialData.read(local_path)
            local_len = len(list(local_sdata.gen_elements()))
            assert local_len > 0
            remote_path = upath / "tmp.zarr"
            upload_to_upath(remote_path, sdata)
            remote_sdata = SpatialData.read(remote_path)
            assert len(list(remote_sdata.gen_elements())) == local_len
            assert_spatial_data_objects_are_identical(local_sdata, remote_sdata)

    @pytest.mark.parametrize(
        "sdata_type",
        [
            # TODO: fix remote writing support images
            # "images",
            # TODO: fix remote writing support labels
            # "labels",
            "points",
            "shapes",
            "table_single_annotation",
            "table_multiple_annotations",
        ],
    )
    def test_writing_mocked_elements(self, upath: UPath, sdata_type: str, request) -> None:
        local_sdata = request.getfixturevalue(sdata_type)
        local_len = len(list(local_sdata.gen_elements()))
        assert local_len > 0
        # test writing to a remote path
        remote_path = upath / "tmp.zarr"
        local_sdata.write(remote_path)
        # test reading the remotely written object
        remote_sdata = SpatialData.read(remote_path)
        assert len(list(remote_sdata.gen_elements())) == local_len
        assert_spatial_data_objects_are_identical(local_sdata, remote_sdata)
