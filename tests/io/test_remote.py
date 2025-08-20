import pytest
import zarr
from upath import UPath

from spatialdata import SpatialData


class TestRemote:
    # Test actual remote datasets from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/datasets/README.html

    @pytest.fixture(params=["merfish", "mibitof", "mibitof_alt"])
    def s3_address(self, request):
        urls = {
            "merfish": UPath(
                "s3://spatialdata/spatialdata-sandbox/merfish.zarr", endpoint_url="https://s3.embl.de", anon=True
            ),
            "mibitof": UPath(
                "s3://spatialdata/spatialdata-sandbox/mibitof.zarr", endpoint_url="https://s3.embl.de", anon=True
            ),
            "mibitof_alt": "https://dl01.irc.ugent.be/spatial/mibitof/data.zarr/",
        }
        return urls[request.param]

    def test_remote(self, s3_address):
        # TODO: remove selection once support for points, shapes and tables is added
        sdata = SpatialData.read(s3_address, selection=("images", "labels"))
        assert len(list(sdata.gen_elements())) > 0

    def test_remote_consolidated(self, s3_address):
        urlpath, storage_options = str(s3_address), getattr(s3_address, "storage_options", {})
        root = zarr.open_consolidated(urlpath, mode="r", metadata_key="zmetadata", storage_options=storage_options)
        sdata = SpatialData.read(root, selection=("images", "labels"))
        assert len(list(sdata.gen_elements())) > 0
