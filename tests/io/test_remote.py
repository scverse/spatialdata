import pytest
import zarr

from spatialdata import SpatialData


class TestRemote:
    # Test actual remote datasets from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/datasets/README.html
    # These tests are disabled by default because they require internet access
    @pytest.fixture(params=["merfish", "mibitof"])
    def s3_address(self, request):
        urls = {
            "merfish": "https://s3.embl.de/spatialdata/spatialdata-sandbox/merfish.zarr/",
            "mibitof": "https://s3.embl.de/spatialdata/spatialdata-sandbox/mibitof.zarr/",
        }
        return urls[request.param]

    # TODO: does not work, problem with opening remote parquet
    @pytest.mark.xfail(reason="Problem with opening remote parquet")
    def test_remote(self, s3_address):
        root = zarr.open_consolidated(s3_address, mode="r", metadata_key="zmetadata")
        sdata = SpatialData.read(root)
        assert len(list(sdata.gen_elements())) > 0
