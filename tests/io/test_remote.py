import pytest
import zarr

from spatialdata import SpatialData


class TestRemote:
    # Test actual remote datasets from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/datasets/README.html
    # TODO: mark these tests so they are disabled by default because they require internet access
    @pytest.fixture(params=["merfish", "mibitof", "mibitof_alt"])
    def s3_address(self, request):
        urls = {
            "merfish": "https://s3.embl.de/spatialdata/spatialdata-sandbox/merfish.zarr/",
            "mibitof": "https://s3.embl.de/spatialdata/spatialdata-sandbox/mibitof.zarr/",
            "mibitof_alt": "https://dl01.irc.ugent.be/spatial/mibitof/data.zarr/",
        }
        return urls[request.param]

    # TODO: does not work for EMBL datasets, problem with opening version 0.6-dev
    @pytest.mark.skip(reason="Problem with ome_zarr on test datasets: ValueError: Version 0.6-dev not recognized")
    def test_remote(self, s3_address):
        # TODO: remove selection once support for points, shapes and tables is added
        sdata = SpatialData.read(s3_address, selection=("images", "labels"))
        assert len(list(sdata.gen_elements())) > 0

    @pytest.mark.skip(reason="Problem with ome_zarr on test datasets: ValueError: Version 0.6-dev not recognized")
    def test_remote_consolidated(self, s3_address):
        # TODO: find a way to check if the consolidated metadata is actually used. Probably best to wait for Zarr v3.
        root = zarr.open_consolidated(s3_address, mode="r", metadata_key="zmetadata")
        # TODO: remove selection once support for points, shapes and tables is added
        sdata = SpatialData.read(root, selection=("images", "labels"))
        assert len(list(sdata.gen_elements())) > 0
