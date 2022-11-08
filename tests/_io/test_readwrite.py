from pathlib import Path

import pytest
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core.elements import Image
from spatialdata.utils import are_directories_identical


class TestReadWrite:
    def test_images(self, tmp_path: str, images: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert images.images.keys() == sdata.images.keys()
        for k1, k2 in zip(images.images.keys(), sdata.images.keys()):
            assert isinstance(sdata.images[k1], SpatialImage)
            assert (images.images[k1].data == sdata.images[k2]).all()
            assert images.images[k1].coords.keys() == sdata.images[k2].coords.keys()

    def test_images_multiscale(self, tmp_path: str, images_multiscale: SpatialData) -> None:
        """Test read/write."""
        images = images_multiscale
        tmpdir = Path(tmp_path) / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert images.images.keys() == sdata.images.keys()
        for k1, k2 in zip(images.images.keys(), sdata.images.keys()):
            assert isinstance(images.images[k1], MultiscaleSpatialImage)
            for kk1, kk2 in zip(sdata.images[k1], sdata.images[k1]):
                assert (images.images[k1][kk1][k1].data == sdata.images[k2][kk2][k2]).all()
                assert images.images[k1][kk1][k1].coords.keys() == sdata.images[k1][kk1][k1].coords.keys()


@pytest.mark.skip("Skip this check for now.")
def test_readwrite_roundtrip(sdata: SpatialData, tmp_path: str):
    print(sdata)

    tmpdir = Path(tmp_path) / "tmp.zarr"
    sdata.write(tmpdir)
    sdata2 = SpatialData.read(tmpdir)

    assert are_directories_identical(tmpdir, tmpdir)
    if sdata.table is not None or sdata2.table is not None:
        assert sdata.table is None and sdata2.table is None or sdata.table.shape == sdata2.table.shape
    assert sdata.images.keys() == sdata2.images.keys()
    for k in sdata.images.keys():
        assert sdata.images[k].shape == sdata2.images[k].shape
        assert isinstance(sdata.images[k], Image) == isinstance(sdata2.images[k], Image)
    assert list(sdata.labels.keys()) == list(sdata2.labels.keys())

    tmpdir2 = Path(tmp_path) / "tmp2.zarr"
    sdata2.write(tmpdir2)
    # install ome-zarr-py from https://github.com/LucaMarconato/ome-zarr-py since this merges some branches with
    # bugfixes (see https://github.com/ome/ome-zarr-py/issues/219#issuecomment-1237263744)
    # also, we exclude the comparison of images that are not full scale in the pyramid representation, as they are
    # different due to a bug ( see discussion in the link above)
    assert are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")
