from pathlib import Path

from spatialdata import SpatialData
from spatialdata._core.elements import Image
from spatialdata.utils import are_directories_identical


# either use parametrize here, either use params in the sdata fixture in conftest.py
# @pytest.mark.parametrize("sdata", "empty_points")
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
