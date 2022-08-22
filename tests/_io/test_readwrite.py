from pathlib import Path

from spatialdata import SpatialData
from spatialdata._core.elements import Image
from spatialdata.utils import are_directories_identical


def test_readwrite_roundtrip(sdata: SpatialData, tmp_path: str):

    tmpdir = Path(tmp_path) / "tmp.zarr"
    sdata.write(tmpdir)
    sdata2 = SpatialData.read(tmpdir)

    assert sdata.table.shape == sdata2.table.shape
    assert sdata.images.keys() == sdata2.images.keys()
    for k in sdata.images.keys():
        assert sdata.images[k].shape == sdata2.images[k].shape
        assert isinstance(sdata.images[k], Image) == isinstance(sdata2.images[k], Image)
    assert list(sdata.labels.keys()) == list(sdata2.labels.keys())

    tmpdir2 = Path(tmp_path) / "tmp2.zarr"
    sdata2.write(tmpdir2)
    assert are_directories_identical(tmpdir, tmpdir2)
