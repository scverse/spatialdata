import tempfile
from pathlib import Path

from spatialdata import read_zarr
from spatialdata._io.format import ShapesFormatV01, ShapesFormatV02
from spatialdata.testing import assert_spatial_data_objects_are_identical


def test_shapes_v1_to_v2(shapes):
    with tempfile.TemporaryDirectory() as tmpdir:
        f0 = Path(tmpdir) / "data0.zarr"
        f1 = Path(tmpdir) / "data1.zarr"

        # write shapes in version 1
        shapes.write(f0, format=ShapesFormatV01())

        # reading from v1 works
        shapes_read = read_zarr(f0)

        assert_spatial_data_objects_are_identical(shapes, shapes_read)

        # write shapes using the v2 version
        shapes_read.write(f1, format=ShapesFormatV02())

        # read again
        shapes_read = read_zarr(f1)

        assert_spatial_data_objects_are_identical(shapes, shapes_read)
