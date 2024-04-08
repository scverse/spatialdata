import tempfile
from pathlib import Path

import pytest
from spatialdata import read_zarr
from spatialdata._io.format import ShapesFormatV01, ShapesFormatV02
from spatialdata.testing import assert_spatial_data_objects_are_identical


def test_shapes_v1_to_v2(shapes):
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "data.zarr"

        # write shapes in version 1
        shapes.write(f, format=ShapesFormatV01())

        # read shapes in version 1 using the latest version, will fail
        with pytest.raises(ValueError):
            read_zarr(f)

        # read using the legacy reader
        shapes_read = read_zarr(f, format=ShapesFormatV01())

        # overwrite shapes using the latest version
        shapes_read.write(f, format=ShapesFormatV02(), overwrite=True)

        # read shapes using the latest version
        shapes_read = read_zarr(f)

        assert_spatial_data_objects_are_identical(shapes, shapes_read)
