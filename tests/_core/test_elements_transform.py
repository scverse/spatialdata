from pathlib import Path

import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transformations import Scale
from tests._core.conftest import xy_cs


class TestElementsTransform:
    @pytest.mark.parametrize("transform", [Scale(np.array([1, 2, 3])), Scale(np.array([2]))])
    def test_points(
        self,
        tmp_path: str,
        points: SpatialData,
        transform: Scale,
        input: CoordinateSystem = xy_cs,
        output: CoordinateSystem = xy_cs,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        transform.input_coordinate_system = input
        transform.output_coordinate_system = output
        points.points["points_0"].uns["transform"] = transform
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.points["points_0"].uns["transform"] == transform

    @pytest.mark.parametrize("transform", [Scale(np.array([1, 2, 3])), Scale(np.array([2]))])
    def test_shapes(
        self,
        tmp_path: str,
        shapes: SpatialData,
        transform: Scale,
        input: CoordinateSystem = xy_cs,
        output: CoordinateSystem = xy_cs,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        transform.input_coordinate_system = input
        transform.output_coordinate_system = output
        shapes.shapes["shapes_0"].uns["transform"] = transform
        shapes.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.shapes["shapes_0"].uns["transform"] == transform
