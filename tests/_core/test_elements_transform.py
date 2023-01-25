from pathlib import Path

import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.core_utils import get_transform, set_transform
from spatialdata._core.ngff.ngff_coordinate_system import NgffAxis, NgffCoordinateSystem
from spatialdata._core.ngff.ngff_transformations import NgffScale
from tests._core.conftest import xy_cs


class TestElementsTransform:
    @pytest.mark.skip("Waiting for the new points implementation")
    @pytest.mark.parametrize("transform", [NgffScale(np.array([1, 2, 3])), NgffScale(np.array([2]))])
    def test_points(
        self,
        tmp_path: str,
        points: SpatialData,
        transform: NgffScale,
        input: NgffCoordinateSystem = xy_cs,
        output: NgffCoordinateSystem = xy_cs,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        transform.input_coordinate_system = input
        transform.output_coordinate_system = output
        points.points["points_0"] = set_transform(points.points["points_0"], transform)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert get_transform(new_sdata.points["points_0"]) == transform

    @pytest.mark.parametrize("transform", [NgffScale(np.array([1, 2, 3])), NgffScale(np.array([2]))])
    def test_shapes(
        self,
        tmp_path: str,
        shapes: SpatialData,
        transform: NgffScale,
        input: NgffCoordinateSystem = xy_cs,
        output: NgffCoordinateSystem = xy_cs,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        transform.input_coordinate_system = input
        transform.output_coordinate_system = output
        set_transform(shapes.shapes["shapes_0"], transform)
        shapes.write(tmpdir)
        SpatialData.read(tmpdir)
        assert get_transform(shapes.shapes["shapes_0"]) == transform

    def test_coordinate_systems(self, points: SpatialData) -> None:
        ct = NgffScale(np.array([1, 2, 3]))
        ct.input_coordinate_system = xy_cs
        ct.output_coordinate_system = NgffCoordinateSystem(name="test", axes=[NgffAxis(name="c", type="channel")])
        points.points["points_0_1"] = set_transform(points.points["points_0_1"], ct)
        assert list(points.coordinate_systems.keys()) == ["cyx", "test"]

    def test_physical_units(self, tmp_path: str, points: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        ct = NgffScale(np.array([1, 2, 3]))
        ct.input_coordinate_system = xy_cs
        ct.output_coordinate_system = NgffCoordinateSystem(
            name="test", axes=[NgffAxis(name="x", type="space", unit="micrometers")]
        )
        points.points["points_0_1"] = set_transform(points.points["points_0_1"], ct)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.coordinate_systems["test"]._axes[0].unit == "micrometers"
