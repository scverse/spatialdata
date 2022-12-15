from pathlib import Path

import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.coordinate_system import CoordinateSystem, Axis
from spatialdata._core.core_utils import get_transform, set_transform
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
        points.points["points_0"] = set_transform(points.points["points_0"], transform)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert get_transform(new_sdata.points["points_0"]) == transform

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
        set_transform(shapes.shapes["shapes_0"], transform)
        shapes.write(tmpdir)
        SpatialData.read(tmpdir)
        assert get_transform(shapes.shapes["shapes_0"]) == transform

    def test_coordinate_systems(self, points: SpatialData) -> None:
        ct = Scale(np.array([1, 2, 3]))
        ct.input_coordinate_system = xy_cs
        ct.output_coordinate_system = CoordinateSystem(name="test", axes=[Axis(name="c", type="channel")])
        points.points['points_0_1'] = set_transform(points.points['points_0_1'], ct)
        assert list(points.coordinate_systems.keys()) == ['cyx', 'test']

    def test_physical_units(self, tmp_path: str, points: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        ct = Scale(np.array([1, 2, 3]))
        ct.input_coordinate_system = xy_cs
        ct.output_coordinate_system = CoordinateSystem(name="test", axes=[Axis(name="x", type="space",
                                                                               unit='micrometers')])
        points.points['points_0_1'] = set_transform(points.points['points_0_1'], ct)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.coordinate_systems['test']._axes[0].unit == 'micrometers'
