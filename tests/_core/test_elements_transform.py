from pathlib import Path

import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.core_utils import _get_transform, _set_transform
from spatialdata._core.transformations import Scale


class TestElementsTransform:
    @pytest.mark.skip("Waiting for the new points implementation")
    @pytest.mark.parametrize(
        "transform", [Scale(np.array([1, 2, 3]), axes=("x", "y", "z")), Scale(np.array([2]), axes=("x",))]
    )
    def test_points(
        self,
        tmp_path: str,
        points: SpatialData,
        transform: Scale,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        points.points["points_0"] = _set_transform(points.points["points_0"], transform)
        points.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert _get_transform(new_sdata.points["points_0"]) == transform

    @pytest.mark.parametrize(
        "transform", [Scale(np.array([1, 2, 3]), axes=("x", "y", "z")), Scale(np.array([2]), axes=("x",))]
    )
    def test_shapes(
        self,
        tmp_path: str,
        shapes: SpatialData,
        transform: Scale,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        _set_transform(shapes.shapes["shapes_0"], transform)
        shapes.write(tmpdir)
        SpatialData.read(tmpdir)
        assert _get_transform(shapes.shapes["shapes_0"]) == transform

    @pytest.mark.skip("Coordinate systems not yet ported to the new transformation implementation")
    def test_coordinate_systems(self, shapes: SpatialData) -> None:
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        shapes.shapes["shapes_0"] = _set_transform(shapes.shapes["shapes_0"], ct)
        assert list(shapes.coordinate_systems.keys()) == ["cyx", "test"]

    @pytest.mark.skip("Coordinate systems not yet ported to the new transformation implementation")
    def test_physical_units(self, tmp_path: str, shapes: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        ct = Scale(np.array([1, 2, 3]), axes=("x", "y", "z"))
        shapes.shapes["shapes_0"] = _set_transform(shapes.shapes["shapes_0"], ct)
        shapes.write(tmpdir)
        new_sdata = SpatialData.read(tmpdir)
        assert new_sdata.coordinate_systems["test"]._axes[0].unit == "micrometers"
