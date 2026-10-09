from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel


@pytest.fixture
def image_store(tmp_path: Path) -> Path:
    path = tmp_path / "image.zarr"
    image = Image2DModel.parse(np.zeros((1, 4, 4)), dims=("c", "y", "x"))
    SpatialData(images={"image": image}).write(path)
    return path


@pytest.mark.parametrize("reader", [read_zarr, SpatialData.read])
@pytest.mark.parametrize("selection", [("table",), ("imagez",), ("images", "table"), "images"])
def test_invalid_selection(image_store: Path, reader: Any, selection: Any) -> None:
    with pytest.raises(ValueError, match="Invalid selection"):
        reader(image_store, selection=selection)


@pytest.mark.parametrize("reader", [read_zarr, SpatialData.read])
@pytest.mark.parametrize("selection", [None, (), ("images",), ("images", "tables"), ("images", "images")])
def test_valid_selection_keeps_images(image_store: Path, reader: Any, selection: Any) -> None:
    result = reader(image_store, selection=selection)
    assert list(result.images) == ["image"]
    np.testing.assert_array_equal(result.images["image"].values, np.zeros((1, 4, 4)))


@pytest.mark.parametrize("selection", [("labels",), ("points",), ("shapes",), ("tables",)])
def test_valid_selection_of_absent_type(image_store: Path, selection: Any) -> None:
    result = read_zarr(image_store, selection=selection)
    assert list(result.gen_elements()) == []
