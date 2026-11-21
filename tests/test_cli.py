from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from click.testing import CliRunner

from spatialdata import SpatialData
from spatialdata.__main__ import cli
from spatialdata.models import Image2DModel


@pytest.mark.parametrize("selection", [[], ["tables"], ["images", "tables"]])
def test_peek_tables(tmp_path: Path, selection: list[str]) -> None:
    path = tmp_path / "data.zarr"
    image = Image2DModel.parse(np.zeros((1, 4, 4)), dims=("c", "y", "x"))
    SpatialData(images={"test_image": image}, tables={"test_table": AnnData(np.ones((2, 3)))}).write(path)

    result = CliRunner().invoke(cli, ["peek", str(path), *selection])

    assert result.exit_code == 0, result.output
    loaded_elements = result.output.split("with the following elements in the Zarr store")[0]
    assert "test_table" in loaded_elements
    assert ("test_image" in loaded_elements) == (not selection or "images" in selection)
