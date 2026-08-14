from __future__ import annotations

from pathlib import Path

import pooch
import pytest

from spatialdata import SpatialData
from spatialdata.datasets import _cache_dir, _shipped_registry, blobs, cells, raccoon


def test_datasets() -> None:
    extra_cs = "test"
    sdata_blobs = blobs(extra_coord_system=extra_cs)

    assert len(sdata_blobs["table"]) == 26
    assert len(sdata_blobs.shapes["blobs_circles"]) == 5
    assert len(sdata_blobs.shapes["blobs_polygons"]) == 5
    assert len(sdata_blobs.shapes["blobs_multipolygons"]) == 2
    assert len(sdata_blobs.points["blobs_points"].compute()) == 200
    assert sdata_blobs.images["blobs_image"].shape == (3, 512, 512)
    assert len(sdata_blobs.images["blobs_multiscale_image"]) == 3
    assert sdata_blobs.labels["blobs_labels"].shape == (512, 512)
    assert len(sdata_blobs.labels["blobs_multiscale_labels"]) == 3
    assert extra_cs in sdata_blobs.coordinate_systems
    # this catches this bug: https://github.com/scverse/spatialdata/issues/269
    _ = str(sdata_blobs)

    sdata_raccoon = raccoon()
    assert "table" not in sdata_raccoon.tables
    assert len(sdata_raccoon.shapes["circles"]) == 4
    assert sdata_raccoon.images["raccoon"].shape == (3, 768, 1024)
    assert sdata_raccoon.labels["segmentation"].shape == (768, 1024)
    _ = str(sdata_raccoon)


def test_cells_registry() -> None:
    # Network-free: the shipped registry parses and exposes the cells dataset.
    base_url, datasets = _shipped_registry()

    assert base_url == "https://exampledata.scverse.org/spatialdata/"
    entry = datasets["cells"]
    assert entry.type == "spatialdata"
    file = entry.file(name="cells.zip")
    assert file.sha256 == "dc9613cb9e16fd2cd8d83f3a9586eeda4af5ba8ba366f1066efb51305820c5fb"
    assert file.resolve_url(base_url) == "https://exampledata.scverse.org/spatialdata/cells.zip"


def test_cache_dir() -> None:
    # Network-free: both branches of the cache-directory resolution.
    assert _cache_dir("/tmp/example") == Path("/tmp/example")
    assert _cache_dir(None) == Path(pooch.os_cache("spatialdata"))


@pytest.mark.slow
def test_cells_download(tmp_path) -> None:
    # Downloads ~3 MB from the scverse example data bucket; opt out with `-m "not slow"`.
    sdata = cells(path=str(tmp_path))
    assert isinstance(sdata, SpatialData)

    assert set(sdata.images) == {"he_aligned", "he_image", "morphology_focus"}
    assert sdata.images["he_aligned"]["scale0"]["image"].shape == (3, 430, 540)
    assert sdata.images["he_image"]["scale0"]["image"].shape == (3, 423, 339)
    assert sdata.images["morphology_focus"]["scale0"]["image"].shape == (4, 430, 540)

    assert set(sdata.labels) == {"cell_labels", "nucleus_labels", "tissue_labels"}
    assert sdata.labels["cell_labels"]["scale0"]["image"].shape == (430, 540)

    assert len(sdata.shapes["cell_boundaries"]) == 94
    assert len(sdata.shapes["nucleus_boundaries"]) == 94
    assert len(sdata.points["transcripts"].compute()) == 19479
    assert sdata.tables["table"].shape == (94, 5101)
