# type: ignore
"""Benchmarks for ImageTilesDataset: init time and iteration time."""

from __future__ import annotations

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import spatialdata as sd
from spatialdata.dataloader import ImageTilesDataset
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

_RNG = np.random.default_rng(42)

_IMAGE_SIZE = 2048
_N_CIRCLES = 500
_N_CHANNELS = 3

_DATASET_KWARGS = {
    "regions_to_images": {"circles": "image"},
    "regions_to_coordinate_systems": {"circles": "global"},
    "table_name": "table",
    "return_annotations": "instance_id",
}


def _make_sdata() -> sd.SpatialData:
    img_data = _RNG.integers(0, 256, size=(_N_CHANNELS, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.uint8).astype(np.float32)
    image = Image2DModel.parse(img_data, dims=["c", "y", "x"], transformations={"global": Identity()})

    radius = 32.0
    cx = _RNG.uniform(radius, _IMAGE_SIZE - radius, size=_N_CIRCLES)
    cy = _RNG.uniform(radius, _IMAGE_SIZE - radius, size=_N_CIRCLES)
    geom = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in zip(cx, cy, strict=True)]})
    geom["radius"] = radius
    circles = ShapesModel.parse(geom, transformations={"global": Identity()})

    table = ad.AnnData(
        _RNG.random((_N_CIRCLES, 10)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "region": pd.Categorical(["circles"] * _N_CIRCLES),
                "instance_id": np.arange(_N_CIRCLES, dtype=np.int64),
            },
            index=[str(i) for i in range(_N_CIRCLES)],
        ),
    )
    table = TableModel.parse(table, region="circles", region_key="region", instance_key="instance_id")

    return sd.SpatialData(images={"image": image}, shapes={"circles": circles}, tables={"table": table})


class TimeDataloader:
    """Time ImageTilesDataset construction and tile iteration."""

    def setup(self):
        self.sdata = _make_sdata()
        self.ds = ImageTilesDataset(sdata=self.sdata, **_DATASET_KWARGS)

    def teardown(self):
        del self.ds
        del self.sdata

    def time_init(self):
        """Time constructing ImageTilesDataset (bounding-box pre-computation)."""
        ImageTilesDataset(sdata=self.sdata, **_DATASET_KWARGS)

    def time_fetch(self):
        """Time iterating over every tile once."""
        for i in range(len(self.ds)):
            _ = self.ds[i]
