import contextlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata.dataloader import ImageTilesDataset
from spatialdata.models import TableModel


@pytest.mark.parametrize("image_element", ["blobs_image", "blobs_multiscale_image"])
@pytest.mark.parametrize(
    "regions_element",
    ["blobs_labels", "blobs_multiscale_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"],
)
def test_tiles_dataset(sdata_blobs, image_element, regions_element):
    if regions_element in ["blobs_labels", "blobs_multipolygons", "blobs_multiscale_labels"]:
        cm = pytest.raises(NotImplementedError)
    else:
        cm = contextlib.nullcontext()
    with cm:
        ds = ImageTilesDataset(
            sdata=sdata_blobs,
            regions_to_images={regions_element: image_element},
            tile_dim_in_units=10,
            tile_dim_in_pixels=32,
            target_coordinate_system="global",
        )
        tile = ds[0].images.values().__iter__().__next__()
        assert tile.shape == (3, 32, 32)


def test_tiles_table(sdata_blobs):
    new_table = AnnData(
        X=np.random.default_rng().random((3, 10)),
        obs=pd.DataFrame({"region": "blobs_circles", "instance_id": np.array([0, 1, 2])}),
    )
    new_table = TableModel.parse(new_table, region="blobs_circles", region_key="region", instance_key="instance_id")
    del sdata_blobs.table
    sdata_blobs.table = new_table
    ds = ImageTilesDataset(
        sdata=sdata_blobs,
        regions_to_images={"blobs_circles": "blobs_image"},
        tile_dim_in_units=10,
        tile_dim_in_pixels=32,
        target_coordinate_system="global",
    )
    assert len(ds) == 3
    assert len(ds[0].table) == 1
    assert np.all(ds[0].table.X == new_table[0].X)


def test_tiles_multiple_elements(sdata_blobs):
    ds = ImageTilesDataset(
        sdata=sdata_blobs,
        regions_to_images={"blobs_circles": "blobs_image", "blobs_polygons": "blobs_multiscale_image"},
        tile_dim_in_units=10,
        tile_dim_in_pixels=32,
        target_coordinate_system="global",
    )
    assert len(ds) == 6
    _ = ds[0]
