import contextlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.dataloader import ImageTilesDataset
from spatialdata.models import TableModel


class TestImageTilesDataset:
    @pytest.mark.parametrize("image_element", ["blobs_image", "blobs_multiscale_image"])
    @pytest.mark.parametrize(
        "regions_element",
        ["blobs_labels", "blobs_multiscale_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"],
    )
    def test_validation(self, sdata_blobs, image_element, regions_element):
        if regions_element in ["blobs_labels", "blobs_multiscale_labels"] or image_element == "blobs_multiscale_image":
            cm = pytest.raises(NotImplementedError)
        elif regions_element in ["blobs_circles", "blobs_polygons", "blobs_multipolygons"]:
            cm = pytest.raises(ValueError)
        else:
            cm = contextlib.nullcontext()
        with cm:
            _ = ImageTilesDataset(
                sdata=sdata_blobs,
                regions_to_images={regions_element: image_element},
                regions_to_coordinate_systems={regions_element: "global"},
            )

    @pytest.mark.parametrize(
        "regions_element",
        ["blobs_circles", "blobs_polygons", "blobs_multipolygons"],
    )
    def test_default(self, sdata_blobs, image_element, regions_element):
        sdata = self._annotate_shapes(sdata_blobs, regions_element)
        ds = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={regions_element: "blobs_image"},
            regions_to_coordinate_systems={regions_element: "global"},
        )

        tile = ds[0].images.values().__iter__().__next__()
        assert tile.shape == (3, 32, 32)

    # TODO: consider adding this logic to blobs, to generate blobs with arbitrary table annotation
    def _annotate_shapes(self, sdata: SpatialData, shape: str) -> SpatialData:
        new_table = AnnData(
            X=np.random.default_rng().random((len(sdata[shape]), 10)),
            obs=pd.DataFrame({"region": shape, "instance_id": sdata[shape].index.values}),
        )
        new_table = TableModel.parse(new_table, region=shape, region_key="region", instance_key="instance_id")
        del sdata.table
        sdata.table = new_table
        return sdata


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
