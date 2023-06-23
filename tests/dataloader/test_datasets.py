import contextlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata._core.spatialdata import SpatialData
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

    @pytest.mark.parametrize("regions_element", ["blobs_circles", "blobs_polygons", "blobs_multipolygons"])
    @pytest.mark.parametrize("raster", [True, False])
    def test_default(self, sdata_blobs, regions_element, raster):
        raster_kwargs = {"target_unit_to_pixels": 2} if raster else {}

        sdata = self._annotate_shapes(sdata_blobs, regions_element)
        ds = ImageTilesDataset(
            sdata=sdata,
            raster=raster,
            regions_to_images={regions_element: "blobs_image"},
            regions_to_coordinate_systems={regions_element: "global"},
            raster_kwargs=raster_kwargs,
        )

        sdata_tile = ds[0].__next__()
        tile = sdata_tile.images.values().__iter__().__next__()

        if regions_element == "blobs_circles":
            if raster:
                assert tile.shape == (3, 50, 50)
            else:
                assert tile.shape == (3, 25, 25)
        elif regions_element == "blobs_polygons":
            if raster:
                assert tile.shape == (3, 164, 164)
            else:
                assert tile.shape == (3, 82, 82)
        elif regions_element == "blobs_multipolygons":
            if raster:
                assert tile.shape == (3, 329, 329)
            else:
                assert tile.shape == (3, 164, 164)
        else:
            raise ValueError(f"Unexpected regions_element: {regions_element}")
        # extent has units in pixel so should be the same as tile shape
        if raster:
            assert round(ds.tiles_coords.extent.unique()[0] * 2) == tile.shape[1]
        else:
            assert int(ds.tiles_coords.extent.unique()[0]) == tile.shape[1]
        assert np.all(sdata_tile.table.obs.columns == ds.sdata.table.obs.columns)
        assert list(sdata_tile.images.keys())[0] == "blobs_image"

    @pytest.mark.parametrize("regions_element", ["blobs_circles", "blobs_polygons", "blobs_multipolygons"])
    @pytest.mark.parametrize("return_annot", ["region", ["region", "instance_id"]])
    def test_return_annot(self, sdata_blobs, regions_element, return_annot):
        sdata = self._annotate_shapes(sdata_blobs, regions_element)
        ds = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={regions_element: "blobs_image"},
            regions_to_coordinate_systems={regions_element: "global"},
            return_annot=return_annot,
        )

        tile, annot = ds[0].__next__()
        if regions_element == "blobs_circles":
            assert tile.shape == (3, 25, 25)
        elif regions_element == "blobs_polygons":
            assert tile.shape == (3, 82, 82)
        elif regions_element == "blobs_multipolygons":
            assert tile.shape == (3, 164, 164)
        else:
            raise ValueError(f"Unexpected regions_element: {regions_element}")
        # extent has units in pixel so should be the same as tile shape
        assert int(ds.tiles_coords.extent.unique()[0]) == tile.shape[1]
        return_annot = [return_annot] if isinstance(return_annot, str) else return_annot
        assert annot.shape[1] == len(return_annot)

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
