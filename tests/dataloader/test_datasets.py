import numpy as np
import pytest
from spatialdata.dataloader import ImageTilesDataset
from spatialdata.datasets import blobs_annotating_element


class TestImageTilesDataset:
    @pytest.mark.parametrize("image_element", ["blobs_image", "blobs_multiscale_image"])
    @pytest.mark.parametrize(
        "regions_element",
        ["blobs_labels", "blobs_multiscale_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"],
    )
    @pytest.mark.parametrize("table", [True, False])
    def test_validation(self, sdata_blobs, image_element: str, regions_element: str, table: bool):
        if table:
            sdata = blobs_annotating_element(regions_element)
        else:
            sdata = sdata_blobs
            del sdata_blobs.tables["table"]
        _ = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={regions_element: image_element},
            regions_to_coordinate_systems={regions_element: "global"},
            table_name="table" if table else None,
            return_annotations="instance_id" if table else None,
        )

    @pytest.mark.parametrize(
        "regions_element",
        ["blobs_circles", "blobs_polygons", "blobs_multipolygons", "blobs_labels", "blobs_multiscale_labels"],
    )
    @pytest.mark.parametrize("rasterize", [True, False])
    def test_default(self, sdata_blobs, regions_element, rasterize):
        rasterize_kwargs = {"target_unit_to_pixels": 2} if rasterize else {}

        sdata = blobs_annotating_element(regions_element)
        ds = ImageTilesDataset(
            sdata=sdata,
            rasterize=rasterize,
            regions_to_images={regions_element: "blobs_image"},
            regions_to_coordinate_systems={regions_element: "global"},
            rasterize_kwargs=rasterize_kwargs,
            table_name="table",
        )

        sdata_tile = ds[0]
        tile = sdata_tile.images.values().__iter__().__next__()

        if regions_element == "blobs_circles":
            if rasterize:
                assert tile.shape == (3, 20, 20)
            else:
                assert tile.shape == (3, 10, 10)
        elif regions_element == "blobs_polygons":
            if rasterize:
                assert tile.shape == (3, 6, 6)
            else:
                assert tile.shape == (3, 3, 3)
        elif regions_element == "blobs_multipolygons":
            if rasterize:
                assert tile.shape == (3, 9, 9)
            else:
                assert tile.shape == (3, 5, 4)
        elif regions_element == "blobs_labels" or regions_element == "blobs_multiscale_labels":
            if rasterize:
                assert tile.shape == (3, 16, 16)
            else:
                assert tile.shape == (3, 8, 8)
        else:
            raise ValueError(f"Unexpected regions_element: {regions_element}")

        # extent has units in pixel so should be the same as tile shape
        if rasterize:
            assert round(ds.tiles_coords.extent.unique()[0] * 2) == tile.shape[1]
        else:
            # here we have a tolerance of 1 pixel because the size of the tile depends on the values of the centroids
            # and of the extenta and here we keep the test simple.
            # For example, if the centroid is 0.5 and the extent is 0.1, the tile will be 1 pixel since the extent will
            # span 0.4 to 0.6; but if the centroid is 0.95 now the tile will be 2 pixels
            assert np.ceil(ds.tiles_coords.extent.unique()[0]) in [tile.shape[1], tile.shape[1] + 1]
        assert np.all(sdata_tile["table"].obs.columns == ds.sdata["table"].obs.columns)
        assert list(sdata_tile.images.keys())[0] == "blobs_image"

    @pytest.mark.parametrize(
        "regions_element",
        ["blobs_circles", "blobs_polygons", "blobs_multipolygons", "blobs_labels", "blobs_multiscale_labels"],
    )
    @pytest.mark.parametrize("return_annot", [None, "region", ["region", "instance_id"]])
    def test_return_annot(self, sdata_blobs, regions_element, return_annot):
        sdata = blobs_annotating_element(regions_element)
        ds = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={regions_element: "blobs_image"},
            regions_to_coordinate_systems={regions_element: "global"},
            return_annotations=return_annot,
            table_name="table",
        )
        if return_annot is None:
            sdata_tile = ds[0]
            tile = sdata_tile["blobs_image"]
        else:
            tile, annot = ds[0]
        if regions_element == "blobs_circles":
            assert tile.shape == (3, 10, 10)
        elif regions_element == "blobs_polygons":
            assert tile.shape == (3, 3, 3)
        elif regions_element == "blobs_multipolygons":
            assert tile.shape == (3, 5, 4)
        elif regions_element == "blobs_labels" or regions_element == "blobs_multiscale_labels":
            assert tile.shape == (3, 8, 8)
        else:
            raise ValueError(f"Unexpected regions_element: {regions_element}")
        # extent has units in pixel so should be the same as tile shape
        # see comment in the test above explaining why we have a tolerance of 1 pixel
        assert np.ceil(ds.tiles_coords.extent.unique()[0]) in [tile.shape[1], tile.shape[1] + 1]
        if return_annot is not None:
            return_annot = [return_annot] if isinstance(return_annot, str) else return_annot
            assert annot.shape[1] == len(return_annot)

    @pytest.mark.parametrize("rasterize", [True, False])
    @pytest.mark.parametrize("return_annot", [None, "region"])
    def test_multiscale_images(self, sdata_blobs, rasterize: bool, return_annot):
        sdata = blobs_annotating_element("blobs_circles")
        ds = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={"blobs_circles": "blobs_multiscale_image"},
            regions_to_coordinate_systems={"blobs_circles": "global"},
            rasterize=rasterize,
            return_annotations=return_annot,
            table_name="table" if return_annot is not None else None,
            rasterize_kwargs={"target_unit_to_pixels": 1} if rasterize else None,
        )
        if return_annot is None:
            sdata_tile = ds[0]
            tile = sdata_tile["blobs_multiscale_image"]
        else:
            tile, annot = ds[0]
        assert tile.shape == (3, 10, 10)
