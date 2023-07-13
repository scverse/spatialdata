from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage
from torch.utils.data import Dataset

from spatialdata._core.operations.rasterize import rasterize
from spatialdata._utils import _affine_matrix_multiplication
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    ShapesModel,
    get_axes_names,
    get_model,
)
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

if TYPE_CHECKING:
    from spatialdata import SpatialData


class ImageTilesDataset(Dataset):
    def __init__(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        tile_dim_in_units: float,
        tile_dim_in_pixels: int,
        target_coordinate_system: str = "global",
        # unused at the moment, see
        transform: Callable[[SpatialData], Any] | None = None,
    ):
        """
        Torch Dataset that returns image tiles around regions from a SpatialData object.

        Parameters
        ----------
        sdata
            The SpatialData object containing the regions and images from which to extract the tiles from.
        regions_to_images
            A dictionary mapping the regions element key we want to extract the tiles around to the images element key
            we want to get the image data from.
        tile_dim_in_units
            The dimension of the requested tile in the units of the target coordinate system. This specifies the extent
            of the image each tile is querying. This is not related he size in pixel of each returned tile.
        tile_dim_in_pixels
            The dimension of the requested tile in pixels. This specifies the size of the output tiles that we will get,
             independently of which extent of the image the tile is covering.
        target_coordinate_system
            The coordinate system in which the tile_dim_in_units is specified.
        """
        # TODO: we can extend this code to support:
        #  - automatic dermination of the tile_dim_in_pixels to match the image resolution (prevent down/upscaling)
        #  - use the bounding box query instead of the raster function if the user wants
        self.sdata = sdata
        self.regions_to_images = regions_to_images
        self.tile_dim_in_units = tile_dim_in_units
        self.tile_dim_in_pixels = tile_dim_in_pixels
        self.transform = transform
        self.target_coordinate_system = target_coordinate_system

        self.n_spots_dict = self._compute_n_spots_dict()
        self.n_spots = sum(self.n_spots_dict.values())

    def _validate_regions_to_images(self) -> None:
        for region_key, image_key in self.regions_to_images.items():
            regions_element = self.sdata[region_key]
            images_element = self.sdata[image_key]
            # we could allow also for points
            if get_model(regions_element) not in [ShapesModel, Labels2DModel, Labels3DModel]:
                raise ValueError("regions_element must be a shapes element or a labels element")
            if get_model(images_element) not in [Image2DModel, Image3DModel]:
                raise ValueError("images_element must be an image element")

    def _compute_n_spots_dict(self) -> dict[str, int]:
        n_spots_dict = {}
        for region_key in self.regions_to_images:
            element = self.sdata[region_key]
            # we could allow also points
            if isinstance(element, GeoDataFrame):
                n_spots_dict[region_key] = len(element)
            elif isinstance(element, (SpatialImage, MultiscaleSpatialImage)):
                raise NotImplementedError("labels not supported yet")
            else:
                raise ValueError("element must be a geodataframe or a spatial image")
        return n_spots_dict

    def _get_region_info_for_index(self, index: int) -> tuple[str, int]:
        # TODO: this implmenetation can be improved
        i = 0
        for region_key, n_spots in self.n_spots_dict.items():
            if index < i + n_spots:
                return region_key, index - i
            i += n_spots
        raise ValueError(f"index {index} is out of range")

    def __len__(self) -> int:
        return self.n_spots

    def __getitem__(self, idx: int) -> Any | SpatialData:
        from spatialdata import SpatialData

        if idx >= self.n_spots:
            raise IndexError()
        regions_name, region_index = self._get_region_info_for_index(idx)
        regions = self.sdata[regions_name]
        # TODO: here we just need to compute the centroids,
        #  we probably want to move this functionality to a different file
        if isinstance(regions, GeoDataFrame):
            dims = get_axes_names(regions)
            region = regions.iloc[region_index]
            shape = regions.geometry.iloc[0]
            if isinstance(shape, Polygon):
                xy = region.geometry.centroid.coords.xy
                centroid = np.array([[xy[0][0], xy[1][0]]])
            elif isinstance(shape, MultiPolygon):
                raise NotImplementedError("MultiPolygon not supported yet")
            elif isinstance(shape, Point):
                xy = region.geometry.coords.xy
                centroid = np.array([[xy[0][0], xy[1][0]]])
            else:
                raise RuntimeError(f"Unsupported type: {type(shape)}")

            t = get_transformation(regions, self.target_coordinate_system)
            assert isinstance(t, BaseTransformation)
            aff = t.to_affine_matrix(input_axes=dims, output_axes=dims)
            transformed_centroid = np.squeeze(_affine_matrix_multiplication(aff, centroid), 0)
        elif isinstance(regions, (SpatialImage, MultiscaleSpatialImage)):
            raise NotImplementedError("labels not supported yet")
        else:
            raise ValueError("element must be shapes or labels")
        min_coordinate = np.array(transformed_centroid) - self.tile_dim_in_units / 2
        max_coordinate = np.array(transformed_centroid) + self.tile_dim_in_units / 2

        raster = self.sdata[self.regions_to_images[regions_name]]
        tile = rasterize(
            raster,
            axes=dims,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=self.target_coordinate_system,
            target_width=self.tile_dim_in_pixels,
        )
        # TODO: as explained in the TODO in the __init__(), we want to let the
        #  user also use the bounding box query instaed of the rasterization
        #  the return function of this function would change, so we need to
        #  decide if instead having an extra Tile dataset class
        # from spatialdata._core._spatial_query import BoundingBoxRequest
        # request = BoundingBoxRequest(
        #     target_coordinate_system=self.target_coordinate_system,
        #     axes=dims,
        #     min_coordinate=min_coordinate,
        #     max_coordinate=max_coordinate,
        # )
        # sdata_item = self.sdata.query.bounding_box(**request.to_dict())
        table = self.sdata.table
        filter_table = False
        if table is not None:
            region = table.uns["spatialdata_attrs"]["region"]
            region_key = table.uns["spatialdata_attrs"]["region_key"]
            instance_key = table.uns["spatialdata_attrs"]["instance_key"]
            if isinstance(region, str):
                if regions_name == region:
                    filter_table = True
            elif isinstance(region, list):
                if regions_name in region:
                    filter_table = True
            else:
                raise ValueError("region must be a string or a list of strings")
        # TODO: maybe slow, we should check if there is a better way to do this
        if filter_table:
            instance = self.sdata[regions_name].iloc[region_index].name
            row = table[(table.obs[region_key] == regions_name) & (table.obs[instance_key] == instance)].copy()
            tile_table = row
        else:
            tile_table = None
        tile_sdata = SpatialData(images={self.regions_to_images[regions_name]: tile}, table=tile_table)
        if self.transform is not None:
            return self.transform(tile_sdata)
        return tile_sdata
