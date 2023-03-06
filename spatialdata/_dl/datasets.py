from typing import Callable, Optional

import numpy as np
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from torch.utils.data import Dataset

from spatialdata import SpatialData
from spatialdata._core._rasterize import rasterize
from spatialdata._core.core_utils import get_dims
from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    ShapesModel,
    get_schema,
)


class ImageTilesDataset(Dataset):
    def __init__(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        tile_dim_in_units: float,
        tile_dim_in_pixels: int,
        target_coordinate_system: str = "global",
        data_dict_transform: Optional[Callable[[SpatialData], dict[str, SpatialImage]]] = None,
    ):
        # TODO: we can extend this code to support:
        #  - automatic dermination of the tile_dim_in_pixels to match the image resolution (prevent down/upscaling)
        #  - use the bounding box query instead of the raster function if the user wants
        self.sdata = sdata
        self.regions_to_images = regions_to_images
        self.tile_dim_in_units = tile_dim_in_units
        self.tile_dim_in_pixels = tile_dim_in_pixels
        self.data_dict_transform = data_dict_transform
        self.target_coordinate_system = target_coordinate_system

        self.n_spots_dict = self._compute_n_spots_dict()
        self.n_spots = sum(self.n_spots_dict.values())

    def _validate_regions_to_images(self) -> None:
        for region_key, image_key in self.regions_to_images.items():
            regions_element = self.sdata[region_key]
            images_element = self.sdata[image_key]
            # we could allow also for points
            if not get_schema(regions_element) in [ShapesModel, Labels2DModel, Labels3DModel]:
                raise ValueError(f"regions_element must be a shapes element or a labels element")
            if not get_schema(images_element) in [Image2DModel, Image3DModel]:
                raise ValueError(f"images_element must be an image element")

    def _compute_n_spots_dict(self) -> dict[str, int]:
        n_spots_dict = {}
        for region_key in self.regions_to_images.keys():
            element = self.sdata[region_key]
            # we could allow also points
            if isinstance(element, GeoDataFrame):
                n_spots_dict[region_key] = len(element)
            elif isinstance(element, SpatialImage):
                raise NotImplementedError("labels not supported yet")
            elif isinstance(element, MultiscaleSpatialImage):
                raise NotImplementedError("labels not supported yet")
            else:
                raise ValueError(f"element must be a geodataframe or a spatial image")
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

    def __getitem__(self, idx: int) -> tuple[SpatialImage, str, int]:
        if idx >= self.n_spots:
            raise IndexError()
        regions_name, region_index = self._get_region_info_for_index(idx)
        regions = self.sdata[regions_name]
        # TODO: here we just need to compute the centroids, we probably want to move this functionality to a different file
        if isinstance(regions, GeoDataFrame):
            dims = get_dims(regions)
            region = regions.iloc[region_index]
            # the function coords.xy is just accessing _coords, and wrapping it with extra information, so we access
            # it directly
            centroid = region.geometry.coords._coords[0]
        elif isinstance(regions, SpatialImage):
            raise NotImplementedError("labels not supported yet")
        elif isinstance(regions, MultiscaleSpatialImage):
            raise NotImplementedError("labels not supported yet")
        else:
            raise ValueError(f"element must be shapes or labels")
        min_coordinate = np.array(centroid) - self.tile_dim_in_units / 2
        max_coordinate = np.array(centroid) + self.tile_dim_in_units / 2

        raster = self.sdata[self.regions_to_images[regions_name]]
        tile = rasterize(
            raster,
            axes=dims,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=self.target_coordinate_system,
            target_width=self.tile_dim_in_pixels,
        )
        # TODO: as explained in the TODO in the __init__(), we want to let the user also use the bounding box query instaed of the rasterization
        #  the return function of this function would change, so we need to decide if instead having an extra Tile dataset class
        # from spatialdata._core._spatial_query import BoundingBoxRequest
        # request = BoundingBoxRequest(
        #     target_coordinate_system=self.target_coordinate_system,
        #     axes=dims,
        #     min_coordinate=min_coordinate,
        #     max_coordinate=max_coordinate,
        # )
        # sdata_item = self.sdata.query.bounding_box(**request.to_dict())
        return tile, regions_name, region_index
