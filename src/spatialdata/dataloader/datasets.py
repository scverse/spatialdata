from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage
from torch.utils.data import Dataset

from spatialdata._core.operations.rasterize import rasterize
from spatialdata._types import ArrayLike
from spatialdata._utils import _affine_matrix_multiplication
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    ShapesModel,
    TableModel,
    get_axes_names,
    get_model,
)
from spatialdata.models._utils import SpatialElement
from spatialdata.transformations import get_transformation
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

if TYPE_CHECKING:
    from spatialdata import SpatialData


class ImageTilesDataset(Dataset):
    CS_KEY = "CS"
    REGION_KEY = "REGION"
    IMAGE_KEY = "IMAGE"
    INSTANCE_KEY = "INSTANCE"

    def __init__(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        tiel_scale: float = 1.0,
        tile_dim_in_units: float | None = None,
        tile_dim_in_pixels: int | None = None,
        coordinate_systems: str | list[str] = None,
        # unused at the moment, see
        transform: Optional[Callable[[SpatialData], Any]] = None,
    ):
        """
        :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData` object.

        Parameters
        ----------
        sdata
            The :class`spatialdata.SpatialData` object.
        regions_to_images
            A mapping betwen region and images. The regions are used to compute the tile centers, while the images are
            used to get the pixel values.
        tile_scale
            The scale of the tiles. This is used only if the `regions` are `shapes`.
            It is a scaling factor applied to either the radius (spots) or length (polygons) of the `shapes`
            according to the geometry type of the `shapes` element:

                - if `shapes` are circles (spots), the radius is scaled by `tile_scale`.
                - if `shapes` are polygons, the length of the polygon is scaled by `tile_scale`.

            If `tile_dim_in_units` is passed, `tile_scale` is ignored.
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
        coordinate_systems = [coordinate_systems] if isinstance(coordinate_systems, str) else coordinate_systems
        self._validate(sdata, regions_to_images, coordinate_systems)

        self.tile_dim_in_units = tile_dim_in_units
        self.tile_dim_in_pixels = tile_dim_in_pixels

        self.n_spots_dict = self._compute_n_spots_dict()
        self.n_spots = sum(self.n_spots_dict.values())

    def _validate(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        coordinate_systems: list[str],
    ) -> None:
        """Validate input parameters."""
        self._region_key = sdata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        self._instance_key = sdata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        available_regions = sdata.obs[region_key].unique()
        cs_region_image = []  # list of tuples (coordinate_system, region, image)

        for region_key, image_key in regions_to_images.items():
            if region_key not in available_regions:
                raise ValueError(f"region {region_key} not found in the spatialdata object.")

            # get elements
            region_elem = sdata[region_key]
            image_elem = sdata[image_key]

            # check that the elements are supported
            if get_model(region_elem) in [Labels2DModel, Labels3DModel]:
                raise NotImplementedError("labels elements are not implemented yet.")
            if get_model(region_elem) not in [ShapesModel]:
                raise ValueError("`regions_element` must be a shapes element.")
            if get_model(image_elem) not in [Image2DModel, Image3DModel]:
                raise ValueError("`images_element` must be an image element.")

            # check that the coordinate systems are valid for the elements
            region_trans = get_transformation(region_elem)
            image_trans = get_transformation(image_elem)

            for cs in coordinate_systems:
                if cs in region_trans and cs in image_trans:
                    cs_region_image.append(tuple(cs, region_key, image_key))

        self.regions = list(available_regions.keys())  # all regions for the dataloader
        self.sdata = sdata
        self._cs_region_image = tuple(cs_region_image)  # tuple(coordinate_system, region_key, image_key)

    def _preprocess(self, tile_scale, tile_dim_in_units) -> None:
        """Preprocess the dataset."""
        index_df = []

        for cs, region, image in self._cs_region_image:
            # get instances from region
            inst = self.sdata.table.obs[self.sdata.table.obs[self._region_key] == region][self._instance_key]
            # get extent for bounding box
            extent = self._get_extent(self.sdata[region], tile_scale, tile_dim_in_units)
            if len(extent) == 1:
                extent = np.repeat(extent, len(inst))
            if len(extent) != len(inst):
                raise ValueError(
                    f"the number of elements in the region {region} ({len(extent)}) does not match the number of "
                    f"instances ({len(inst)})."
                )
            df = pd.DataFrame({self.INSTANCE_KEY: inst})
            df[self.CS_KEY] = cs
            df[self.REGION_KEY] = region
            df[self.IMAGE_KEY] = image
            index_df.append(df)

        self.index = pd.concat(index_df).reset_index(inplace=True, drop=True)

    def _get_extent(
        self,
        elem: SpatialElement,
        tile_scale: float | None = None,
        tile_dim_in_units: float | None = None,
    ) -> ArrayLike:
        """Get the extent of the region."""
        if tile_dim_in_units is None:
            if elem.iloc[0][0].geom_type == "Point":
                return elem[ShapesModel.RADIUS_KEY].values * tile_scale
            if elem.iloc[0][0].geom_type == "Polygon":
                return elem[ShapesModel.GEOMETRY_KEY].length * tile_scale
            raise ValueError("Only point and polygon shapes are supported.")
        return tile_dim_in_units

    def _get_unique_instances(self) -> dict[str, Any]:
        n_spots_dict = {}
        region_key = self.sdata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
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

    @property
    def regions(self) -> list[str]:
        return self._regions

    @regions.setter
    def regions(self, regions: list[str]) -> None:
        self._regions = regions

    @property
    def sdata(self) -> SpatialData:
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: SpatialData) -> None:
        self._sdata = sdata

    @property
    def coordinate_systems(self) -> list[str]:
        return self._coordinate_systems

    @coordinate_systems.setter
    def coordinate_systems(self, coordinate_systems: list[str]) -> None:
        self._coordinate_systems = coordinate_systems
