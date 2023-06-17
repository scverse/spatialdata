from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame
from scipy.sparse import issparse
from torch.utils.data import Dataset

from spatialdata import SpatialData, bounding_box_query
from spatialdata._core.operations.rasterize import rasterize
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
from spatialdata.transformations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation


class ImageTilesDataset(Dataset):
    INSTANCE_KEY = "instance"
    CS_KEY = "cs"
    REGION_KEY = "region"
    IMAGE_KEY = "image"

    def __init__(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
        tile_scale: float = 1.0,
        tile_dim_in_units: float | None = None,
        raster: bool = False,
        return_table: str | list[str] | None = None,
        *kwargs: Any,
    ):
        """
        :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData` object.

        By default, the dataset returns spatialdata object, but when `return_image` and `return_table`
        are set, the dataset may return a tuple containing:

            - the tile image, centered in the target coordinate system of the region.
            - a vector or scala value from the table.

        Parameters
        ----------
        sdata
            The :class`spatialdata.SpatialData` object.
        regions_to_images
            A mapping betwen region and images. The regions are used to compute the tile centers, while the images are
            used to get the pixel values.
        regions_to_coordinate_systems
            A mapping between regions and coordinate systems. The coordinate systems are used to transform both
            regions coordinates for tiles as well as images.
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
        rasterize
            If True, the regions are rasterized using :func:`spatialdata.rasterize`.
            If False, uses the :func:`spatialdata.bounding_box_query`.
        return_table
            If not None, a value from the table is returned together with the image.
            Only columns in :attr:`anndata.AnnData.obs` and :attr:`anndata.AnnData.X`
            can be returned. It will not be returned a spatialdata object but only a tuple
            containing the image and the table value.

        Returns
        -------
        :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData`.
        """
        # TODO: we can extend this code to support:
        #  - automatic dermination of the tile_dim_in_pixels to match the image resolution (prevent down/upscaling)
        #  - use the bounding box query instead of the raster function if the user wants
        self._validate(sdata, regions_to_images, regions_to_coordinate_systems)
        self._preprocess(tile_scale, tile_dim_in_units)
        self._crop_image: Callable[..., Any] = rasterize if raster else bounding_box_query
        self._return_table = self._get_return_table(return_table)

    def _validate(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
    ) -> None:
        """Validate input parameters."""
        self._region_key = sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        self._instance_key = sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        available_regions = sdata.table.obs[self._region_key].unique()
        cs_region_image = []  # list of tuples (coordinate_system, region, image)

        # check unique matching between regions and images and coordinate systems
        assert len(set(regions_to_images.values())) == len(
            regions_to_images.keys()
        ), "One region cannot be paired to multiple regions."
        assert len(set(regions_to_coordinate_systems.values())) == len(
            regions_to_coordinate_systems.keys()
        ), "One region cannot be paired to multiple coordinate systems."

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

            try:
                cs = regions_to_coordinate_systems[region_key]
                region_trans = get_transformation(region_elem, cs)
                image_trans = get_transformation(image_elem, cs)
                if isinstance(region_trans, BaseTransformation) and isinstance(image_trans, BaseTransformation):
                    cs_region_image.append((cs, region_key, image_key))
            except KeyError as e:
                raise KeyError(f"region {region_key} not found in `regions_to_coordinate_systems`") from e

        self.regions = list(available_regions.keys())  # all regions for the dataloader
        self.sdata = sdata
        self.dataset_table = self.sdata.table.obs[self.sdata.table.obs[self._region_key].isin(self.regions)]
        self._cs_region_image = tuple(cs_region_image)  # tuple of tuples (coordinate_system, region_key, image_key)

    def _preprocess(
        self,
        tile_scale: float = 1.0,
        tile_dim_in_units: float | None = None,
    ) -> None:
        """Preprocess the dataset."""
        index_df = []
        tile_coords_df = []
        dims_l = []

        for cs, region, image in self._cs_region_image:
            # get dims and transformations for the region element
            dims = get_axes_names(region)
            dims_l.append(dims)
            t = get_transformation(region, cs)
            assert isinstance(t, BaseTransformation)

            # get coordinates of centroids and extent for tiles
            tile_coords = _get_tile_coords(self.sdata[region], t, dims, tile_scale, tile_dim_in_units)
            tile_coords_df.append(tile_coords)

            # get instances from region
            inst = self.sdata.table.obs[self.sdata.table.obs[self._region_key] == region][self._instance_key].values

            # get index dictionary, with `instance_id`, `cs`, `region`, and `image`
            df = pd.DataFrame({self.INSTANCE_KEY: inst})
            df[self.CS_KEY] = cs
            df[self.REGION_KEY] = region
            df[self.IMAGE_KEY] = image
            index_df.append(df)

        # concatenate and assign to self
        self.dataset_index = pd.concat(index_df).reset_index(inplace=True, drop=True)
        self.tiles_coords = pd.concat(tile_coords_df).reset_index(inplace=True, drop=True)
        # get table filtered by regions
        self.filtered_table = self.sdata.table.obs[self.sdata.table.obs[self._region_key].isin[self._cs_region_imag[1]]]

        assert len(self.tiles_coords) == len(self.dataset_index)
        dims_ = set(chain(*dims_l))
        assert np.all([i in self.tiles_coords for i in dims_])
        self.dims = list(dims_)

    def _get_return_table(self, return_table: str | list[str] | None) -> Optional[Callable[[int], Any]] | None:
        """Get function to return values from the table of the dataset."""
        if return_table is not None:
            return_table = [return_table] if isinstance(return_table, str) else return_table
            # return callable that always return array of shape (1, len(return_table))
            if return_table in self.dataset_table.obs:
                return lambda x: self.dataset_table.obs[return_table].iloc[x].values.reshape(1, -1)
            if return_table in self.dataset_table.var_names:
                if issparse(self.dataset_table.X):
                    return lambda x: self.dataset_table.X[:, return_table].X[x].A
                return lambda x: self.dataset_table.X[:, return_table].X[x]
        return None

    def __len__(self) -> int:
        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Any | SpatialData:
        """Get item from the dataset."""
        # get the row from the index
        row = self.dataset_index.iloc[idx]
        # get the tile coordinates
        t_coords = self.tiles_coords.iloc[idx]

        image = self.sdata[row["image"]]
        tile = self._crop_image(
            image,
            axes=self.dims,
            min_coordinate=t_coords[[f"min{i}" for i in self.dims]],
            max_coordinate=t_coords[[f"min{i}" for i in self.dims]],
            target_coordinate_system=row["cs"],
        )

        if self._return_table is not None:
            return tile, self.filtered_table(idx)
        return SpatialData(images={t_coords[self.REGION_KEY][idx]: tile}, table=self.dataset_table[idx])

    @property
    def regions(self) -> list[str]:
        """List of regions in the dataset."""
        return self._regions

    @regions.setter
    def regions(self, regions: list[str]) -> None:  # D102
        self._regions = regions

    @property
    def sdata(self) -> SpatialData:
        """The original SpatialData object."""
        return self._sdata

    @sdata.setter
    def sdata(self, sdata: SpatialData) -> None:  # D102
        self._sdata = sdata

    @property
    def coordinate_systems(self) -> list[str]:
        """List of coordinate systems in the dataset."""
        return self._coordinate_systems

    @coordinate_systems.setter
    def coordinate_systems(self, coordinate_systems: list[str]) -> None:  # D102
        self._coordinate_systems = coordinate_systems

    @property
    def tiles_coords(self) -> pd.DataFrame:
        """DataFrame with the index of tiles.

        It contains axis coordinates of the centroids, and extent of the tiles.
        For example, for a 2D image, it contains the following columns:

            - `x`: the x coordinate of the centroid.
            - `y`: the y coordinate of the centroid.
            - `extent`: the extent of the tile.
            - `minx`: the minimum x coordinate of the tile.
            - `miny`: the minimum y coordinate of the tile.
            - `maxx`: the maximum x coordinate of the tile.
            - `maxy`: the maximum y coordinate of the tile.
        """
        return self._tiles_coords

    @tiles_coords.setter
    def tiles_coords(self, tiles: pd.DataFrame) -> None:
        self._tiles_coords = tiles

    @property
    def dataset_index(self) -> pd.DataFrame:
        """DataFrame with the metadata of the tiles.

        It contains the following columns:

            - `instance`: the name of the instance in the region.
            - `cs`: the coordinate system of the region-image pair.
            - `region`: the name of the region.
            - `image`: the name of the image.
        """
        return self._dataset_index

    @dataset_index.setter
    def dataset_index(self, dataset_index: pd.DataFrame) -> None:
        self._dataset_index = dataset_index

    @property
    def dataset_table(self) -> AnnData:
        """AnnData table filtered by the `region` and `cs` present in the dataset."""
        return self._dataset_table

    @dataset_table.setter
    def dataset_table(self, dataset_table: AnnData) -> None:
        self._dataset_table = dataset_table

    @property
    def dims(self) -> list[str]:
        """Dimensions of the dataset."""
        return self._dims

    @dims.setter
    def dims(self, dims: list[str]) -> None:
        self._dims = dims


def _get_tile_coords(
    elem: GeoDataFrame,
    transformation: BaseTransformation,
    dims: tuple[str, ...],
    tile_scale: float | None = None,
    tile_dim_in_units: float | None = None,
) -> pd.DataFrame:
    """Get the (transformed) centroid of the region and the extent."""
    # get centroids and transform them
    centroids = elem.centroid.get_coordinates().values
    aff = transformation.to_affine_matrix(input_axes=dims, output_axes=dims)
    centroids = np.squeeze(_affine_matrix_multiplication(aff, centroids), 0)

    # get extent, first by checking shape defaults, then by using the `tile_dim_in_units`
    if tile_dim_in_units is None:
        if elem.iloc[0][0].geom_type == "Point":
            extent = elem[ShapesModel.RADIUS_KEY].values * tile_scale
        if elem.iloc[0][0].geom_type == "Polygon":
            extent = elem[ShapesModel.GEOMETRY_KEY].length * tile_scale
        raise ValueError("Only point and polygon shapes are supported.")
    if tile_dim_in_units is not None:
        if isinstance(tile_dim_in_units, float):
            extent = np.repeat(tile_dim_in_units, len(centroids))
        if len(extent) != len(centroids):
            raise ValueError(
                f"the number of elements in the region ({len(extent)}) does not match"
                f" the number of instances ({len(centroids)})."
            )

    # transform extent
    aff = transformation.to_affine_matrix(input_axes=tuple(dims[0]), output_axes=tuple(dims[0]))
    extent = np.squeeze(_affine_matrix_multiplication(aff, extent), 0)

    # get min and max coordinates
    min_coordinates = np.array(centroids.values) - extent / 2
    max_coordinates = np.array(centroids.values) + extent / 2

    # return a dataframe with columns e.g.  ["x", "y", "extent", "minx", "miny", "maxx", "maxy"]
    return pd.DataFrame(
        np.hstack([centroids, extent[:, np.newaxis], min_coordinates, max_coordinates]),
        columns=list(dims) + ["extent"] + ["min" + dim for dim in dims] + ["max" + dim for dim in dims],
    )
