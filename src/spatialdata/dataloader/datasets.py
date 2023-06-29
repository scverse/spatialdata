from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from scipy.sparse import issparse
from torch.utils.data import Dataset

from spatialdata._core.spatialdata import SpatialData
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

__all__ = ["ImageTilesDataset"]


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
        return_annot: str | list[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        raster_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ):
        """
        :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData` object.

        By default, the dataset returns spatialdata object, but when `return_image` and `return_annot`
        are set, the dataset may return a tuple containing:

            - the tile image, centered in the target coordinate system of the region.
            - a vector or scalar value from the table.

        Parameters
        ----------
        sdata
            The :class`spatialdata.SpatialData` object.
        regions_to_images
            A mapping between region and images. The regions are used to compute the tile centers, while the images are
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
        raster
            If True, the images are rasterized using :func:`spatialdata.rasterize`.
            If False, they are rasterized using :func:`spatialdata.bounding_box_query`.
        return_annot
            If not None, a value from the table is returned together with the image tile.
            Only columns in :attr:`anndata.AnnData.obs` and :attr:`anndata.AnnData.X`
            can be returned. If None, it will return a spatialdata object with only the tuple
            containing the image and the table value.
        raster_kwargs
            Keyword arguments passed to :func:`spatialdata.rasterize` if `raster` is True.

        Returns
        -------
        :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData`.
        """
        from spatialdata import bounding_box_query
        from spatialdata._core.operations.rasterize import rasterize

        self._validate(sdata, regions_to_images, regions_to_coordinate_systems)
        self._preprocess(tile_scale, tile_dim_in_units)

        self._crop_image: Callable[..., Any] = (
            partial(rasterize, **dict(raster_kwargs)) if raster else bounding_box_query  # type: ignore[assignment]
        )
        self._return = self._get_return(return_annot)
        self.transform = transform

    def _validate(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
    ) -> None:
        """Validate input parameters."""
        self._region_key = sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        self._instance_key = sdata.table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        available_regions = sdata.table.obs[self._region_key].cat.categories
        cs_region_image = []  # list of tuples (coordinate_system, region, image)

        # check unique matching between regions and images and coordinate systems
        assert len(set(regions_to_images.values())) == len(
            regions_to_images.keys()
        ), "One region cannot be paired to multiple regions."
        assert len(set(regions_to_coordinate_systems.values())) == len(
            regions_to_coordinate_systems.keys()
        ), "One region cannot be paired to multiple coordinate systems."

        for region_key, image_key in regions_to_images.items():
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
            if isinstance(image_elem, MultiscaleSpatialImage):
                raise NotImplementedError("Multiscale images are not implemented yet.")

            if region_key not in available_regions:
                raise ValueError(f"region {region_key} not found in the spatialdata object.")

            # check that the coordinate systems are valid for the elements
            try:
                cs = regions_to_coordinate_systems[region_key]
                region_trans = get_transformation(region_elem, cs)
                image_trans = get_transformation(image_elem, cs)
                if isinstance(region_trans, BaseTransformation) and isinstance(image_trans, BaseTransformation):
                    cs_region_image.append((cs, region_key, image_key))
            except KeyError as e:
                raise KeyError(f"region {region_key} not found in `regions_to_coordinate_systems`") from e

        self.regions = list(regions_to_coordinate_systems.keys())  # all regions for the dataloader
        self.sdata = sdata
        self.dataset_table = self.sdata.table[
            self.sdata.table.obs[self._region_key].isin(self.regions)
        ]  # filtered table for the data loader
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
        shapes_l = []

        for cs, region, image in self._cs_region_image:
            # get dims and transformations for the region element
            dims = get_axes_names(self.sdata[region])
            dims_l.append(dims)
            t = get_transformation(self.sdata[region], cs)
            assert isinstance(t, BaseTransformation)

            # get instances from region
            inst = self.sdata.table.obs[self.sdata.table.obs[self._region_key] == region][self._instance_key].values

            # subset the regions by instances
            subset_region = self.sdata[region].iloc[inst]
            # get coordinates of centroids and extent for tiles
            tile_coords = _get_tile_coords(subset_region, t, dims, tile_scale, tile_dim_in_units)
            tile_coords_df.append(tile_coords)

            # get shapes
            shapes_l.append(self.sdata[region])

            # get index dictionary, with `instance_id`, `cs`, `region`, and `image`
            df = pd.DataFrame({self.INSTANCE_KEY: inst})
            df[self.CS_KEY] = cs
            df[self.REGION_KEY] = region
            df[self.IMAGE_KEY] = image
            index_df.append(df)

        # concatenate and assign to self
        self.dataset_index = pd.concat(index_df).reset_index(drop=True)
        self.tiles_coords = pd.concat(tile_coords_df).reset_index(drop=True)
        # get table filtered by regions
        self.filtered_table = self.sdata.table.obs[self.sdata.table.obs[self._region_key].isin(self.regions)]

        assert len(self.tiles_coords) == len(self.dataset_index)
        dims_ = set(chain(*dims_l))
        assert np.all([i in self.tiles_coords for i in dims_])
        self.dims = list(dims_)

    def _get_return(
        self,
        return_annot: str | list[str] | None,
    ) -> Callable[[int, Any], tuple[Any, Any] | SpatialData]:
        """Get function to return values from the table of the dataset."""
        if return_annot is not None:
            # table is always returned as array shape (1, len(return_annot))
            # where return_table can be a single column or a list of columns
            return_annot = [return_annot] if isinstance(return_annot, str) else return_annot
            # return tuple of (tile, table)
            if np.all([i in self.dataset_table.obs for i in return_annot]):
                return lambda x, tile: (tile, self.dataset_table.obs[return_annot].iloc[x].values.reshape(1, -1))
            if np.all([i in self.dataset_table.var_names for i in return_annot]):
                if issparse(self.dataset_table.X):
                    return lambda x, tile: (tile, self.dataset_table[x, return_annot].X.A)
                return lambda x, tile: (tile, self.dataset_table[x, return_annot].X)
            raise ValueError(
                f"`return_annot` must be a column name in the table or a variable name in the table. "
                f"Got {return_annot}."
            )
        # return spatialdata consisting of the image tile and the associated table
        return lambda x, tile: SpatialData(
            images={self.dataset_index.iloc[x][self.IMAGE_KEY]: tile},
            table=self.dataset_table[x],
        )

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
            min_coordinate=t_coords[[f"min{i}" for i in self.dims]].values,
            max_coordinate=t_coords[[f"max{i}" for i in self.dims]].values,
            target_coordinate_system=row["cs"],
        )

        if self.transform is not None:
            out = self._return(idx, tile)
            return self.transform(out)
        return self._return(idx, tile)

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
    centroids = _affine_matrix_multiplication(aff, centroids)

    # get extent, first by checking shape defaults, then by using the `tile_dim_in_units`
    if tile_dim_in_units is None:
        if elem.iloc[0][0].geom_type == "Point":
            extent = elem[ShapesModel.RADIUS_KEY].values * tile_scale
        elif elem.iloc[0][0].geom_type in ["Polygon", "MultiPolygon"]:
            extent = elem[ShapesModel.GEOMETRY_KEY].length * tile_scale
        else:
            raise ValueError("Only point and polygon shapes are supported.")
    if tile_dim_in_units is not None:
        if isinstance(tile_dim_in_units, (float, int)):
            extent = np.repeat(tile_dim_in_units, len(centroids))
        else:
            raise TypeError(
                f"`tile_dim_in_units` must be a `float`, `int`, `list`, `tuple` or `np.ndarray`, "
                f"not {type(tile_dim_in_units)}."
            )
        if len(extent) != len(centroids):
            raise ValueError(
                f"the number of elements in the region ({len(extent)}) does not match"
                f" the number of instances ({len(centroids)})."
            )

    # transform extent
    aff = transformation.to_affine_matrix(input_axes=tuple(dims[0]), output_axes=tuple(dims[0]))
    extent = _affine_matrix_multiplication(aff, np.array(extent)[:, np.newaxis])

    # get min and max coordinates
    min_coordinates = np.array(centroids) - extent / 2
    max_coordinates = np.array(centroids) + extent / 2

    # return a dataframe with columns e.g.  ["x", "y", "extent", "minx", "miny", "maxx", "maxy"]
    return pd.DataFrame(
        np.hstack([centroids, extent, min_coordinates, max_coordinates]),
        columns=list(dims) + ["extent"] + ["min" + dim for dim in dims] + ["max" + dim for dim in dims],
    )
