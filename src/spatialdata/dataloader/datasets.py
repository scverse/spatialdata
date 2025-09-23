import warnings
from collections.abc import Callable, Mapping
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame
from pandas import CategoricalDtype
from scipy.sparse import issparse
from torch.utils.data import Dataset
from xarray import DataArray, DataTree

from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata._core.operations.vectorize import to_circles
from spatialdata._core.query.relational_query import get_element_instances, join_spatialelement_table
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    TableModel,
    get_axes_names,
    get_model,
    get_table_keys,
)
from spatialdata.transformations import BaseTransformation, get_transformation, set_transformation

__all__ = ["ImageTilesDataset"]


class ImageTilesDataset(Dataset):
    """
    :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData` object.

    By default, the dataset returns a :class:`spatialdata.SpatialData` object, but when `return_annotations` is not
    `None`, the dataset returns a tuple containing:

        - the tile image, centered in the target coordinate system of the region.
        - a vector or scalar value from the table.

    Parameters
    ----------
    sdata
        The :class:`spatialdata.SpatialData` object.
    regions_to_images
        A mapping between regions (labels, shapes) and images.
        The regions' centroids will be the tile centers, while the images will be used to get the pixel values.
    regions_to_coordinate_systems
        A mapping between regions and coordinate systems.
        The coordinate systems are used to transform both the centroid coordinates of the regions and the images.
    tile_scale
        This parameter is used to determine the size (width and height) of the tiles.
        Each tile will have size in units equal to tile_scale times the diameter of the circle that approximates (=same
        area) the region that defines the tile.

        For example, suppose the regions to be multiscale labels; this is how the tiles are created:

            1) for each tile, each label region is approximated with a circle with the same area of the label region.
            2) The tile is then created as having the width/height equal to the diameter of the circle,
               multiplied by `tile_scale`.

        If `tile_dim_in_units` is passed, `tile_scale` is ignored.
    tile_dim_in_units
        The dimension of the requested tile in the units of the target coordinate system.
        This specifies the extent of the tile; this parameter is not related to the size in pixel of each returned tile.
        If `tile_dim_in_units` is passed, `tile_scale` is ignored.
    rasterize
        If `True`, the images are rasterized using :func:`spatialdata.rasterize` into the target coordinate system;
        this applies the coordinate transformations to the data.
        If `False`, the images are queried using :func:`spatialdata.bounding_box_query` from the pixel coordinate
        system; this back-transforms the target tile into the pixel coordinates. If the back-transformed tile is not
        aligned with the pixel grid, the returned tile will correspond to the bounding box of the back-transformed tile
        (so that the returned tile is axis-aligned to the pixel grid).
    return_annotations
        If not `None`, one or more values from the table are returned together with the image tile in a tuple.
        Only columns in :attr:`anndata.AnnData.obs` and :attr:`anndata.AnnData.X` can be returned.
        If `None`, it will return a `SpatialData` object with the table consisting of the row that annotates the region
        from which the tile was extracted.
    table_name
        The name of the table in the `SpatialData` object to be used for the annotations. Currently only a single table
        is supported. If you have multiple tables, you can concatenate them into a single table that annotates multiple
        regions.
    transform
        A data transformations (for instance, a normalization operation; not to be confused with a coordinate
        transformation) to be applied to the image and the table value.
        It is a `Callable`, with `Any` as return type, that takes as input the (image, table_value) tuple (when
        `return_annotations` is not `None`) or a `Callable` that takes as input the `SpatialData` object (when
        `return_annotations` is `None`).
    rasterize_kwargs
        Keyword arguments passed to :func:`spatialdata.rasterize` if `rasterize` is `True`.
        This argument can be used in particular to choose the pixel dimension of the produced image tiles; please refer
        to the :func:`spatialdata.rasterize` documentation for this use case.

    Returns
    -------
    :class:`torch.utils.data.Dataset` for loading tiles from a :class:`spatialdata.SpatialData`.
    """

    INSTANCE_KEY = "instance_id"
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
        rasterize: bool = False,
        return_annotations: str | list[str] | None = None,
        table_name: str | None = None,
        transform: Callable[[Any], Any] | None = None,
        rasterize_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ):
        from spatialdata import bounding_box_query
        from spatialdata._core.operations.rasterize import rasterize as rasterize_fn

        self._validate(sdata, regions_to_images, regions_to_coordinate_systems, return_annotations, table_name)
        self._preprocess(tile_scale, tile_dim_in_units, rasterize, table_name)

        if rasterize_kwargs is not None and len(rasterize_kwargs) > 0 and rasterize is False:
            warnings.warn(
                "rasterize_kwargs are passed to the rasterize function, but rasterize is set to False. The arguments "
                "will be ignored. If you want to use the rasterize function, please set rasterize to True.",
                UserWarning,
                stacklevel=2,
            )

        self._crop_image: Callable[..., Any] = (
            partial(
                rasterize_fn,
                **dict(rasterize_kwargs),
            )
            if rasterize
            else bounding_box_query
        )
        self._return = self._get_return(return_annotations, table_name)
        self.transform = transform

    def _validate(
        self,
        sdata: SpatialData,
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
        return_annotations: str | list[str] | None,
        table_name: str | None,
    ) -> None:
        """Validate input parameters."""
        self.sdata = sdata
        if return_annotations is not None and table_name is None:
            raise ValueError("`table_name` must be provided if `return_annotations` is not `None`.")

        # check that the regions specified in the two dicts are the same
        assert set(regions_to_images.keys()) == set(regions_to_coordinate_systems.keys()), (
            "The keys in `regions_to_images` and `regions_to_coordinate_systems` must be the same."
        )
        self.regions = list(regions_to_coordinate_systems.keys())  # all regions for the dataloader

        cs_region_image: list[tuple[str, str, str]] = []  # list of tuples (coordinate_system, region, image)
        for region_name in self.regions:
            image_name = regions_to_images[region_name]

            # get elements
            region_elem = sdata[region_name]
            image_elem = sdata[image_name]

            # check that the elements are supported
            if get_model(region_elem) == PointsModel:
                raise ValueError(
                    "`regions_element` must be a shapes or labels element, points are currently not supported."
                )
            if get_model(image_elem) not in [Image2DModel, Image3DModel]:
                raise ValueError("`images_element` must be an image element.")

            # check that the coordinate systems are valid for the elements
            cs = regions_to_coordinate_systems[region_name]
            region_trans = get_transformation(region_elem, get_all=True)
            image_trans = get_transformation(image_elem, get_all=True)
            assert isinstance(region_trans, dict)
            assert isinstance(image_trans, dict)
            if cs in region_trans and cs in image_trans:
                cs_region_image.append((cs, region_name, image_name))
            else:
                raise ValueError(
                    f"The coordinate system `{cs}` is not valid for the region `{region_name}` and image `{image_name}`"
                    "."
                )

            if table_name is not None:
                _, region_key, instance_key = get_table_keys(sdata.tables[table_name])
                if get_model(region_elem) in [Labels2DModel, Labels3DModel]:
                    indices = get_element_instances(region_elem).tolist()
                else:
                    indices = region_elem.index.tolist()
                table = sdata.tables[table_name]
                if not isinstance(sdata.tables[table_name].obs[region_key].dtype, CategoricalDtype):
                    raise TypeError(
                        f"The `regions_element` column `{region_key}` in the table must be a categorical dtype. "
                        f"Please convert it."
                    )
                instance_column = table.obs[table.obs[region_key] == region_name][instance_key].tolist()
                if not set(indices).issubset(instance_column):
                    raise RuntimeError(
                        "Some of the instances in the region element are not annotated by the table. Instances of the "
                        f"regions element: {indices}. Instances in the table: {instance_column}. You can remove some "
                        "instances from the region element, add them to the table, or set `table_name` to `None` (if "
                        "the table annotation can be excluded from the dataset)."
                    )
        self._cs_region_image = tuple(cs_region_image)  # tuple of tuples (coordinate_system, region_name, image_name)

    def _preprocess(
        self,
        tile_scale: float,
        tile_dim_in_units: float | None,
        rasterize: bool,
        table_name: str | None,
    ) -> None:
        """Preprocess the dataset."""
        if table_name is not None:
            _, region_key, instance_key = get_table_keys(self.sdata.tables[table_name])
            filtered_table = self.sdata.tables[table_name][
                self.sdata.tables[table_name].obs[region_key].isin(self.regions)
            ]  # filtered table for the data loader

        index_df = []
        tile_coords_df = []
        dims_l = []
        tables_l = []
        for cs, region_name, image_name in self._cs_region_image:
            circles = to_circles(self.sdata[region_name])
            dims_l.append(get_axes_names(circles))

            tile_coords = _get_tile_coords(
                circles=circles,
                cs=cs,
                rasterize=rasterize,
                tile_scale=tile_scale,
                tile_dim_in_units=tile_dim_in_units,
            )
            tile_coords_df.append(tile_coords)

            inst = circles.index.values
            df = pd.DataFrame({self.INSTANCE_KEY: inst})
            df[self.CS_KEY] = cs
            df[self.REGION_KEY] = region_name
            df[self.IMAGE_KEY] = image_name
            index_df.append(df)

            if table_name is not None:
                table_subset = filtered_table[filtered_table.obs[region_key] == region_name]
                circles_sdata = SpatialData.init_from_elements({region_name: circles, "table": table_subset.copy()})
                _, table = join_spatialelement_table(
                    sdata=circles_sdata,
                    spatial_element_names=region_name,
                    table_name="table",
                    how="left",
                    match_rows="left",
                )
                # get index dictionary, with `instance_id`, `cs`, `region`, and `image`
                tables_l.append(table)

        # concatenate and assign to self
        self.tiles_coords = pd.concat(tile_coords_df).reset_index(drop=True)
        self.dataset_index = pd.concat(index_df).reset_index(drop=True)
        assert len(self.tiles_coords) == len(self.dataset_index)
        if table_name:
            self.dataset_table = ad.concat(*tables_l)
            assert len(self.tiles_coords) == len(self.dataset_table)

        dims_ = set(chain(*dims_l))
        assert np.all([i in self.tiles_coords for i in dims_])
        self.dims = list(dims_)

    @staticmethod
    def _ensure_single_scale(data: DataArray | DataTree) -> DataArray:
        if isinstance(data, DataArray):
            return data
        if isinstance(data, DataTree):
            return next(iter(data["scale0"].ds.values()))
        raise ValueError(f"Expected a DataArray or DataTree, got {type(data)}.")

    @staticmethod
    def _return_function(
        idx: int,
        tile: Any,
        dataset_table: AnnData,
        dataset_index: pd.DataFrame,
        table_name: str | None,
        return_annot: str | list[str] | None,
    ) -> tuple[Any, Any] | SpatialData:
        tile = ImageTilesDataset._ensure_single_scale(tile)
        if return_annot is not None:
            # table is always returned as array shape (1, len(return_annot))
            # where return_table can be a single column or a list of columns
            return_annot = [return_annot] if isinstance(return_annot, str) else return_annot
            # return tuple of (tile, table)
            if np.all([i in dataset_table.obs for i in return_annot]):
                return tile, dataset_table.obs[return_annot].iloc[idx].values.reshape(1, -1)
            if np.all([i in dataset_table.var_names for i in return_annot]):
                if issparse(dataset_table.X):
                    return tile, dataset_table[idx, return_annot].X.A
                return tile, dataset_table[idx, return_annot].X
            raise ValueError(
                f"If `return_annot` is a `str`, it must be a column name in the table or a variable name in the table. "
                f"If it is a `list` of `str`, each element should be as above, and they should all be entirely in obs "
                f"or entirely in var. Got {return_annot}."
            )
        # return spatialdata consisting of the image tile and, if available, the associated table
        if table_name:
            table_row = dataset_table[idx].copy()
            # let's reset the target annotation metadata to avoid a warning when constructing the SpatialData object
            if TableModel.ATTRS_KEY in table_row.uns:
                del table_row.uns[TableModel.ATTRS_KEY]
            # TODO: add the shape used for constructing the tile; in the case of the label consider adding the circles
            # or a crop of the label
            return SpatialData(
                images={dataset_index.iloc[idx][ImageTilesDataset.IMAGE_KEY]: tile},
                tables={table_name: table_row},
            )
        return SpatialData(images={dataset_index.iloc[idx][ImageTilesDataset.IMAGE_KEY]: tile})

    def _get_return(
        self,
        return_annot: str | list[str] | None,
        table_name: str | None,
    ) -> Callable[[int, Any], tuple[Any, Any] | SpatialData]:
        """Get function to return values from the table of the dataset."""
        return partial(
            ImageTilesDataset._return_function,
            dataset_table=self.dataset_table if table_name else None,
            dataset_index=self.dataset_index,
            table_name=table_name,
            return_annot=return_annot,
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
            axes=tuple(self.dims),
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
    circles: GeoDataFrame,
    cs: str,
    rasterize: bool,
    tile_scale: float | None = None,
    tile_dim_in_units: float | None = None,
) -> pd.DataFrame:
    """Get the (transformed) centroid of the region and the extent."""
    transform(circles, to_coordinate_system=cs)
    if tile_dim_in_units is not None:
        circles.radius = tile_dim_in_units / 2
    else:
        circles.radius *= tile_scale
    # if rasterize is True, the tile dim is determined from the diameter of the circles in cs; else we need to
    # transform the circles to the intrinsic coordinate system of the element
    if not rasterize:
        transformation = get_transformation(circles, to_coordinate_system=cs)
        assert isinstance(transformation, BaseTransformation)
        back_transformation = transformation.inverse()
        set_transformation(circles, back_transformation, to_coordinate_system="intrinsic_of_element")
        transform(circles, to_coordinate_system="intrinsic_of_element")

    # extent, aka the tile size
    extent = (circles.radius * 2).values.reshape(-1, 1)
    centroids_points = get_centroids(circles, coordinate_system=cs)
    axes = get_axes_names(centroids_points)
    centroids_numpy = centroids_points.compute().values

    # get min and max coordinates
    min_coordinates = np.array(centroids_numpy) - extent / 2
    max_coordinates = np.array(centroids_numpy) + extent / 2

    # return a dataframe with columns e.g.  ["x", "y", "extent", "minx", "miny", "maxx", "maxy"]
    return pd.DataFrame(
        np.hstack([centroids_numpy, extent, min_coordinates, max_coordinates]),
        columns=list(axes) + ["extent"] + ["min" + ax for ax in axes] + ["max" + ax for ax in axes],
    )
