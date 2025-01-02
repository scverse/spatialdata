from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from numpy.random import default_rng
from scipy.sparse import csc_matrix
from shapely import MultiPolygon, Point, Polygon
from skimage.transform import estimate_transform
from xarray import DataArray

from spatialdata._core.query.relational_query import get_values
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata.models import Image2DModel, Labels2DModel, get_table_keys
from spatialdata.transformations import Affine, Sequence, get_transformation

RNG = default_rng(0)


if TYPE_CHECKING:
    from spatialdata import SpatialData


def rasterize_bins(
    sdata: SpatialData,
    bins: str,
    table_name: str,
    col_key: str,
    row_key: str,
    value_key: str | list[str] | None = None,
    return_region_as_labels: bool = False,
) -> DataArray:
    """
    Rasterizes grid-like binned shapes/points annotated by a table (e.g. Visium HD data).

    Parameters
    ----------
    sdata
        The spatial data object containing the grid-like binned element to be rasterized.
    bins
        The name SpatialElement which defines the grid-like bins.
    table_name
        The name of the table annotating the SpatialElement.
    col_key
        Name of a column in `sdata[table_name].obs` containing the column indices (integer) for the bins.
    row_key
        Name of a column in `sdata[table_name].obs` containing the row indices (integer) for the bins.
    value_key
        The key(s) (obs columns/var names) in the table that will be used to rasterize the bins.
        If `None`, all the var names will be used, and the returned object will be lazily constructed.
        Ignored if `return_region_as_labels` is `True`.
    return_regions_as_labels
        If `False` this function returns a `xarray.DataArray` of shape `(c, y, x)` with dimension
         of `c` equal to the number of key(s) specified in `value_key`, or the number of var names
         in `table_name` if `value_key` is `None`.  If `True`, will return labels of shape `(y, x)`,
        where each bin of the `bins` element will be represented as a pixel.

    Returns
    -------
    A spatial image object created by rasterizing the specified bins from the spatial data.

    Notes
    -----
    Before calling this function you should ensure that the data geometries are organized in grid-like bins
    (e.g. Visium HD data, but not Visium data). Also you should ensure that bin indices (integer) are defined
    in the `.obs` dataframe of the table associated with the spatial geometries. If variables from `table.X` are
    being rasterized (typically, gene counts), then the table should be a `csc_matrix` matrix (this can be done
    by calling `sdata[table_name].X = sdata[table_name].X.tocsc()`).

    The returned image will have one pixel for each bin, and a coordinate transformation to map the image to the
    original data orientation. In particular, the bins of Visium HD data are in a grid that is slightly rotated;
    the coordinate transformation will adjust for this, so that the returned data is aligned to the original geometries.

    If `spatialdata-plot` is used to visualized the returned image, the parameter `scale='full'` needs to be passed to
    `.render_shapes()`, to disable an automatic rasterization that would confict with the rasterization performed here.
    """
    element = sdata[bins]
    table = sdata.tables[table_name]
    if not isinstance(element, GeoDataFrame | DaskDataFrame | DataArray):
        raise ValueError("The bins should be a GeoDataFrame, a DaskDataFrame or a DataArray.")
    if isinstance(element, DataArray):
        if "c" in element.dims:
            raise ValueError(
                "If bins is a DataArray, it should hold labels; found a image element instead, with"
                f" 'c': {element.dims}."
            )
        if not np.issubdtype(element.dtype, np.integer):
            raise ValueError(f"If bins is a DataArray, it should hold integers. Found dtype {element.dtype}.")

    _, region_key, instance_key = get_table_keys(table)
    if not table.obs[region_key].dtype == "category":
        raise ValueError(f"Please convert `table.obs['{region_key}']` to a category series to improve performances")
    unique_regions = table.obs[region_key].cat.categories
    if len(unique_regions) > 1:
        raise ValueError(
            f"Found multiple regions annotated by the table: {', '.join(list(unique_regions))}, "
            "currently only tables annotating a single region are supported. Please open a feature request if you are "
            "interested in the general case."
        )
    if unique_regions[0] != bins:
        raise ValueError("The table should be associated with the specified bins.")

    if isinstance(element, DataArray) and return_region_as_labels:
        raise ValueError(
            f"bins is already a labels layer that annotates the table '{table_name}'. "
            "Consider setting 'return_region_as_labels' to 'False' to create a lazy spatial image."
        )

    min_row, min_col = table.obs[row_key].min(), table.obs[col_key].min()
    n_rows, n_cols = table.obs[row_key].max() - min_row + 1, table.obs[col_key].max() - min_col + 1
    y = (table.obs[row_key] - min_row).values
    x = (table.obs[col_key] - min_col).values

    if isinstance(element, DataArray):
        transformations = get_transformation(element, get_all=True)
        assert isinstance(transformations, dict)
    else:
        # get the transformation
        if table.n_obs < 6:
            raise ValueError("At least 6 bins are needed to estimate the transformation.")

        random_indices = RNG.choice(table.n_obs, min(20, table.n_obs), replace=True)
        location_ids = table.obs[instance_key].iloc[random_indices].values
        sub_df, sub_table = element.loc[location_ids], table[random_indices]

        src = np.stack([sub_table.obs[col_key] - min_col, sub_table.obs[row_key] - min_row], axis=1)
        if isinstance(sub_df, GeoDataFrame):
            if isinstance(sub_df.iloc[0].geometry, Point):
                sub_x = sub_df.geometry.x.values
                sub_y = sub_df.geometry.y.values
            else:
                assert isinstance(sub_df.iloc[0].geometry, Polygon | MultiPolygon)
                sub_x = sub_df.centroid.x
                sub_y = sub_df.centroid.y
        else:
            assert isinstance(sub_df, DaskDataFrame)
            sub_x = sub_df.x.compute().values
            sub_y = sub_df.y.compute().values
        dst = np.stack([sub_x, sub_y], axis=1)

        to_bins = Sequence(
            [
                Affine(
                    estimate_transform(ttype="affine", src=src, dst=dst).params,
                    input_axes=("x", "y"),
                    output_axes=("x", "y"),
                )
            ]
        )
        bins_transformations = get_transformation(element, get_all=True)

        assert isinstance(bins_transformations, dict)

        transformations = {cs: to_bins.compose_with(t) for cs, t in bins_transformations.items()}

    if return_region_as_labels:
        dtype = _get_uint_dtype(table.obs[instance_key].max())
        _min_value = table.obs[instance_key].min()
        # TODO: add a new column instead of modyfing the table inplace
        # TODO: do not modify the index of the elements
        if _min_value == 0:
            logger.info(
                f"The minimum value of the instance key column ('table.obs[{instance_key}]') has been"
                " detected to be 0. Since the label 0 is reserved for the background, "
                "both the instance key column in 'table.obs' "
                f"and the index of the annotating element '{bins}' is incremented by 1."
            )
            table.obs[instance_key] += 1
            element.index += 1
        labels_element = np.zeros((n_rows, n_cols), dtype=dtype)
        # make labels layer that can visualy represent the cells
        labels_element[y, x] = table.obs[instance_key].values.T

        return Labels2DModel.parse(data=labels_element, dims=("y", "x"), transformations=transformations)

    keys = ([value_key] if isinstance(value_key, str) else value_key) if value_key is not None else table.var_names

    if (value_key is None or any(key in table.var_names for key in keys)) and not isinstance(
        table.X, csc_matrix | np.ndarray
    ):
        raise ValueError(
            "To speed up bins rasterization, the X matrix in the table, when sparse, should be a csc_matrix matrix. "
            "This can be done by calling `table.X = table.X.tocsc()`.",
        )
    sparse_matrix = isinstance(table.X, csc_matrix)
    if isinstance(value_key, str):
        value_key = [value_key]

    if value_key is None:
        dtype = table.X.dtype
    else:
        values = get_values(value_key=value_key, element=table)
        assert isinstance(values, pd.DataFrame)
        dtype = values[value_key[0]].dtype

    if value_key is None:
        shape = (n_rows, n_cols)

        def channel_rasterization(block_id: tuple[int, int, int] | None) -> ArrayLike:
            image: ArrayLike = np.zeros((1, *shape), dtype=dtype)

            if block_id is None:
                return image

            col = table.X[:, block_id[0]]
            if sparse_matrix:
                bins_indices, data = col.indices, col.data
                image[0, y[bins_indices], x[bins_indices]] = data
            else:
                image[0, y, x] = col
            return image

        image = da.map_blocks(
            channel_rasterization,
            chunks=((1,) * len(keys), *shape),
            dtype=np.uint32,
        )
    else:
        image = np.zeros((len(value_key), n_rows, n_cols))

        if keys[0] in table.obs:
            image[:, y, x] = table.obs[keys].values.T
        else:
            for i, key in enumerate(keys):
                key_index = table.var_names.get_loc(key)
                if sparse_matrix:
                    bins_indices = table.X[:, key_index].indices
                    image[i, y[bins_indices], x[bins_indices]] = table.X[:, key_index].data
                else:
                    image[i, y, x] = table.X[:, key_index]

    return Image2DModel.parse(
        data=image,
        dims=("c", "y", "x"),
        transformations=transformations,
        c_coords=keys,
    )


def _get_uint_dtype(value: int) -> str:
    max_uint64 = np.iinfo(np.uint64).max
    max_uint32 = np.iinfo(np.uint32).max
    max_uint16 = np.iinfo(np.uint16).max

    if max_uint16 >= value:
        dtype = "uint16"
    elif max_uint32 >= value:
        dtype = "uint32"
    elif max_uint64 >= value:
        dtype = "uint64"
    else:
        raise ValueError(f"Maximum cell number is {value}. Values higher than {max_uint64} are not supported.")
    return dtype
