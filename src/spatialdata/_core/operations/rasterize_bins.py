from __future__ import annotations

from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
from numpy.random import default_rng
from scipy.sparse import csc_matrix
from skimage.transform import estimate_transform
from spatial_image import SpatialImage

from spatialdata.models import Image2DModel, get_table_keys
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
) -> SpatialImage:
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
    min_col
        If specified, the minimum column index to consider.
    max_col
        If specified, the maximum column index to consider (inclusive).
    min_row
        If specified, the minimum row index to consider.
    max_row
        If specified, the maximum row index to consider (inclusive).
    value_key
        The key(s) (obs columns/var names) in the table that will be used to rasterize the bins.
        If `None`, all the var names will be used, and the returned object will be lazily constructed.

    Returns
    -------
    SpatialImage
        A spatial image object created by rasterizing the specified bins from the spatial data.

    Notes
    -----
    Before calling this function you should ensure that the data geometries are organized in grid-like bins
    (e.g. Visium HD data, but not Visium data). Also you should ensure that bin indices (integer) are defined
    in the `.obs` dataframe of the table associated with the spatial geometries.

    The returned image will have one pixel for each bin, and a coordinate transformation to map the image to the
    original data orientation. In particular, the bins of Visium HD data are in a grid that is slightly rotated;
    the coordinate transformation will adjust for this, so that the returned data is aligned to the original geometries.
    """
    element = sdata[bins]
    table = sdata.tables[table_name]

    _, region_key, instance_key = get_table_keys(table)
    if not table.obs[region_key].dtype == "category":
        raise ValueError(f"Please convert `table.obs['{region_key}']` to a category series to improve performances")
    unique_regions = table.obs[region_key].cat.categories
    if len(unique_regions) > 1 or unique_regions[0] != bins:
        raise ValueError(
            "The table should be associated with the specified bins. "
            f"Found multiple regions annotated by the table: {', '.join(list(unique_regions))}."
        )

    min_row, min_col = table.obs[row_key].min(), table.obs[col_key].min()
    n_rows, n_cols = table.obs[row_key].max() - min_row + 1, table.obs[col_key].max() - min_col + 1
    y = (table.obs[row_key] - min_row).values
    x = (table.obs[col_key] - min_col).values

    keys = ([value_key] if isinstance(value_key, str) else value_key) if value_key is not None else table.var_names

    if (value_key is None or any(key in table.var_names for key in keys)) and not isinstance(table.X, csc_matrix):
        raise ValueError(
            "To speed up bins rasterization, the table should be a csc_matrix matrix. "
            "This can be done by calling `table.X = table.X.tocsc()`.",
        )

    if value_key is None:
        shape = (n_rows, n_cols)

        def channel_rasterization(block_id: tuple[int, int, int] | None) -> np.ndarray:  # type: ignore[type-arg]
            image = np.zeros((1, *shape), dtype=np.uint32)

            if block_id is None:
                return image

            col = table.X[:, block_id[0]]
            bins_indices, data = col.indices, col.data
            image[0, y[bins_indices], x[bins_indices]] = data
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
                bins_indices = table.X[:, key_index].indices
                image[i, y[bins_indices], x[bins_indices]] = table.X[:, key_index].data

    # get the transformation
    assert table.n_obs >= 6, "At least 6 bins are needed to estimate the transformation."

    random_indices = RNG.choice(table.n_obs, min(20, table.n_obs), replace=True)
    location_ids = table.obs[instance_key].iloc[random_indices].values
    sub_gdf, sub_table = element.loc[location_ids], table[random_indices]

    src = np.stack([sub_table.obs[col_key] - min_col, sub_table.obs[row_key] - min_row], axis=1)
    dst = np.stack([sub_gdf.geometry.x, sub_gdf.geometry.y], axis=1)

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

    return Image2DModel.parse(image, transformations=transformations, c_coords=keys, dims=("c", "y", "x"))
