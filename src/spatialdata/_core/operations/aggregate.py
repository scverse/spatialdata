from __future__ import annotations

import warnings
from typing import Any

import anndata as ad
import dask as da
import dask.dataframe as ddf
import geopandas as gpd
import numpy as np
import pandas as pd
from multiscale_spatial_image import MultiscaleSpatialImage
from scipy import sparse
from spatial_image import SpatialImage
from xrspatial import zonal_stats

from spatialdata._core.operations.transform import transform
from spatialdata._core.query._utils import circles_to_polygons
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    get_model,
)
from spatialdata.transformations import BaseTransformation, Identity, get_transformation

__all__ = ["aggregate"]


def aggregate(
    values: ddf.DataFrame | gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage,
    by: gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage,
    *,
    value_key: list[str] | str | None = None,
    agg_func: str | list[str] = "mean",
    target_coordinate_system: str = "global",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Aggregate values by given region.

    Parameters
    ----------
    values
        Values to aggregate.
    by
        Regions to aggregate by.
    value_key
        Name (or list of names) of the columns containing the values to aggregate; can refer both to numerical or
        categorical values. If the values are categorical, value_key can't be a list.

        The key can be:

             - the name of a column(s) in the dataframe (Dask DataFrame for points or GeoDataFrame for shapes);
             - the name of obs column(s) in the associated AnnData table (for shapes and labels);
             - the name of a var(s), referring to the column(s) of the X matrix in the table (for shapes and labels).

        If nothing is passed here, it defaults to the equivalent of a column of ones.
        Defaults to `FEATURE_KEY` for points (if present).
    agg_func
        Aggregation function to apply over point values, e.g. "mean", "sum", "count".
        Passed to :func:`pandas.DataFrame.groupby.agg` or to :func:`xrspatial.zonal_stats`
        according to the type of `values`.
    target_coordinate_system
        Coordinate system to transform to before aggregating.
    kwargs
        Additional keyword arguments to pass to :func:`xrspatial.zonal_stats`.

    Returns
    -------
    If value_key refers to a categorical variable, returns an AnnData of shape (by.shape[0], <n categories>).
    AnnData of shape (by.shape[0], values[id_key].nunique())])

    Notes
    -----
    This function returns an AnnData object. Use :func:`spatialdata.SpatialData.aggregate` to return a `SpatialData`
    object instead (with the table already referring to the regions passed in `by`).

    When aggregation points by shapes, the current implementation loads all the points into
    memory and thus could lead to a large memory usage. This Github issue
    https://github.com/scverse/spatialdata/issues/210 keeps track of the changes required to
    address this behavior.
    """
    # get schema
    by_type = get_model(by)
    values_type = get_model(values)

    # get transformation between coordinate systems
    by_transform: BaseTransformation = get_transformation(by, target_coordinate_system)  # type: ignore[assignment]
    values_transform: BaseTransformation = get_transformation(
        values,
        target_coordinate_system,  # type: ignore[assignment]
    )
    if not (by_transform == values_transform and isinstance(values_transform, Identity)):
        by = transform(by, by_transform)
        values = transform(values, values_transform)

    # dispatch
    if by_type is ShapesModel:
        if values_type is PointsModel:
            return _aggregate_points_by_shapes(values, by, value_key=value_key, agg_func=agg_func)
        if values_type is ShapesModel:
            return _aggregate_shapes_by_shapes(values, by, value_key=value_key, agg_func=agg_func)
    if by_type is Labels2DModel and values_type is Image2DModel:
        return _aggregate_image_by_labels(values, by, agg_func, **kwargs)
    raise NotImplementedError(f"Cannot aggregate {values_type} by {by_type}")


def _aggregate_points_by_shapes(
    points: ddf.DataFrame | pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    value_key: str | None = None,
    agg_func: str | list[str] = "count",
) -> ad.AnnData:
    from spatialdata.models import points_dask_dataframe_to_geopandas

    # Default value for value_key is ATTRS_KEY for points (if present)
    if value_key is None and PointsModel.ATTRS_KEY in points.attrs:
        value_key = points.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY]

    points = points_dask_dataframe_to_geopandas(points, suppress_z_warning=True)
    shapes = circles_to_polygons(shapes)

    return _aggregate_shapes(values=points, by=shapes, value_key=value_key, agg_func=agg_func)


def _aggregate_shapes_by_shapes(
    values: gpd.GeoDataFrame,
    by: gpd.GeoDataFrame,
    *,
    value_key: str | None = None,
    agg_func: str | list[str] = "count",
) -> ad.AnnData:
    values = circles_to_polygons(values)
    by = circles_to_polygons(by)

    return _aggregate_shapes(values=values, by=by, value_key=value_key, agg_func=agg_func)


def _aggregate_image_by_labels(
    values: SpatialImage | MultiscaleSpatialImage,
    by: SpatialImage | MultiscaleSpatialImage,
    agg_func: str | list[str] = "mean",
    **kwargs: Any,
) -> ad.AnnData:
    """
    Aggregate values by given labels.

    Parameters
    ----------
    values
        Values to aggregate.
    by
        Regions to aggregate by.
    agg_func
        Aggregation function to apply over point values, e.g. "mean", "sum", "count"
        from :func:`xrspatial.zonal_stats`.
    kwargs
        Additional keyword arguments to pass to :func:`xrspatial.zonal_stats`.

    Returns
    -------
    AnnData of shape `(by.shape[0], len(agg_func)]`.
    """
    if isinstance(by, MultiscaleSpatialImage):
        assert len(by["scale0"]) == 1
        by = next(iter(by["scale0"].values()))
    if isinstance(values, MultiscaleSpatialImage):
        assert len(values["scale0"]) == 1
        values = next(iter(values["scale0"].values()))

    agg_func = [agg_func] if isinstance(agg_func, str) else agg_func
    outs = []

    for i, c in enumerate(values.coords["c"].values):
        with warnings.catch_warnings():  # ideally fix upstream
            warnings.filterwarnings(
                "ignore",
                message=".*unknown divisions.*",
            )
            out = zonal_stats(by, values[i, ...], stats_funcs=agg_func, **kwargs).compute()
        out.columns = [f"channel_{c}_{col}" if col != "zone" else col for col in out.columns]
        out = out.loc[out["zone"] != 0].copy()
        zones: ArrayLike = out["zone"].values
        outs.append(out.drop(columns=["zone"]))  # remove the 0 (background)
    df = pd.concat(outs, axis=1)

    X = sparse.csr_matrix(df.values)

    index = kwargs.get("zone_ids", None)  # `zone_ids` allows the user to select specific labels to aggregate by
    if index is None:
        index = np.array(da.array.unique(by.data))
        assert np.array(index == np.insert(zones, 0, 0)).all(), "Index mismatch between zonal stats and labels."
    return ad.AnnData(
        X,
        obs=pd.DataFrame(index=zones.astype(str)),
        var=pd.DataFrame(index=df.columns),
        dtype=X.dtype,
    )


def _aggregate_shapes(
    values: gpd.GeoDataFrame,
    by: gpd.GeoDataFrame,
    value_key: str | None = None,
    agg_func: str | list[str] = "count",
) -> ad.AnnData:
    """
    Inner function to aggregate geopandas objects.

    See docstring for `aggregate` for semantics.

    Parameters
    ----------
    values
        Geopandas dataframe to be aggregated. Must have a geometry column.
    by
        Geopandas dataframe to group values by. Must have a geometry column.
    value_key
        Column in value dataframe to perform aggregation on.
    agg_func
        Aggregation function to apply over grouped values. Passed to pandas.DataFrame.groupby.agg.
    """
    if by.index.name is None:
        by.index.name = "cell"
    by_id_key = by.index.name
    joined = by.sjoin(value)

    if value_key is None:
        point_values = np.broadcast_to(True, joined.shape[0])
        value_key = "count"
    else:
        point_values = joined[value_key]
    to_agg = pd.DataFrame(
        {
            by_id_key: joined.index,
            id_key: joined[id_key].values,
            value_key: point_values,
        }
    )
    ##
    aggregated = to_agg.groupby([by_id_key, id_key]).agg(agg_func).reset_index()

    # this is for only shapes in "by" that intersect with something in "value"
    obs_id_categorical_categories = by.index.tolist()
    obs_id_categorical = pd.Categorical(aggregated[by_id_key], categories=obs_id_categorical_categories)

    X = sparse.coo_matrix(
        (
            aggregated[value_key].values,
            (obs_id_categorical.codes, aggregated[id_key].cat.codes),
        ),
        shape=(len(obs_id_categorical.categories), len(joined[id_key].cat.categories)),
    ).tocsr()
    return ad.AnnData(
        X,
        obs=pd.DataFrame(index=pd.Categorical(obs_id_categorical_categories).categories),
        var=pd.DataFrame(index=joined[id_key].cat.categories),
        dtype=X.dtype,
    )
