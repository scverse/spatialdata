from __future__ import annotations

import anndata as ad
import dask.dataframe as ddf
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

import spatialdata as sd
from spatialdata._core.models import PointsModel, ShapesModel, get_schema


def aggregate(
    values: ddf.DataFrame | gpd.GeoDataFrame,
    by: gpd.GeoDataFrame,
    id_key: str,
    *,
    value_key: str | None = None,
    agg_func: str = "mean",
) -> ad.AnnData:
    """
    Aggregate values by given shapes

    Parameters
    ----------
    values
        Values to aggregate
    by
        Shapes to aggregate by
    id_key
        Key to group values by. This is the key in the shapes dataframe.
    value_key
        Key to aggregate values by. This is the key in the values dataframe.
        If nothing is passed here, assumed to be 1.
    agg_func
        Aggregation function to apply over point values, e.g. "mean", "sum", "count".
        Passed to pandas.DataFrame.groupby.agg.
    """
    # TODO: Check that values are in the same space
    # Dispatch
    by_type = get_schema(by)
    values_type = get_schema(values)
    if by_type is ShapesModel:
        if values_type is PointsModel:
            return _aggregate_points_by_shapes(values, by, id_key, value_key=value_key, agg_func=agg_func)
        elif values_type is ShapesModel:
            return _aggregate_shapes_by_shapes(values, by, id_key, value_key=value_key, agg_func=agg_func)
        else:
            raise NotImplementedError(f"Cannot aggregate {values_type} by {by_type}")
    else:
        raise NotImplementedError(f"Cannot aggregate {values_type} by {by_type}")


def _aggregate_points_by_shapes(
    points: ddf.DataFrame | pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    id_key: str,
    *,
    value_key: str | None = None,
    agg_func: str = "count",
) -> ad.AnnData:
    # Have to get dims on dask dataframe, can't get from pandas
    dims = sd.get_dims(points)
    if isinstance(points, ddf.DataFrame):
        points = points.compute()
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(*[points[dim] for dim in dims]))
    return _aggregate(points, shapes, id_key, value_key, agg_func)


def _aggregate_shapes_by_shapes(
    values: gpd.GeoDataFrame,
    by: gpd.GeoDataFrame,
    id_key: str,
    *,
    value_key: str | None = None,
    agg_func: str = "count",
) -> ad.AnnData:
    def circles_to_polygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # We should only be buffering points, not polygons. Unfortunately this is an expensive check.
        values_geotypes = list(values.geom_type.unique())
        if values_geotypes == ["Point"]:
            df = df.buffer(df[ShapesModel.RADIUS_KEY])
        elif "Point" in values_geotypes:
            raise TypeError("Geometry contained shapes and polygons.")
        return df

    values = circles_to_polygons(values)
    by = circles_to_polygons(by)

    return _aggregate(values, by, id_key, value_key, agg_func)


def _aggregate(
    value: gpd.GeoDataFrame,
    by: gpd.GeoDataFrame,
    id_key: str,
    value_key: str | None = None,
    agg_func: str = "count",
) -> ad.AnnData:
    """
    Aggregate points by polygons.

    Params
    ------
    value
        GeoDataFrame of points to aggregate
    by
        GeoDataFrame of polygons
    id_key
        Column in values that indicate value type, e.g. probe id or organelle type
    value_key
        Column in values that indicate point value, e.g. probe intensity. If not provided, uses a fill value of 1.
    agg_func
        Aggregation function to apply over point values, e.g. "mean", "sum", "count".
        Passed to pandas.DataFrame.groupby.agg.
    """
    assert pd.api.types.is_categorical_dtype(value[id_key]), f"{id_key} must be categorical"

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

    aggregated = to_agg.groupby([by_id_key, id_key]).agg(agg_func).reset_index()
    obs_id_categorical = pd.Categorical(aggregated[by_id_key])

    X = sparse.coo_matrix(
        (
            aggregated[value_key].values,
            (obs_id_categorical.codes, aggregated[id_key].cat.codes),
        ),
        shape=(len(obs_id_categorical.categories), len(joined[id_key].cat.categories)),
    ).tocsr()

    adata = ad.AnnData(
        X,
        obs=pd.DataFrame(index=obs_id_categorical.categories),
        var=pd.DataFrame(index=joined[id_key].cat.categories),
    )

    return adata
