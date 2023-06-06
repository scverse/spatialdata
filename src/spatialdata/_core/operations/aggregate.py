from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import anndata as ad
import dask as da
import dask.dataframe as ddf
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from scipy import sparse
from shapely import Point
from spatial_image import SpatialImage
from xrspatial import zonal_stats

from spatialdata._core.operations.transform import transform
from spatialdata._core.query._utils import circles_to_polygons
from spatialdata._core.query.relational_query import get_values
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    get_model,
)
from spatialdata.transformations import BaseTransformation, Identity, get_transformation

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = ["aggregate"]


def aggregate(
    values_sdata: Optional[SpatialData] = None,
    values: Optional[ddf.DataFrame | gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage | str] = None,
    by: Optional[gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage] = None,
    value_key: list[str] | str | None = None,
    agg_func: str | list[str] = "mean",
    target_coordinate_system: str = "global",
    fractions: bool = False,
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
    fractions
        Adjusts for partial areas overlap between regions in values and by.
        More precisely: in the case in which a region in by partially overlaps with a region in values, this setting
        specifies whether the value to aggregate should be consider as it is (fractions = False) or is it to be
        multiplied by the following ratio: "area of the intersection between the two regions" / "area of the region in
        values".

        Additional details:

             - default is fractions = False.
             - when aggregating points this values shuold be left to False, as the points don't have area, thus
             otherwise a table of zeros will be obtained;
             - for categorical values count and sum are equivalent when fractions = False, but when fractions = True
             count and sum are different: count behaves like if fractions = False, while sum actually sums the values
             of the intersecting regions, and should thus be used.

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
    assert by is not None
    if not ((values_sdata is not None and isinstance(values, str)) ^ (not isinstance(values, str))):
        raise ValueError(
            "To specify the spatial element element with the values to aggregate, please do one of the following: "
            "- either pass a SpatialElement to the `values` parameter (and keep `values_sdata` = None);"
            "- either `values_sdata` needs to be a SpatialData object, and `values` needs to be the string nane of "
            "the element."
        )
    if values_sdata is not None:
        assert isinstance(values, str)
        values = values_sdata[values]
    assert values is not None

    if id(values) == id(by):
        # this case breaks the groupy aggregation in _aggregate_shapes(), probably a non relavant edge case so
        # skipping it for now
        raise NotImplementedError(
            "Aggregating an element by itself is not currenlty supported. If you have an use case for this please open "
            "an issue and we will implement this case."
        )

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
    if by_type is ShapesModel and values_type in [PointsModel, ShapesModel]:
        # Default value for value_key is ATTRS_KEY for values (if present)
        if values_type is PointsModel:
            assert isinstance(values, DaskDataFrame)
            if value_key is None and PointsModel.ATTRS_KEY in values.attrs:
                value_key = values.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY]

        # if value_key is not specified, add a columns on ones
        ONES_KEY = None
        if value_key is None:
            ONES_KEY = "__ones_column_aggregate"
            assert (
                ONES_KEY not in values.columns
            ), f"Column {ONES_KEY} is reserved for internal use and cannot be already present in values"
            values[ONES_KEY] = 1
            value_key = ONES_KEY

        out = _aggregate_shapes(
            values=values, by=by, values_sdata=values_sdata, value_key=value_key, agg_func=agg_func, fractions=fractions
        )

        # eventually remove the colum of ones if it was added
        if ONES_KEY is not None:
            del values[ONES_KEY]
        return out

    if by_type is Labels2DModel and values_type is Image2DModel:
        if fractions is True:
            raise NotImplementedError("fractions = True is not yet supported for raster aggregation")
        return _aggregate_image_by_labels(values=values, by=by, agg_func=agg_func, **kwargs)
    raise NotImplementedError(f"Cannot aggregate {values_type} by {by_type}")


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
    values_sdata: Optional[SpatialData] = None,
    value_key: str | list[str] | None = None,
    agg_func: str | list[str] = "count",
    fractions: bool = False,
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
    from spatialdata.models import points_dask_dataframe_to_geopandas

    assert value_key is not None
    if values_sdata is not None:
        element_name = values_sdata._locate_spatial_element(values)[0]
        actual_values = get_values(value_key=value_key, sdata=values_sdata, element_name=element_name)
    else:
        actual_values = get_values(value_key=value_key, element=values)
    assert isinstance(actual_values, pd.DataFrame), f"Expected pd.DataFrame, got {type(actual_values)}"

    assert not isinstance(values, str)
    if isinstance(values, DaskDataFrame):
        values = points_dask_dataframe_to_geopandas(values, suppress_z_warning=True)
    elif isinstance(values, GeoDataFrame):
        values = circles_to_polygons(values)
    else:
        raise RuntimeError(f"Unsupported type {type(values)}, this is most likely due to a bug, please report.")
    by = circles_to_polygons(by)

    categorical = pd.api.types.is_categorical_dtype(actual_values.iloc[:, 0])

    ONES_COLUMN = "__ones_column"
    AREAS_COLUMN = "__areas_column"
    JOINED_AREAS_COLUMN = "__joined_areas_column"

    e = f"Column names {ONES_COLUMN} and {AREAS_COLUMN} are reserved for internal use. Please rename your columns."
    assert ONES_COLUMN not in by.columns, e
    assert AREAS_COLUMN not in by.columns, e
    assert ONES_COLUMN not in actual_values.columns, e
    assert AREAS_COLUMN not in actual_values.columns, e

    by[ONES_COLUMN] = 1
    by[AREAS_COLUMN] = by.geometry.area
    values[ONES_COLUMN] = 1
    values[AREAS_COLUMN] = values.geometry.area

    if isinstance(value_key, str):
        value_key = [value_key]
    # either all the values of value_key are in the GeoDataFrame values, either none of them are (and in such a case
    # value_key refer to either obs or var of the table)
    assert not (any(vk in values.columns for vk in value_key) and not all(vk in values.columns for vk in value_key))
    # if the values of value_key are from the table, add them to values, but then remove them after the aggregation is
    # done
    to_remove = []
    for vk in value_key:
        if vk not in values.columns:
            s = actual_values[vk]
            s.index = values.index
            values[vk] = s
            to_remove.append(vk)

    ##
    by["__index"] = by.index
    values["__index"] = values.index

    # when values are points, we need to use sjoin(); when they are polygons and fractions is True, we need to use
    # overlay() also, we use sjoin() when fractions is False and values are polygons, because they are equivalent and
    # I think that sjoin() is faster
    if fractions is False or isinstance(values.iloc[0].geometry, Point):
        joined = by.sjoin(values)

        assert "__index" not in joined
        joined["__index"] = joined.index
        joined[JOINED_AREAS_COLUMN] = joined.geometry.area
    else:
        overlayed = gpd.overlay(by, values, how="intersection")

        assert "__index" not in overlayed
        overlayed.rename(
            columns={
                "__index_1": "__index_left",
                "__index_2": "__index_right",
                ONES_COLUMN + "_1": ONES_COLUMN + "_left",
                ONES_COLUMN + "_2": ONES_COLUMN + "_right",
                AREAS_COLUMN + "_1": AREAS_COLUMN + "_left",
                AREAS_COLUMN + "_2": AREAS_COLUMN + "_right",
            },
            inplace=True,
        )
        overlayed["__index"] = overlayed["__index_left"]
        overlayed[JOINED_AREAS_COLUMN] = overlayed.geometry.area

        joined = overlayed
    ##
    # with pd.option_context(
    #     "display.max_rows",
    #     None,
    #     "display.max_columns",
    #     None,
    #     "display.precision",
    #     3,
    # ):
    #     print("joined:")
    #     print(joined)
    #     print("overlayed:")
    #     print(overlayed)
    ##
    if categorical:
        # we only allow the aggregation of one categorical column at the time, because each categorical column would
        # give a different table as result of the aggregation, and we only support single tables
        assert len(value_key) == 1
        vk = value_key[0]
        aggregated = joined.groupby(["__index", vk])[ONES_COLUMN + "_right"].agg(agg_func).reset_index()
        aggregated_values = aggregated[ONES_COLUMN + "_right"].values
        # joined.groupby([joined.index, vk])[[ONES_COLUMN + '_right', AREAS_COLUMN + '_right']].agg("sum")
    else:
        ##
        # agg_func
        # joined
        # by
        # values
        # joined.iloc[3]
        # overlayed.iloc[3]
        ##
        aggregated = joined.groupby(["__index"])[value_key].agg(agg_func).reset_index()
        aggregated_values = aggregated[value_key].values
        # joined.groupby([joined.index, vk])[[ONES_COLUMN + '_right', AREAS_COLUMN + '_right']].agg("sum")

    # Here we prepare some variables to construct a sparse matrix in the coo format (edges + nodes)
    rows_categories = by.index.tolist()
    indices_of_aggregated_rows = np.array(aggregated["__index"])
    # In the categorical case len(value_key) == 1 so np.repeat does nothing, in the non-categorical case len(value_key)
    # can be > 1, so the aggregated table len(value_key) columns. This table will be flattened to a vector, so when
    # constructing the sparse matrix we need to repeat each row index len(value_key) times.
    # Example: the rows indices [0, 1, 2] in the case of len(value_key) == 2, will become [0, 0, 1, 1, 2, 2]
    indices_of_aggregated_rows = np.repeat(indices_of_aggregated_rows, len(value_key))
    rows_nodes = pd.Categorical(indices_of_aggregated_rows, categories=rows_categories, ordered=True)
    if categorical:
        columns_categories = values[vk].cat.categories.tolist()
        columns_nodes = pd.Categorical(aggregated[vk], categories=columns_categories, ordered=True)
    else:
        columns_categories = value_key
        numel = np.prod(aggregated_values.shape)
        assert numel % len(columns_categories) == 0
        columns_nodes = pd.Categorical(
            columns_categories * (numel // len(columns_categories)), categories=columns_categories
        )

    ##
    X = sparse.coo_matrix(
        (
            aggregated_values.ravel(),
            (rows_nodes.codes, columns_nodes.codes),
        ),
        shape=(len(rows_categories), len(columns_categories)),
    ).tocsr()

    # print(X.todense())

    ##
    anndata = ad.AnnData(
        X,
        obs=pd.DataFrame(index=rows_categories),
        var=pd.DataFrame(index=columns_categories),
        dtype=X.dtype,
    )
    # print(anndata)
    # print(anndata.obs_names)
    # print(anndata.var_names)
    # print(anndata.X.todense())
    ##

    # cleanup: remove columns previously added
    by.drop(columns=[ONES_COLUMN, AREAS_COLUMN], inplace=True)
    values.drop(columns=[ONES_COLUMN, AREAS_COLUMN] + to_remove, inplace=True)

    return anndata
