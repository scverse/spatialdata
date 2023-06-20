from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

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
    SpatialElement,
    TableModel,
    get_model,
)
from spatialdata.transformations import BaseTransformation, Identity, get_transformation

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = ["aggregate"]


def _parse_element(element: str | SpatialElement, sdata: SpatialData | None, str_for_exception: str) -> SpatialElement:
    if not ((sdata is not None and isinstance(element, str)) ^ (not isinstance(element, str))):
        raise ValueError(
            f"To specify the {str_for_exception!r} SpatialElement, please do one of the following: "
            f"- either pass a SpatialElement to the {str_for_exception!r} parameter (and keep "
            f"`{str_for_exception}_sdata` = None);"
            f"- either `{str_for_exception}_sdata` needs to be a SpatialData object, and {str_for_exception!r} needs "
            f"to be the string name of the element."
        )
    if sdata is not None:
        assert isinstance(element, str)
        return sdata[element]
    assert element is not None
    return element


def aggregate(
    values: ddf.DataFrame | gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage | str,
    by: gpd.GeoDataFrame | SpatialImage | MultiscaleSpatialImage | str,
    values_sdata: SpatialData | None = None,
    by_sdata: SpatialData | None = None,
    value_key: list[str] | str | None = None,
    agg_func: str | list[str] = "sum",
    target_coordinate_system: str = "global",
    fractions: bool = False,
    region_key: str = "region",
    instance_key: str = "instance_id",
    deepcopy: bool = True,
    **kwargs: Any,
) -> SpatialData:
    """
    Aggregate values by given region.

    Parameters
    ----------
    values_sdata
        SpatialData object containing the values to aggregate: if `None`, `values` must be a SpatialElement; if not
        `None`, `values` must be a string.
    values
        The values to aggregate: if `values_sdata` is `None`, must be a SpatialElement, otherwise must be a string
        specifying the name of the SpatialElement in `values_sdata`
    by_sdata
        Regions to aggregate by: if `None`, `by` must be a SpatialElement; if not `None`, `by` must be a string.
    by
        The regions to aggregate by: if `by_sdata` is None, must be a SpatialElement, otherwise must be a string
        specifying the name of the SpatialElement in `by_sdata`
    value_key
        Name (or list of names) of the columns containing the values to aggregate; can refer both to numerical or
        categorical values. If the values are categorical, `value_key` can't be a list.

        The key can be:

             - the name of a column(s) in the dataframe (Dask `DataFrame` for points or `GeoDataFrame` for shapes);
             - the name of obs column(s) in the associated `AnnData` table (for shapes and labels);
             - the name of a var(s), referring to the column(s) of the X matrix in the table (for shapes and labels).

        If nothing is passed here, it defaults to the equivalent of a column of ones.
        Defaults to `FEATURE_KEY` for points (if present).
    agg_func
        Aggregation function to apply over point values, e.g. `"mean"`, `"sum"`, `"count"`.
        Passed to :func:`pandas.DataFrame.groupby.agg` or to :func:`xrspatial.zonal_stats`
        according to the type of `values`.
    target_coordinate_system
        Coordinate system to transform to before aggregating.
    fractions
        Adjusts for partial areas overlap between regions in `values` and `by`.
        More precisely: in the case in which a region in `by` partially overlaps with a region in `values`, this setting
        specifies whether the value to aggregate should be considered as it is (`fractions = False`) or it is to be
        multiplied by the following ratio: "area of the intersection between the two regions" / "area of the region in
        `values`".

        Additional details:

             - default is `fractions = False`.
             - when aggregating points this parameter must be left to `False`, as the points don't have area (otherwise
                 a table of zeros would be obtained);
             - for categorical values `"count"` and `"sum"` are equivalent when `fractions = False`, but when
                `fractions = True`, `"count"` and `"sum"` are different: `count` would give not meaningful results and
                so it's not allowed, while `"sum"` actually sums the values of the intersecting regions, and should
                therefore be used.
             - aggregating categorical values with `agg_func = "mean"` is not allowed as it give not meaningful results.

    region_key
        Name that will be given to the new region column in the returned aggregated table.
    instance_key
        Name that will be given to the new instance id column in the returned aggregated table.
    deepcopy
        Whether to deepcopy the shapes in the returned `SpatialData` object. If the shapes are large (e.g. large
        multiscale labels), you may consider disabling the deepcopy to use a lazy Dask representation.
    kwargs
        Additional keyword arguments to pass to :func:`xrspatial.zonal_stats`.

    Returns
    -------
    Returns a `SpatialData` object with the `by` shapes as SpatialElement and a table with the aggregated values
    annotating the shapes.

    If `value_key` refers to a categorical variable, the table in the `SpaitalData` object has shape
    (`by.shape[0]`, <n categories>).

    Notes
    -----
    This function returns a `SpatialData` object, so to access the aggregated table you can use the `table` attribute`.

    The shapes in the returned `SpatialData` objects are a reference to the original one. If you want them to be a
    different object you can do a deepcopy manually (this loads the data into memory), or you can save the `SpatialData`
    object to disk and reload it (this keeps the data lazily represented).

    When aggregation points by shapes, the current implementation loads all the points into memory and thus could lead
    to a large memory usage. This Github issue https://github.com/scverse/spatialdata/issues/210 keeps track of the
    changes required to address this behavior.
    """
    values_ = _parse_element(element=values, sdata=values_sdata, str_for_exception="values")
    by_ = _parse_element(element=by, sdata=by_sdata, str_for_exception="by")

    if values_ is by_:
        # this case breaks the groupy aggregation in _aggregate_shapes(), probably a non relevant edge case so
        # skipping it for now
        raise NotImplementedError(
            "Aggregating an element by itself is not currenlty supported. If you have an use case for this please open "
            "an issue and we will implement this case."
        )

    # get schema
    by_type = get_model(by_)
    values_type = get_model(values_)

    # get transformation between coordinate systems
    by_transform: BaseTransformation = get_transformation(by_, target_coordinate_system)  # type: ignore[assignment]
    values_transform: BaseTransformation = get_transformation(
        values_,
        target_coordinate_system,  # type: ignore[assignment]
    )
    if not (by_transform == values_transform and isinstance(values_transform, Identity)):
        by_ = transform(by_, by_transform)
        values_ = transform(values_, values_transform)

    # dispatch
    adata = None
    if by_type is ShapesModel and values_type in [PointsModel, ShapesModel]:
        # Default value for value_key is ATTRS_KEY for values_ (if present)
        if values_type is PointsModel:
            assert isinstance(values_, DaskDataFrame)
            if value_key is None and PointsModel.ATTRS_KEY in values_.attrs:
                value_key = values_.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY]

        # if value_key is not specified, add a columns on ones
        ONES_KEY = None
        if value_key is None:
            ONES_KEY = "__ones_column_aggregate"
            assert (
                ONES_KEY not in values_.columns
            ), f"Column {ONES_KEY} is reserved for internal use and cannot be already present in values_"
            values_[ONES_KEY] = 1
            value_key = ONES_KEY

        adata = _aggregate_shapes(
            values=values_,
            by=by_,
            values_sdata=values_sdata,
            values_element_name=values if isinstance(values, str) else None,
            value_key=value_key,
            agg_func=agg_func,
            fractions=fractions,
        )

        # eventually remove the colum of ones if it was added
        if ONES_KEY is not None:
            del values_[ONES_KEY]

    if by_type is Labels2DModel and values_type is Image2DModel:
        if fractions is True:
            raise NotImplementedError("fractions = True is not yet supported for raster aggregation")
        adata = _aggregate_image_by_labels(values=values_, by=by_, agg_func=agg_func, **kwargs)

    if adata is None:
        raise NotImplementedError(f"Cannot aggregate {values_type} by {by_type}")

    # create a SpatialData object with the aggregated table and the "by" shapes
    shapes_name = by if isinstance(by, str) else "by"
    return _create_sdata_from_table_and_shapes(
        table=adata,
        shapes_name=shapes_name,
        shapes=by_,
        region_key=region_key,
        instance_key=instance_key,
        deepcopy=deepcopy,
    )


def _create_sdata_from_table_and_shapes(
    table: ad.AnnData,
    shapes: GeoDataFrame | SpatialImage | MultiscaleSpatialImage,
    shapes_name: str,
    region_key: str,
    instance_key: str,
    deepcopy: bool,
) -> SpatialData:
    from spatialdata import SpatialData
    from spatialdata._utils import _deepcopy_geodataframe

    table.obs[instance_key] = table.obs_names.copy()
    table.obs[region_key] = shapes_name
    table = TableModel.parse(table, region=shapes_name, region_key=region_key, instance_key=instance_key)

    # labels case, needs conversion from str to int
    if isinstance(shapes, (SpatialImage, MultiscaleSpatialImage)):
        table.obs[instance_key] = table.obs[instance_key].astype(int)

    if deepcopy:
        shapes = _deepcopy_geodataframe(shapes)

    return SpatialData.from_elements_dict({shapes_name: shapes, "": table})


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
    values_sdata: SpatialData | None = None,
    values_element_name: str | None = None,
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
    values_sdata
        Optional SpatialData object containing the table annotating the values. Note: when values is transformed (
        because by and values are not in the same coordinate system), value_sdata will not contain the transformed
        values. This is why we can't infer the element name from values_sdata (using
        values_sdata._locate_spatial_element(values)[0]) and we need to pass it explicitly.
    values_element_name
        Name of the element in values_sdata that contains the values. If values_sdata is None, this is not needed.
    by
        Geopandas dataframe to group values by. Must have a geometry column.
    value_key
        Column in value dataframe to perform aggregation on.
    agg_func
        Aggregation function to apply over grouped values. Passed to pandas.DataFrame.groupby.agg.
    """
    from spatialdata.models import points_dask_dataframe_to_geopandas

    assert value_key is not None
    assert (values_sdata is None) == (values_element_name is None)
    if values_sdata is not None:
        actual_values = get_values(value_key=value_key, sdata=values_sdata, element_name=values_element_name)
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

    # deal with edge cases
    if fractions:
        assert not (categorical and agg_func == "count"), (
            "Aggregating categorical values using fractions=True and agg_func='count' will most likely give "
            "not meaningful results. Please consider using a different aggregation function, for instance "
            "agg_func='sum' instead."
        )
        assert not isinstance(values.iloc[0].geometry, Point), (
            "Fractions cannot be computed when values are points. " "Please use fractions=False."
        )
    assert not (categorical and agg_func == "mean"), (
        "Incompatible choice: aggregating a categorical column with " "agg_func='mean'"
    )

    # we need to add a column of ones to the values dataframe to be able to count the number of instances in each zone
    # and we need to add a column with the area of each zone to be able to compute the fractions describing the overlaps
    # between what is being aggregated and in the "by" regions, where the denominators of these fractions are the areas
    # of the regions in "values"
    ONES_COLUMN = "__ones_column"
    AREAS_COLUMN = "__areas_column"

    e = f"Column names {ONES_COLUMN} and {AREAS_COLUMN} are reserved for internal use. Please rename your columns."
    assert ONES_COLUMN not in by.columns, e
    assert AREAS_COLUMN not in by.columns, e
    assert ONES_COLUMN not in actual_values.columns, e

    values[ONES_COLUMN] = 1
    values[AREAS_COLUMN] = values.geometry.area

    INDEX = "__index"
    assert INDEX not in by, f"{INDEX} is a reserved column name"
    assert "__index" not in values, f"{INDEX} is a reserved column name"

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

    # when values are points, we need to use sjoin(); when they are polygons and fractions is True, we need to use
    # overlay() also, we use sjoin() when fractions is False and values are polygons, because they are equivalent and
    # I think that sjoin() is faster
    if fractions is False or isinstance(values.iloc[0].geometry, Point):
        joined = by.sjoin(values)

        assert INDEX not in joined
        joined[INDEX] = joined.index
    else:
        by[INDEX] = by.index
        overlayed = gpd.overlay(by, values, how="intersection")
        del by[INDEX]
        joined = overlayed
    assert INDEX in joined

    fractions_of_values = None
    if fractions:
        fractions_of_values = joined.geometry.area / joined[AREAS_COLUMN]

    if categorical:
        # we only allow the aggregation of one categorical column at the time, because each categorical column would
        # give a different table as result of the aggregation, and we only support single tables
        assert len(value_key) == 1
        vk = value_key[0]
        if fractions_of_values is not None:
            joined[ONES_COLUMN] = fractions_of_values
        aggregated = joined.groupby([INDEX, vk])[ONES_COLUMN].agg(agg_func).reset_index()
        aggregated_values = aggregated[ONES_COLUMN].values
    else:
        if fractions_of_values is not None:
            joined[value_key] = joined[value_key].to_numpy() * fractions_of_values.to_numpy().reshape(-1, 1)
        aggregated = joined.groupby([INDEX])[value_key].agg(agg_func).reset_index()
        aggregated_values = aggregated[value_key].values

    # Here we prepare some variables to construct a sparse matrix in the coo format (edges + nodes)
    rows_categories = by.index.tolist()
    indices_of_aggregated_rows = np.array(aggregated[INDEX])
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

    X = sparse.coo_matrix(
        (
            aggregated_values.ravel(),
            (rows_nodes.codes, columns_nodes.codes),
        ),
        shape=(len(rows_categories), len(columns_categories)),
    ).tocsr()

    anndata = ad.AnnData(
        X,
        obs=pd.DataFrame(index=rows_categories),
        var=pd.DataFrame(index=columns_categories),
        dtype=X.dtype,
    )

    # cleanup: remove columns previously added
    values.drop(columns=[ONES_COLUMN, AREAS_COLUMN] + to_remove, inplace=True)

    return anndata
