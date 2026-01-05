from functools import singledispatch
from typing import Any

import dask
import numpy as np
import pandas as pd
import shapely
import skimage.measure
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Point, Polygon
from skimage.measure._regionprops import RegionProperties
from xarray import DataArray, DataTree

from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels3DModel,
    ShapesModel,
    SpatialElement,
    get_axes_names,
    get_model,
)
from spatialdata.models._utils import points_dask_dataframe_to_geopandas
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import Identity

INTRINSIC_COORDINATE_SYSTEM = "__intrinsic"


@singledispatch
def to_circles(
    data: SpatialElement,
    radius: float | ArrayLike | None = None,
) -> GeoDataFrame:
    """
    Convert a set of geometries (2D/3D labels, 2D shapes) to approximated circles/spheres.

    Parameters
    ----------
    data
        The SpatialElement representing the geometries to approximate as circles/spheres.
    radius
        Radius/radii for the circles. For points elements, radius can either be specified as an argument, or be a column
         of the dataframe. For non-points elements, radius must be `None`.

    Returns
    -------
    The approximated circles/spheres.

    Notes
    -----
    The approximation is done by computing the centroids and the area/volume of the geometries. The geometries are then
    replaced by circles/spheres with the same centroids and area/volume.
    """
    raise RuntimeError(f"Unsupported type: {type(data)}")


@to_circles.register(DataArray)
@to_circles.register(DataTree)
def _(element: DataArray | DataTree, **kwargs: Any) -> GeoDataFrame:
    assert len(kwargs) == 0
    model = get_model(element)
    if model in (Image2DModel, Image3DModel):
        raise RuntimeError("Cannot apply to_circles() to images.")
    if model == Labels3DModel:
        raise RuntimeError("to_circles() is not supported for 3D labels.")

    # reduce to the single scale case
    if isinstance(element, DataTree):
        element_single_scale = element["scale0"].values().__iter__().__next__()
    else:
        element_single_scale = element
    shape = element_single_scale.shape

    # find the area of labels, estimate the radius from it; find the centroids
    axes = get_axes_names(element)
    model = Image3DModel if "z" in axes else Image2DModel
    ones = model.parse(np.ones((1,) + shape), dims=("c",) + axes)
    aggregated = aggregate(values=ones, by=element_single_scale, agg_func="sum")["table"]
    areas = aggregated.X.todense().A1.reshape(-1)
    aobs = aggregated.obs
    aobs["areas"] = areas
    aobs["radius"] = np.sqrt(areas / np.pi)

    # get the centroids; remove the background if present (the background is not considered during aggregation)
    centroids = _get_centroids(element)
    if 0 in centroids.index:
        centroids = centroids.drop(index=0)
    # instance_id is the key used by the aggregation APIs
    aobs.index = aobs["instance_id"]
    aobs.index.name = None
    assert len(aobs) == len(centroids)
    obs = pd.merge(aobs, centroids, left_index=True, right_index=True, how="inner")
    assert len(obs) == len(centroids)
    return _make_circles(element, obs)


@to_circles.register(GeoDataFrame)
def _(element: GeoDataFrame, **kwargs: Any) -> GeoDataFrame:
    assert len(kwargs) == 0
    if isinstance(element.geometry.iloc[0], Polygon | MultiPolygon):
        radius = np.sqrt(element.geometry.area / np.pi)
        centroids = _get_centroids(element)
        obs = pd.DataFrame({"radius": radius})
        obs = pd.merge(obs, centroids, left_index=True, right_index=True, how="inner")
        return _make_circles(element, obs)
    if isinstance(element.geometry.iloc[0], Point):
        return element
    raise RuntimeError(f"Unsupported geometry type: {type(element.geometry.iloc[0])}")


@to_circles.register(DaskDataFrame)
def _(element: DaskDataFrame, radius: float | ArrayLike | None = None) -> GeoDataFrame:
    gdf = points_dask_dataframe_to_geopandas(element)
    if ShapesModel.RADIUS_KEY in gdf.columns and radius is None:
        logger.info(f"{ShapesModel.RADIUS_KEY} already found in the object, ignoring the provided {radius} argument.")
    elif radius is None:
        raise RuntimeError(
            "When calling `to_circles()` on points, `radius` must either be provided, either be a column."
        )
    else:
        gdf[ShapesModel.RADIUS_KEY] = radius
    return gdf


def _get_centroids(element: SpatialElement) -> pd.DataFrame:
    d = get_transformation(element, get_all=True)
    assert isinstance(d, dict)
    if INTRINSIC_COORDINATE_SYSTEM in d:
        raise RuntimeError(f"The name {INTRINSIC_COORDINATE_SYSTEM} is reserved.")
    d[INTRINSIC_COORDINATE_SYSTEM] = Identity()
    centroids = get_centroids(element, coordinate_system=INTRINSIC_COORDINATE_SYSTEM).compute()
    del d[INTRINSIC_COORDINATE_SYSTEM]
    return centroids


def _make_circles(element: DataArray | DataTree | GeoDataFrame, obs: pd.DataFrame) -> GeoDataFrame:
    spatial_axes = sorted(get_axes_names(element))
    centroids = obs[spatial_axes].values
    transformations = get_transformation(element, get_all=True)
    assert isinstance(transformations, dict)
    return ShapesModel.parse(
        centroids,
        geometry=0,
        index=obs.index,
        radius=obs["radius"].values,
        transformations=transformations.copy(),
    )


@singledispatch
def to_polygons(data: SpatialElement, buffer_resolution: int | None = None) -> GeoDataFrame:
    """
    Convert a set of geometries (2D labels, 2D shapes) to approximated 2D polygons/multypolygons.

    For optimal performance when converting rasters (:class:`xarray.DataArray` or :class:`datatree.DataTree`)
    to polygons, it is recommended to configure `Dask` to use 'processes' rather than 'threads'.
    For example, you can set this configuration with:

    >>> import dask
    >>> dask.config.set(scheduler='processes')

    Parameters
    ----------
    data
        The SpatialElement representing the geometries to approximate as 2D polygons/multipolygons.
    buffer_resolution
        Used only when constructing polygons from circles. Value of the `resolution` parement for the `buffer()`
        internal call.

    Returns
    -------
    The approximated 2D polygons/multipolygons in the specified coordinate system.
    """
    raise RuntimeError(f"Unsupported type: {type(data)}")


@to_polygons.register(DataArray)
@to_polygons.register(DataTree)
def _(
    element: DataArray | DataArray | DataTree,
    **kwargs: Any,
) -> GeoDataFrame:
    assert len(kwargs) == 0
    model = get_model(element)
    if model in (Image2DModel, Image3DModel):
        raise RuntimeError("Cannot apply to_polygons() to images.")
    if model == Labels3DModel:
        raise RuntimeError("to_polygons() is not supported for 3D labels.")

    # reduce to the single scale case
    if isinstance(element, DataTree):
        element_single_scale = element["scale0"].values().__iter__().__next__()
    else:
        element_single_scale = element

    chunk_sizes = element_single_scale.data.chunks

    def _vectorize_chunk(chunk: np.ndarray, yoff: int, xoff: int) -> GeoDataFrame:  # type: ignore[type-arg]
        gdf = _vectorize_mask(chunk)
        gdf["chunk-location"] = f"({yoff}, {xoff})"
        gdf.geometry = gdf.translate(xoff, yoff)
        return gdf

    tasks = [
        dask.delayed(_vectorize_chunk)(chunk, sum(chunk_sizes[0][:iy]), sum(chunk_sizes[1][:ix]))
        for iy, row in enumerate(element_single_scale.data.to_delayed())
        for ix, chunk in enumerate(row)
    ]

    results = dask.compute(*tasks)
    gdf = pd.concat(results)
    gdf = GeoDataFrame([_dissolve_on_overlaps(*item) for item in gdf.groupby("label")], columns=["label", "geometry"])
    gdf.index = gdf["label"]

    transformations = get_transformation(element_single_scale, get_all=True)

    assert isinstance(transformations, dict)

    return ShapesModel.parse(gdf, transformations=transformations.copy())


def _region_props_to_polygon(region_props: RegionProperties) -> MultiPolygon | Polygon:
    mask = np.pad(region_props.image, 1)
    contours = skimage.measure.find_contours(mask, 0.5)

    invalid_polygons = any(contour.shape[0] <= 3 for contour in contours)
    if invalid_polygons:
        # this should not happen because even a single pixel should can be converted to a polygon
        raise RuntimeError("Invalid polygon found in the region properties.")
    polygons = [Polygon(contour[:, [1, 0]]) for contour in contours]

    polygon = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

    yoff, xoff, *_ = region_props.bbox
    return shapely.affinity.translate(polygon, xoff - 1, yoff - 1)  # account for the padding


def _vectorize_mask(
    mask: np.ndarray,  # type: ignore[type-arg]
) -> GeoDataFrame:
    if mask.max() == 0:
        return GeoDataFrame({"label": []}, geometry=[])

    regions = skimage.measure.regionprops(mask)

    return GeoDataFrame(
        {"label": [region.label for region in regions]},
        geometry=[_region_props_to_polygon(region) for region in regions],
    )


def _dissolve_on_overlaps(label: int, group: GeoDataFrame) -> GeoDataFrame:
    if len(group) == 1:
        return (label, group.geometry.iloc[0])
    if len(np.unique(group["chunk-location"])) == 1:
        return (label, MultiPolygon(list(group.geometry)))
    return (label, group.dissolve().geometry.iloc[0])


@to_polygons.register(GeoDataFrame)
def _(gdf: GeoDataFrame, buffer_resolution: int = 16) -> GeoDataFrame:
    if isinstance(gdf.geometry.iloc[0], Polygon | MultiPolygon):
        return gdf
    if isinstance(gdf.geometry.iloc[0], Point):
        ShapesModel.validate_shapes_not_mixed_types(gdf)
        if isinstance(gdf.geometry.iloc[0], Point):
            buffered_df = gdf.copy()
            buffered_df["geometry"] = buffered_df.apply(
                lambda row: row.geometry.buffer(row[ShapesModel.RADIUS_KEY], quad_segs=buffer_resolution), axis=1
            )

            # Ensure the GeoDataFrame recognizes the updated geometry column
            buffered_df = buffered_df.set_geometry("geometry")

            # TODO replace with a function to copy the metadata (the parser could also do this): https://github.com/scverse/spatialdata/issues/258
            buffered_df.attrs[ShapesModel.TRANSFORM_KEY] = gdf.attrs[ShapesModel.TRANSFORM_KEY]
            return buffered_df
        assert isinstance(gdf.geometry.iloc[0], Polygon | MultiPolygon)
        return gdf
    raise RuntimeError(f"Unsupported geometry type: {type(gdf.geometry.iloc[0])}")


@to_polygons.register(DaskDataFrame)
def _(element: DaskDataFrame, **kwargs: Any) -> None:
    assert len(kwargs) == 0
    raise RuntimeError(
        "Cannot convert points to polygons. To overcome this you can construct circles from points with `to_circles()` "
        "and then call `to_polygons()`."
    )
