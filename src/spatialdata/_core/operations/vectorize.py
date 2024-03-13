from __future__ import annotations

from functools import singledispatch

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage

from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.aggregate import aggregate
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels3DModel,
    ShapesModel,
    SpatialElement,
    get_axes_names,
    get_model,
)
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import Identity

INTRINSIC_COORDINATE_SYSTEM = "__intrinsic"


@singledispatch
def to_circles(
    data: SpatialElement,
) -> GeoDataFrame:
    """
    Convert a set of geometries (2D/3D labels, 2D shapes) to approximated circles/spheres.

    Parameters
    ----------
    data
        The SpatialElement representing the geometries to approximate as circles/spheres.

    Returns
    -------
    The approximated circles/spheres.

    Notes
    -----
    The approximation is done by computing the centroids and the area/volume of the geometries. The geometries are then
    replaced by circles/spheres with the same centroids and area/volume.
    """
    raise RuntimeError(f"Unsupported type: {type(data)}")


@to_circles.register(SpatialImage)
@to_circles.register(MultiscaleSpatialImage)
def _(
    element: SpatialImage | MultiscaleSpatialImage,
) -> GeoDataFrame:
    model = get_model(element)
    if model in (Image2DModel, Image3DModel):
        raise RuntimeError("Cannot apply to_circles() to images.")
    if model == Labels3DModel:
        raise RuntimeError("to_circles() is not yet implemented for 3D labels.")

    # reduce to the single scale case
    if isinstance(element, MultiscaleSpatialImage):
        element_single_scale = SpatialImage(element["scale0"].values().__iter__().__next__())
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
def _(
    element: GeoDataFrame,
) -> GeoDataFrame:
    if isinstance(element.geometry.iloc[0], (Polygon, MultiPolygon)):
        radius = np.sqrt(element.geometry.area / np.pi)
        centroids = _get_centroids(element)
        obs = pd.DataFrame({"radius": radius})
        obs = pd.merge(obs, centroids, left_index=True, right_index=True, how="inner")
        return _make_circles(element, obs)
    assert isinstance(element.geometry.iloc[0], Point), (
        f"Unsupported geometry type: " f"{type(element.geometry.iloc[0])}"
    )
    return element


def _get_centroids(element: SpatialElement) -> pd.DataFrame:
    d = get_transformation(element, get_all=True)
    assert isinstance(d, dict)
    if INTRINSIC_COORDINATE_SYSTEM in d:
        raise RuntimeError(f"The name {INTRINSIC_COORDINATE_SYSTEM} is reserved.")
    d[INTRINSIC_COORDINATE_SYSTEM] = Identity()
    centroids = get_centroids(element, coordinate_system=INTRINSIC_COORDINATE_SYSTEM).compute()
    del d[INTRINSIC_COORDINATE_SYSTEM]
    return centroids


def _make_circles(element: SpatialImage | MultiscaleSpatialImage | GeoDataFrame, obs: pd.DataFrame) -> GeoDataFrame:
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


# TODO: depending of the implementation, add a parameter to control the degree of approximation of the constructed
# polygons/multipolygons
@singledispatch
def to_polygons(
    data: SpatialElement,
    target_coordinate_system: str,
) -> GeoDataFrame:
    """
    Convert a set of geometries (2D labels, 2D shapes) to approximated 2D polygons/multypolygons.

    Parameters
    ----------
    data
        The SpatialElement representing the geometries to approximate as 2D polygons/multipolygons.
    target_coordinate_system
        The coordinate system to which the geometries to consider should be transformed.

    Returns
    -------
    The approximated 2D polygons/multipolygons in the specified coordinate system.
    """
    raise RuntimeError("Unsupported type: {type(data)}")
