from functools import singledispatch
from typing import Optional, Union
from warnings import warn

import dask_image.ndinterp
import numpy as np
import pandas as pd
from dask.array.core import Array as DaskArray
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray
from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.aggregate import aggregate

from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata._utils import Number, _parse_list_into_array
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    ShapesModel,
    Labels2DModel,
    Labels3DModel,
    SpatialElement,
    get_axes_names,
    get_model,
)
from shapely import Polygon, MultiPolygon, Point
from spatialdata.models._utils import get_spatial_axes
from spatialdata.transformations._utils import _get_scale, compute_coordinates
from spatialdata.transformations.operations import (
    get_transformation,
    remove_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import (
    Affine,
    BaseTransformation,
    Scale,
    Sequence,
    Translation,
    Identity,
    _get_affine_for_element,
)


@singledispatch
def to_circles(
    data: Union[SpatialElement],
    target_coordinate_system: str,
) -> GeoDataFrame:
    """
    Convert a set of geometries (2D/3D labels, 2D shapes) to approximated circles/spheres.

    Parameters
    ----------
    data
        The SpatialElement representing the geometries to approximate as circles/spheres.
    target_coordinate_system
        The coordinate system to which the geometries to consider should be transformed.

    Returns
    -------
    GeoDataFrame
        The approximated circles/spheres in the specified coordinate system.

    Notes
    -----
    The approximation is done by computing the centroids and the area/volume of the geometries. The geometries are then
    replaced by circles/spheres with the same centroids and area/volume.
    """
    raise RuntimeError("Unsupported type: {type(data)}")

@to_circles.register(SpatialImage)
@to_circles.register(MultiscaleSpatialImage)
def _(
    element: SpatialImage,
    target_coordinate_system: str,
) -> GeoDataFrame:
    # find the area of labels, estimate the radius from it; find the centroids
    if isinstance(element, MultiscaleSpatialImage):
        shape = element["scale0"].values().__iter__().__next__().shape
    else:
        shape = element.shape

    axes = get_axes_names(element)
    if "z" in axes:
        model = Image3DModel
    else:
        model = Image2DModel
    ones = model.parse(np.ones((1,) + shape), dims=("c",) + axes)
    aggregated = aggregate(values=ones, by=element, agg_func="sum").table
    areas = aggregated.X.todense().A1.reshape(-1)
    aobs = aggregated.obs
    aobs["areas"] = areas
    aobs["radius"] = np.sqrt(areas / np.pi)

    # get the centroids; remove the background if present (the background is not considered during aggregation)
    centroids = get_centroids(element, coordinate_system=target_coordinate_system).compute()
    if 0 in centroids.index:
        centroids = centroids.drop(index=0)
    centroids.index = centroids.index
    aobs.index = aobs[instance_key]
    aobs.index.name = None
    assert len(aobs) == len(centroids)
    obs = pd.merge(aobs, centroids, left_index=True, right_index=True, how="inner")
    assert len(obs) == len(centroids)



@to_circles.register(GeoDataFrame)
def _(
    element: GeoDataFrame,
    target_coordinate_system: str,
) -> GeoDataFrame:
    if isinstance(element.geometry.iloc[0], (Polygon, MultiPolygon)):
        radius = np.sqrt(element.geometry.area / np.pi)
            centroids = get_centroids(element, coordinate_system=coordinate_system).compute()
            obs = pd.DataFrame({"radius": radius})
            obs = pd.merge(obs, centroids, left_index=True, right_index=True, how="inner")
        return _make_circles(element, obs, coordinate_system)
    else:
        return sdata[region_name]

def _make_circles(element: SpatialImage | MultiscaleSpatialImage | GeoDataFrame, obs: pd.DataFrame, coordinate_system: str) -> GeoDataFrame:
    spatial_axes = sorted(get_axes_names(element))
    centroids = obs[spatial_axes].values
    return ShapesModel.parse(
        centroids,
        geometry=0,
        index=obs.index,
        radius=obs["radius"].values,
        transformations={coordinate_system: Identity()},
    )

# TODO: depending of the implementation, add a parameter to control the degree of approximation of the constructed
# polygons/multipolygons
@singledispatch
def to_polygons(
    data: Union[SpatialElement],
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
    GeoDataFrame
        The approximated 2D polygons/multipolygons in the specified coordinate system.
    """
    raise RuntimeError("Unsupported type: {type(data)}")
