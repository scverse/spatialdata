from __future__ import annotations

# Functions to compute the bounding box describing the extent of a SpatialElement or SpatialData object
from functools import singledispatch

from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.operations.transform import transform
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel, get_model
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

BoundingBoxDescription = dict[str, tuple[float, float]]


def _validate_coordinate_system(e: SpatialElement, coordinate_system: str) -> None:
    d = get_transformation(e, get_all=True)
    assert isinstance(d, dict)
    assert coordinate_system in d, (
        f"No transformation to coordinate system {coordinate_system} is available for the given element.\n"
        f"Available coordinate systems: {list(d.keys())}"
    )


@singledispatch
def get_centroids(
    e: SpatialElement,
    coordinate_system: str = "global",
) -> DaskDataFrame:
    """
    Get the centroids of the geometries contained in a SpatialElement, as a new Points element.

    Parameters
    ----------
    e
        The SpatialElement. Only points, shapes (circles, polygons and multipolygons) and labels are supported.
    coordinate_system
        The coordinate system in which the centroids are computed.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


@get_centroids.register(SpatialImage)
@get_centroids.register(MultiscaleSpatialImage)
def _(
    e: SpatialImage | MultiscaleSpatialImage,
    coordinate_system: str = "global",
) -> DaskDataFrame:
    model = get_model(e)
    if model in [Image2DModel, Image3DModel]:
        raise ValueError("Cannot compute centroids for images.")
    assert model in [Labels2DModel, Labels3DModel]

    _validate_coordinate_system(e, coordinate_system)


# def _get_extent_of_shapes(e: GeoDataFrame) -> BoundingBoxDescription:
#     # remove potentially empty geometries
#     e_temp = e[e["geometry"].apply(lambda geom: not geom.is_empty)]
#     assert len(e_temp) > 0, "Cannot compute extent of an empty collection of geometries."
#
#     # separate points from (multi-)polygons
#     first_geometry = e_temp["geometry"].iloc[0]
#     if isinstance(first_geometry, Point):
#         return _get_extent_of_circles(e)
#     assert isinstance(first_geometry, (Polygon, MultiPolygon))
#     return _get_extent_of_polygons_multipolygons(e)


@get_centroids.register(GeoDataFrame)
def _(e: GeoDataFrame, coordinate_system: str = "global") -> DaskDataFrame:
    _validate_coordinate_system(e, coordinate_system)


@get_centroids.register(DaskDataFrame)
def _(e: DaskDataFrame, coordinate_system: str = "global") -> DaskDataFrame:
    _validate_coordinate_system(e, coordinate_system)
    axes = get_axes_names(e)
    assert axes in [("x", "y"), ("x", "y", "z")]
    coords = e[list(axes)].compute().values
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    centroids = PointsModel.parse(coords, transformations={coordinate_system: t})
    return transform(centroids, to_coordinate_system=coordinate_system)
