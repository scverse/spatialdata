from importlib.metadata import version

__version__ = version("spatialdata")

# Forcing usage of shapely 2.0 by geopandas
# https://geopandas.org/en/stable/getting_started/install.html#using-the-optional-pygeos-dependency
from ._compat import _check_geopandas_using_shapely

_check_geopandas_using_shapely()


__all__ = [
    # --- from spaitaldata._core._spatialdata ---
    "SpatialData",
    # --- from spatialdata._core.transformations ---
    "NgffIdentity",
    # "MapIndex",
    "NgffMapAxis",
    "NgffTranslation",
    "NgffScale",
    "NgffAffine",
    "NgffRotation",
    "NgffSequence",
    # "Displacements",
    # "Coordinates",
    # "VectorField",
    # "InverseOf",
    # "Bijection",
    "NgffByDimension",
    # --- from spatialdata._core.models ---
    "Image3DModel",
    "Image2DModel",
    "Labels2DModel",
    "Labels3DModel",
    "PointsModel",
    "PolygonsModel",
    "ShapesModel",
    "TableModel",
    # --- from spatialdata._core.core_utils ---
    "SpatialElement",
    "get_transform",
    "set_transform",
    "get_dims",
    # --- from spatialdata._io ---
    "read_zarr",
]

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.core_utils import (
    SpatialElement,
    get_dims,
    get_transform,
    set_transform,
)
from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    PolygonsModel,
    ShapesModel,
    TableModel,
)
from spatialdata._core.ngff.ngff_transformations import (  # Bijection,; NgffByDimension,; Coordinates,; Displacements,; InverseOf,; MapIndex,; VectorField,
    NgffAffine,
    NgffByDimension,
    NgffIdentity,
    NgffMapAxis,
    NgffRotation,
    NgffScale,
    NgffSequence,
    NgffTranslation,
)
from spatialdata._io.read import read_zarr
