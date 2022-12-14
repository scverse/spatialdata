from importlib.metadata import version

__version__ = version("spatialdata")

__all__ = [
    # --- from spaitaldata._core._spatialdata ---
    "SpatialData",
    # --- from spatialdata._core.transformations ---
    "Identity",
    # "MapIndex",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "Sequence",
    # "Displacements",
    # "Coordinates",
    # "VectorField",
    # "InverseOf",
    # "Bijection",
    # "ByDimension",
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
from spatialdata._core.transformations import (  # Bijection,; ByDimension,; Coordinates,; Displacements,; InverseOf,; MapIndex,; VectorField,
    Affine,
    Identity,
    MapAxis,
    Rotation,
    Scale,
    Sequence,
    Translation,
)
from spatialdata._io.read import read_zarr
