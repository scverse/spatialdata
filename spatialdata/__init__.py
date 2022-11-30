from importlib.metadata import version

__version__ = version("spatialdata")

__all__ = [
    "SpatialData",
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
    "Image3DModel",
    "Image2DModel",
    "Labels2DModel",
    "Labels3DModel",
    "PointsModel",
    "PolygonsModel",
    "ShapesModel",
    "TableModel",
]

from spatialdata._core._spatialdata import SpatialData
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
