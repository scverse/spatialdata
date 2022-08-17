from importlib.metadata import version

__version__ = version("spatialdata")

__all__ = ["SpatialData", "Transform"]

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.transform import Transform
