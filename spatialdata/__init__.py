from importlib.metadata import version

__version__ = version("spatialdata")

from spatialdata._core.spatialdata import SpatialData
from spatialdata._core.transform import Transform
