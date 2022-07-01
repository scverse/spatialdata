from importlib.metadata import version

__version__ = version("spatialdata")

from ._core import SpatialData
from ._core.transform import Transform
