# type: ignore
"""Benchmarks for import times of the spatialdata package and its submodules.

Each ``timeraw_*`` function returns a snippet of Python code that asv runs in
a fresh interpreter, so the measured time reflects a cold import with an empty
module cache.
"""

from collections.abc import Callable


def _timeraw(func: Callable) -> Callable:
    """Set asv benchmark attributes for a cold-import timeraw function."""
    func.repeat = 5  # number of independent subprocess measurements
    func.number = 1  # must be 1: second import in same process hits module cache
    return func


@_timeraw
def timeraw_import_spatialdata():
    """Time a bare ``import spatialdata``."""
    return """
    import spatialdata
    """


@_timeraw
def timeraw_import_SpatialData():
    """Time importing the top-level ``SpatialData`` class."""
    return """
    from spatialdata import SpatialData
    """


@_timeraw
def timeraw_import_read_zarr():
    """Time importing ``read_zarr`` from the top-level namespace."""
    return """
    from spatialdata import read_zarr
    """


@_timeraw
def timeraw_import_models_elements():
    """Time importing the main element model classes."""
    return """
    from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
    """


@_timeraw
def timeraw_import_transformations():
    """Time importing the ``spatialdata.transformations`` submodule."""
    return """
    from spatialdata.transformations import Affine, Scale, Translation, Sequence
    """
