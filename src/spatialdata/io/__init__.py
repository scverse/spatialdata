"""Experimental bridge to the spatialdata_io package."""

try:
    from spatialdata_io import *  # noqa: F403
except ImportError as e:
    raise ImportError(
        "To access spatialdata.io, `spatialdata_io` must be installed, e.g. via `pip install spatialdata-io`. "
        f"Original exception: {e}"
    ) from e
