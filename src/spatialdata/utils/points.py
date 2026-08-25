from __future__ import annotations

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame

from spatialdata.models import PointsModel


def _make_points(coordinates: np.ndarray) -> DaskDataFrame:
    """Helper function to make a Points element."""  # noqa: D401
    k0 = int(len(coordinates) / 3)
    k1 = len(coordinates) - k0
    genes = np.hstack((np.repeat("a", k0), np.repeat("b", k1)))
    return PointsModel.parse(coordinates, annotation=pd.DataFrame({"genes": genes}), feature_key="genes")
