import numpy as np

import spatialdata


def test_package_has_version():
    spatialdata.__version__


def test_instantiation():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    _ = spatialdata.SpatialData(a)
