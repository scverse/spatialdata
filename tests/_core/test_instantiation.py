import numpy as np

from spatialdata import SpatialData


def test_empty():
    SpatialData()
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)


def test_only_features():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    _ = SpatialData(tables={"tables": a})
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)


def test_only_regions():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    _ = SpatialData(points={"instances": a})
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)


def test_only_images():
    pass

    _ = SpatialData(images={"image": np.ones((20, 10))})
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)


def test_only_points():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    _ = SpatialData(points={"points": a})
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)


def test_multi_layer_object():
    from anndata import AnnData

    old_state = np.random.get_state()
    np.random.seed(0)

    def n():
        return np.random.randint(20, 40)

    def a():
        return AnnData(np.random.rand(n(), n()))

    def b():
        return np.random.rand(n(), n())

    _ = SpatialData(
        tables={"tables": a()},
        points={"regions0": a(), "regions1": a(), "regions2": a()},
        images={"image0": b(), "image1": b(), "image2": b()},
    )
    # TODO: for now repr removed cause it fails, should be updated and this checked again
    # print(sdata)

    np.random.set_state(old_state)
