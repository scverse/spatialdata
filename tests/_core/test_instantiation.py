import numpy as np

from spatialdata import SpatialData


def test_empty():
    sdata = SpatialData()
    print(sdata)


def test_only_features():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = SpatialData(tables=a)
    print(sdata)


def test_only_regions():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = SpatialData(points={"instances": a})
    print(sdata)


def test_only_images():
    pass

    sdata = SpatialData(images={"image": np.ones((20, 10))})
    print(sdata)


def test_only_points():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = SpatialData(points={"points": a})
    print(sdata)


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

    sdata = SpatialData(
        tables=a(),
        points={"regions0": a(), "regions1": a(), "regions2": a()},
        images={"image0": b(), "image1": b(), "image2": b()},
    )
    print(sdata)

    np.random.set_state(old_state)
