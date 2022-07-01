import spatialdata as sd
import numpy as np


def test_empty():
    sdata = sd.SpatialData()
    print(sdata)


def test_only_features():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = sd.SpatialData(a)
    print(sdata)


def test_only_regions():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = sd.SpatialData(regions={"instances": a})
    print(sdata)


def test_only_images():
    from anndata import AnnData

    sdata = sd.SpatialData(images={"image": np.ones((20, 10))})
    print(sdata)


def test_only_points():
    from anndata import AnnData

    a = AnnData(np.ones((20, 10)), dtype=np.float64)
    sdata = sd.SpatialData(points=a)
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

    sdata = sd.SpatialData(
        adata=a(),
        regions={"regions0": a(), "regions1": a(), "regions2": a()},
        images={"image0": b(), "image1": b(), "image2": b()},
        points=a(),
    )
    print(sdata)

    np.random.set_state(old_state)


if __name__ == "__main__":
    test_multi_layer_object()
