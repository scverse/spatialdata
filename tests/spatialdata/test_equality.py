import spatialdata as sd
import numpy as np
from anndata import AnnData


def get_empty_sdata():
    return sd.SpatialData()


def get_features_sdata(dim0=20):
    return sd.SpatialData(adata=AnnData(np.ones((dim0, 10))))


def get_regions_sdata(dim0=20):
    return sd.SpatialData(regions={"instances": AnnData(np.ones((dim0, 10)))})


def get_images_sdata(dim0=20):
    return sd.SpatialData(images={"image": np.ones((dim0, 10))})


def get_points_sdata(dim0=20):
    return sd.SpatialData(points=AnnData(np.ones((dim0, 10))))


def test_single_component_sdata():
    getters = [get_empty_sdata, get_features_sdata, get_regions_sdata, get_images_sdata, get_points_sdata]
    for i, g in enumerate(getters):
        for j, h in enumerate(getters):
            if i > j:
                continue
            if g == h:
                assert g() == h()
                if g != get_empty_sdata:
                    assert g(dim0=21) != h()
            else:
                assert g() != h()


if __name__ == "__main__":
    test_single_component_sdata()
