import itertools
from typing import Callable

import numpy as np
import pytest
from anndata import AnnData

from spatialdata import SpatialData
from spatialdata.utils import compare_sdata_on_disk


def get_empty_sdata():
    return SpatialData()


def get_table_sdata(dim0=20):
    return SpatialData(table=AnnData(np.ones((dim0, 10))))


def get_labels_sdata(dim0=20):
    return SpatialData(labels={"label": np.ones((dim0, 10))})


def get_images_sdata(dim0=20):
    return SpatialData(images={"image": np.ones((dim0, 10, 10))})


def get_points_sdata(dim0=20):
    return SpatialData(points={"points": AnnData(np.ones((dim0, 10)))})


@pytest.mark.parametrize(
    "getter1", [get_empty_sdata, get_table_sdata, get_images_sdata, get_labels_sdata, get_points_sdata]
)
@pytest.mark.parametrize(
    "getter2", [get_empty_sdata, get_table_sdata, get_images_sdata, get_labels_sdata, get_points_sdata]
)
def test_single_component_sdata(getter1: Callable, getter2: Callable):
    if getter1 == getter2:
        assert compare_sdata_on_disk(getter1(), getter2())


# pycharm debugging
if __name__ == "__main__":
    for getter1, getter2 in itertools.product(
        [get_empty_sdata, get_table_sdata, get_images_sdata, get_labels_sdata, get_points_sdata], repeat=2
    ):
        if getter1 == getter2:
            assert compare_sdata_on_disk(getter1(), getter2())
