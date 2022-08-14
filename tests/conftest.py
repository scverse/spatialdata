from typing import Mapping, Sequence

import pytest
from anndata import AnnData
from numpy.random import default_rng

from spatialdata import SpatialData
from spatialdata._types import NDArray

RNG = default_rng()


@pytest.fixture()
def images() -> SpatialData:
    return SpatialData(images=_get_images(3))


@pytest.fixture()
def labels() -> SpatialData:
    return SpatialData(labels=_get_labels(3))


@pytest.fixture()
def points() -> SpatialData:
    return SpatialData(points=_get_points(3))


@pytest.fixture()
def tables() -> SpatialData:
    return SpatialData(tables=_get_tables(3))


@pytest.fixture()
def spatialdata() -> SpatialData:
    return SpatialData(images=_get_images(1), labels=_get_labels(1), points=_get_points(2), tables=_get_tables(3))


def _get_images(n: int) -> Mapping[str, Sequence[NDArray]]:
    return {f"image_{i}": RNG.normal(size=(3, 100, 100)) for i in range(n)}


def _get_labels(n: int) -> Mapping[str, Sequence[NDArray]]:
    return {f"labels_{i}": RNG.integers(0, size=(100, 100)) for i in range(n)}


def _get_points(n: int) -> Mapping[str, Sequence[NDArray]]:
    return {f"points_{i}": AnnData(RNG.integers(0, 100, size=(100, 2))) for i in range(n)}


def _get_tables(n: int) -> Mapping[str, Sequence[NDArray]]:
    return {f"tables_{i}": AnnData(RNG.normal(size=(100, 10))) for i in range(n)}
