import pytest
from spatialdata.dataloader import build_graph
from spatialdata.datasets import blobs

sdata = blobs()


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("knn", {"k": 2}),
        ("knn", {"k": 2, "percentile": 50}),
        ("knn", {"k": 2, "max_distance": 50}),
        ("radius", {"max_distance": 50}),
        ("expansion", {"max_distance": 50}),
    ],
)
@pytest.mark.parametrize("self_loops", [True, False])
def test_build_graph(method, self_loops, kwargs):
    gdf = sdata["blobs_circles"]
    build_graph(gdf=gdf, method=method, **kwargs)


# invalid choices
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("knn", {"k": None}),
        ("knn", {"k": 2, "percentile": 50, "max_distance": 50}),
        ("radius", {"k": 50}),
        ("radius", {"max_distance": 50, "percentile": 50}),
        ("expansion", {"k": 50}),
        ("expansion", {"percentile": 50}),
        ("invalid", {}),
    ],
)
def test_build_graph_invalid_arguments(method, kwargs):
    gdf = sdata["blobs_circles"]
    with pytest.raises(AssertionError):
        build_graph(gdf=gdf, method=method, **kwargs)
