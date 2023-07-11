import pytest
from spatialdata.dataloader import build_graph
from spatialdata.datasets import blobs

sdata = blobs()


@pytest.mark.parametrize(
    "method, kwargs", [("knn", {"neighbors": 10}), ("distance", {"threshold": 0.5}), ("expansion", {"distance": 5.0})]
)
def test_build_graph(method, kwargs):
    gdf = sdata["blobs_circles"]
    build_graph(gdf=gdf, method=method, **kwargs)


@pytest.mark.parametrize("method", ["knn", "distance", "expansion", "invalid"])
@pytest.mark.parametrize("kwargs", [{"neighbors": 10, "threshold": 0.5}, {"neighbors": 0.5}, {"neighbors": 5.0}])
def test_build_graph_invalid_arguments(method, kwargs):
    gdf = sdata["blobs_circles"]
    if method == "invalid":
        with pytest.raises(AssertionError):
            build_graph(gdf=gdf, method=method, **kwargs)
