from __future__ import annotations

from spatialdata.datasets import blobs
from spatialdata.transformations._graph import IdentityEdge, ProjectAxisEdge
from spatialdata.transformations._graph.utils import get_intrinsic_coordinate_system_name


def test_blobs_dataset_visualization_export():
    sdata = blobs()
    # has elements of all types mapping to "global" cs with Identity transformation
    graph = sdata.transformations._graph

    global_cs = graph.get_coordinate_system("global")
    assert global_cs.axes_names == ("c", "y", "x")

    for element_name, _element in (sdata.images | sdata.labels | sdata.shapes | sdata.points).items():
        element_cs = graph.get_element_coordinate_system(element_name)
        assert element_cs.name == get_intrinsic_coordinate_system_name(element_name)

        created_axes = set(global_cs.axes) - set(element_cs.axes)

        if created_axes:
            intermediate_cs = graph.get_coordinate_system(f"global_{element_name}")
            direct_edges_element_to_intermediate = graph.get_direct_transformations(element_cs, intermediate_cs)
            assert len(direct_edges_element_to_intermediate) == 1
            assert isinstance(direct_edges_element_to_intermediate[0], IdentityEdge)

            direct_edges_intermediate_to_global = graph.get_direct_transformations(intermediate_cs, global_cs)
            assert len(direct_edges_intermediate_to_global) == 1
            assert isinstance(direct_edges_intermediate_to_global[0], ProjectAxisEdge)
            assert direct_edges_intermediate_to_global[0].created_outputs == created_axes
            assert direct_edges_intermediate_to_global[0].dropped_inputs == set()

        else:
            direct_edges_element_to_global = graph.get_direct_transformations(element_cs, global_cs)
            assert len(direct_edges_element_to_global) == 1
            assert isinstance(direct_edges_element_to_global[0], IdentityEdge)
