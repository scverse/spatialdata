from __future__ import annotations

from collections.abc import Sequence

from spatialdata._core.transformation_manager._transformation_graph import TransformationGraph
from spatialdata._core.transformation_manager.exceptions import CoordinateSystemNotFoundError
from spatialdata.models import SpatialElement
from spatialdata.transformations import BaseTransformation
from spatialdata.transformations.graph.edge import BaseTransformationEdge
from spatialdata.transformations.graph.utils import get_default_coordinate_system


class TransformationsFacade:
    def __init__(self) -> None:
        self._graph = TransformationGraph()

    def _get_element_transformation_edges(
        self, element_name: str, target_coordinate_system_names: Sequence[str] | None = None
    ) -> list[BaseTransformationEdge]:

        return self._graph.get_outgoing_edges(element_name, target_coordinate_system_names)

    def _add_new_element_to_graph_creating_default_cs(self, new_element: SpatialElement, new_element_key: str) -> None:

        element_cs = get_default_coordinate_system(
            element_to_be_added_to_graph=new_element, coordinate_system_name=new_element_key
        )

        self._graph.add_coordinate_system(element_cs)
        self._graph.add_element(element_name=new_element_key, coordinate_system=element_cs)

    def set_element_transformations(
        self,
        element_name: str,
        transformations_config: dict[str, BaseTransformation],
    ) -> list[BaseTransformationEdge]:

        element_cs = self._graph.get_element_coordinate_system(element_name)

        transformation_edges = []
        for output_coordinate_system_name, t in transformations_config.items():
            ngff_transformation_edge = t._to_ngff_transformation_edge(
                input_coordinate_system=element_cs, output_coordinate_system_name=output_coordinate_system_name
            )
            self._graph.add_transformation_edge(ngff_transformation_edge)
            transformation_edges.append(ngff_transformation_edge)

        return transformation_edges

    def remove_transformations(self, transformation_edges: list[BaseTransformationEdge]) -> None:

        for transformation_edge in transformation_edges:
            self._graph.remove_specific_transformation(transformation_edge)

    def remove_all_transformations_of_element(self, element_name: str) -> None:

        all_outgoing_edges_from_element = self._get_element_transformation_edges(element_name)
        self.remove_transformations(all_outgoing_edges_from_element)

    def remove_all_transformations_to_coordinate_system(self, coordinate_system_name: str) -> None:

        cs_probables = [x for x in self._graph.list_coordinate_systems() if x.name == coordinate_system_name]
        if len(cs_probables) == 1:
            for predecessor in self._graph._graph.predecessors(cs_probables[0]):
                self._graph.remove_all_transformations_between_coordinate_systems(predecessor, cs_probables[0])
        elif len(cs_probables) == 0:
            CoordinateSystemNotFoundError(name=coordinate_system_name)
        else:
            raise AssertionError(
                "This case should never happen, please raise an issue at https://github.com/scverse/spatialdata/issues"
            )
