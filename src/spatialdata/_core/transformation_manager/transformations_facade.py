from __future__ import annotations

from collections.abc import Sequence

from spatialdata.models import SpatialElement
from spatialdata.transformations import BaseTransformation
from spatialdata.transformations._graph import (
    BaseTransformationEdge,
    CoordinateSystemNotFoundError,
    TransformationGraph,
    get_default_coordinate_system,
)


class TransformationsFacade:
    def __init__(self) -> None:
        """Initialize a TransformationsFacade with an empty transformation graph."""
        self._graph = TransformationGraph()

    def _get_element_transformation_edges(
        self, element_name: str, target_coordinate_system_names: Sequence[str] | None = None
    ) -> list[BaseTransformationEdge]:
        """
        Get the NGFF transformation edges associated with an element.

        Parameters
        ----------
        element_name
            The name of the element.
        target_coordinate_system_names
            Optional sequence of target coordinate system names to filter by.

        Returns
        -------
        List of transformation edges for the element.
        """
        return self._graph.get_outgoing_edges(element_name, target_coordinate_system_names)

    def _add_new_element_to_graph_creating_default_cs(self, new_element: SpatialElement, new_element_key: str) -> None:
        """
        Add a new element to the transformation graph with a default coordinate system.

        Parameters
        ----------
        new_element
            The spatial element to add to the graph.
        new_element_key
            The key/name to use for the new element in the graph.
        """
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
        """
        Set transformations for an element to specified target coordinate systems

        This is done by adding corresponding coordinate systems and transformation edges to the graph

        Parameters
        ----------
        element_name
            The name of the element.
        transformations_config
            Dictionary mapping target coordinate system names to (spatialdata) transformations.

        Returns
        -------
        List of NGFF transformation edges that were created and added to the graph.
        """
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
        """
        Remove specific transformation edges from the graph.

        Parameters
        ----------
        transformation_edges
            List of transformation edges to remove.
        """
        for transformation_edge in transformation_edges:
            self._graph.remove_specific_transformation(transformation_edge)

    def remove_all_transformations_of_element(self, element_name: str) -> None:
        """
        Remove all transformation edges associated with an element.

        Parameters
        ----------
        element_name
            The name of the element whose transformations should be removed.
        """
        all_outgoing_edges_from_element = self._get_element_transformation_edges(element_name)
        self.remove_transformations(all_outgoing_edges_from_element)

    def remove_all_transformations_to_coordinate_system(self, coordinate_system_name: str) -> None:
        """
        Remove all transformation edges that target a specific coordinate system.

        Parameters
        ----------
        coordinate_system_name
            The name of the target coordinate system.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system is not found.
        AssertionError
            If multiple coordinate systems with the same name exist (should never happen).
        """
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
