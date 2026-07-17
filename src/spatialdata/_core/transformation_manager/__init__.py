from __future__ import annotations

import warnings

import networkx as nx

from spatialdata._core.transformation_manager.exceptions import (
    CoordinateSystemNotFoundError,
    ElementNotFoundError,
    TransformationNotFoundError,
)
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata.transformations.transformations import BaseTransformation

TRANSFORM_KEY = "transformation"


class TransformationManager:
    def __init__(self) -> None:
        """Initialize a TransformationManager with empty graph and mappings."""
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # MultiDiGraph with NgffCoordinateSystem objects as nodes and transforms as edge attributes
        self._element_to_cs_mapping: dict[str, NgffCoordinateSystem] = {}
        # mapping element_name to the coordinate system to which the element belongs

    @property
    def graph(self) -> nx.MultiDiGraph:
        """
        Get the internal transformation graph.

        Returns
        -------
        The MultiDiGraph containing coordinate systems and transformations.
        """
        return self._graph

    @property
    def element_to_cs_mapping(self) -> dict[str, NgffCoordinateSystem]:
        """
        Get the element to coordinate system mapping.

        Returns
        -------
        A dictionary mapping element names to their coordinate systems.
        """
        return self._element_to_cs_mapping

    def check_if_element_exists(self, element_name: str) -> None:
        """
        Check if an element exists in the transformation manager.

        Parameters
        ----------
        element_name
            The name of the element to check.

        Raises
        ------
        ElementNotFoundError
            If the element does not exist.
        """
        if element_name not in self.element_to_cs_mapping:
            raise ElementNotFoundError(element_name)

    def check_if_coordinate_system_exists(self, cs: NgffCoordinateSystem) -> None:
        """
        Check if a coordinate system exists in the graph.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system does not exist.
        """
        if cs not in self.graph:
            raise CoordinateSystemNotFoundError(cs.name)

    def check_if_edge_exists(self, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem) -> None:
        """
        Check if an edge exists between coordinate systems.

        Parameters
        ----------
        input_cs
            The input coordinate system.
        output_cs
            The output coordinate system.

        Raises
        ------
        TransformationNotFoundError
            If the edge does not exist.
        """
        if not self.graph.has_edge(input_cs, output_cs):
            raise TransformationNotFoundError(input_cs.name, output_cs.name)

    def add_coordinate_system(self, cs: NgffCoordinateSystem) -> None:
        """
        Register a new coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to add.

        Raises
        ------
        ValueError
            If the coordinate system already exists.
        """
        if cs in self.graph:
            raise ValueError(f"Coordinate system '{cs.name}' already exists in the transformation manager")
        self.graph.add_node(cs)

    def list_coordinate_systems(self) -> list[NgffCoordinateSystem]:
        """
        List all registered coordinate systems.

        Returns
        -------
        A list of coordinate system objects.
        """
        return list(self.graph.nodes())

    def add_element(self, element_name: str, coordinate_system: NgffCoordinateSystem) -> None:
        """
        Register an element and associate it with a coordinate system.

        Parameters
        ----------
        element_name
            The name of the element.
        coordinate_system
            The coordinate system to which the element belongs.

        Warnings
        --------
        UserWarning
            If the coordinate system does not exist.
        """
        try:
            self.check_if_element_exists(element_name)
        except ElementNotFoundError as _enfe:
            warnings.warn(
                f"Cannot add element with name '{element_name}') as it already "
                f"exists in the transformation manager. Skipping",
                UserWarning,
                stacklevel=2,
            )

        try:
            self.check_if_coordinate_system_exists(coordinate_system)
        except CoordinateSystemNotFoundError as _csnfe:
            warnings.warn(
                f"Cannot set coordinate system ('{coordinate_system.name}') to element as the "
                f"coordinate system does not exist.",
                UserWarning,
                stacklevel=2,
            )
            return

        self.element_to_cs_mapping[element_name] = coordinate_system

    def get_element_coordinate_system(self, element_name: str) -> NgffCoordinateSystem:
        """
        Get the coordinate system to which an element belongs.

        Parameters
        ----------
        element_name
            The name of the element.

        Returns
        -------
        The coordinate system or None if not found

        Raises
        ------
        ElementNotFoundError
            If the element does not exist.
        """
        self.check_if_element_exists(element_name)
        return self.element_to_cs_mapping[element_name]

    def unset_element(self, element_name: str) -> None:
        """
        Unregister an element from the coordinate system to which it belongs.

        Parameters
        ----------
        element_name
            The name of the element.

        Raises
        ------
        ElementNotFoundError
            If the element has not been registered to any coordinate system.
        """
        self.check_if_element_exists(element_name)
        del self.element_to_cs_mapping[element_name]

    def add_transformation(
        self, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem, transformation: BaseTransformation
    ) -> None:
        """
        Add a transformation between coordinate systems.

        Parameters
        ----------
        input_cs
            The input coordinate system.
        output_cs
            The output coordinate system.
        transformation
            The transformation to add.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        self.graph.add_edge(input_cs, output_cs, **{TRANSFORM_KEY: transformation})

    def get_existing_transformation(
        self, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem
    ) -> BaseTransformation:
        """
        Retrieve a transformation defined between coordinate systems.

        Parameters
        ----------
        input_cs
            The input coordinate system.
        output_cs
            The output coordinate system.

        Returns
        -------
        The transformation or None if not found

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the transformation does not exist.
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        self.check_if_edge_exists(input_cs, output_cs)

        transform: BaseTransformation = self.graph[input_cs][output_cs][0][TRANSFORM_KEY]
        return transform

    def remove_transformation(self, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem) -> None:
        """
        Remove a transformation between coordinate systems.

        Parameters
        ----------
        input_cs
            The input coordinate system.
        output_cs
            The output coordinate system.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the transformation does not exist.
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        self.check_if_edge_exists(input_cs, output_cs)
        self.graph.remove_edge(input_cs, output_cs)

    def get_shortest_transformation_sequence(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem
    ) -> list[BaseTransformation]:
        """
        Get the shortest sequence of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.

        Returns
        -------
        list[BaseTransformation]
            The shortest sequence of transformations from source_cs to target_cs.

        Raises
        ------
        ValueError
            If no path exists between the source and target coordinate systems.
        """
        if self.graph.has_edge(source_cs, target_cs):
            return [self.graph[source_cs][target_cs][0][TRANSFORM_KEY]]

        try:
            path = nx.shortest_path(self.graph, source=source_cs, target=target_cs)

        except nx.NetworkXNoPath as nxe:
            raise ValueError(f"No path found from {source_cs.name} to {target_cs.name}") from nxe

        transformations = []
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i + 1]]
            transformations.append(edge_data[0][TRANSFORM_KEY])
        return transformations

    def get_all_transformation_sequences(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem
    ) -> list[list[BaseTransformation]]:
        """
        Get all existing sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.

        Returns
        -------
        list[list[BaseTransformation]]
            All existing sequences of transformations from source_cs to target_cs.
        """
        paths = list(nx.all_simple_paths(self.graph, source=source_cs, target=target_cs))

        all_sequences = []
        for path in paths:
            sequence = []
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                sequence.append(edge_data[0][TRANSFORM_KEY])
            all_sequences.append(sequence)
        return all_sequences

    def _get_transformations_associated_with_cs(
        self, cs: NgffCoordinateSystem
    ) -> list[tuple[NgffCoordinateSystem, NgffCoordinateSystem]]:
        """
        Get all transformations associated with a coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Returns
        -------
        List of tuples representing transformations (input_cs, output_cs).
        """
        self.check_if_coordinate_system_exists(cs)

        transformations = []
        # Check outgoing edges (cs -> other)
        for successor in self.graph.successors(cs):
            transformations.append((cs, successor))
        # Check incoming edges (other -> cs)
        for predecessor in self.graph.predecessors(cs):
            transformations.append((predecessor, cs))

        return transformations

    def _get_elements_belonging_to_cs(self, cs: NgffCoordinateSystem) -> list[str]:
        """
        Get all elements belonging to a coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Returns
        -------
        List of element names belonging to the coordinate system.
        """
        self.check_if_coordinate_system_exists(cs)

        elements = []
        for element_name, element_cs in self.element_to_cs_mapping.items():
            if element_cs == cs:
                elements.append(element_name)

        return elements

    def remove_coordinate_system(self, cs: NgffCoordinateSystem) -> None:
        """
        Remove a coordinate system from the transformation manager.

        Parameters
        ----------
        cs
            The coordinate system to remove.

        Raises
        ------
        ValueError
            If the coordinate system has associated transformations or elements.
        KeyError
            If the coordinate system does not exist.
        """
        self.check_if_coordinate_system_exists(cs)

        # Check if coordinate system has any transformations
        if len(list(self.graph.edges(cs))) > 0:
            raise ValueError(f"Cannot remove coordinate system '{cs.name}' as it has associated transformations")

        # Check if coordinate system has any associated elements
        associated_elements = self._get_elements_belonging_to_cs(cs)
        if associated_elements:
            raise ValueError(
                f"Cannot remove coordinate system '{cs.name}' as it has associated elements: {associated_elements}"
            )

        self.graph.remove_node(cs)

    def __repr__(self) -> str:
        return (
            f"TransformationManager("
            f"  coordinate_systems={list(self.graph.nodes())}, "
            f"  coordinate_transforms={list(self.graph.edges())}, "
            f"  elements={list(self.element_to_cs_mapping.keys())}"
            f")"
        )
