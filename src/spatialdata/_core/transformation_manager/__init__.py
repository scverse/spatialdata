from __future__ import annotations

import warnings

import networkx as nx

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

    def check_if_coordinate_system_exists(self, cs: NgffCoordinateSystem) -> None:
        """
        Check if a coordinate system exists in the graph.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Raises
        ------
        ValueError
            If the coordinate system does not exist.
        """
        if cs not in self._graph:
            raise ValueError(f"Coordinate system '{cs.name}' does not exist in the transformation manager.")

    def list_coordinate_systems(self) -> list[NgffCoordinateSystem]:
        """
        List all registered coordinate systems.

        Returns
        -------
        A list of coordinate system objects.
        """
        return list(self._graph.nodes())

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
        if coordinate_system not in self._graph:
            warnings.warn(
                f"Cannot set coordinate system ('{coordinate_system.name}') to element as the "
                f"coordinate system does not exist.",
                UserWarning,
                stacklevel=2,
            )
            return

        self._element_to_cs_mapping[element_name] = coordinate_system

    def get_element_coordinate_system(self, element_name: str) -> NgffCoordinateSystem | None:
        """
        Get the coordinate system to which an element belongs.

        Parameters
        ----------
        element_name
            The name of the element.

        Returns
        -------
        The coordinate system or None if not found
        """
        return self._element_to_cs_mapping.get(element_name)

    def unset_element(self, element_name: str) -> None:
        """
        Unregister an element from the coordinate system to which it belongs.

        Parameters
        ----------
        element_name
            The name of the element.

        Warnings
        --------
        UserWarning
            If the element has not been registered to any coordinate system.
        """
        if element_name not in self._element_to_cs_mapping:
            warnings.warn(f"Element '{element_name}' not found in any coordinate system.", UserWarning, stacklevel=2)
            return
        del self._element_to_cs_mapping[element_name]

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
        ValueError
            If either coordinate system does not exist.
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        self._graph.add_edge(input_cs, output_cs, **{TRANSFORM_KEY: transformation})

    def get_transformation(
        self, input_cs: NgffCoordinateSystem, output_cs: NgffCoordinateSystem
    ) -> BaseTransformation | None:
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
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        if self._graph.has_edge(input_cs, output_cs):
            transform: BaseTransformation = self._graph[input_cs][output_cs][0][TRANSFORM_KEY]
            return transform
        return None

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
        ValueError
            If either coordinate system does not exist.
        KeyError
            If the transformation does not exist.
        """
        self.check_if_coordinate_system_exists(input_cs)
        self.check_if_coordinate_system_exists(output_cs)

        if not self._graph.has_edge(input_cs, output_cs):
            raise KeyError(f"Transformation from '{input_cs.name}' to '{output_cs.name}' not found.")
        self._graph.remove_edge(input_cs, output_cs)

    def get_element_transformation(
        self, element_name: str, target_cs: NgffCoordinateSystem
    ) -> BaseTransformation | None:
        """
        Get the transformation from the coordinate system to which the element belongs to a target coordinate system.

        Parameters
        ----------
        element_name
            The name of the element.
        target_cs
            The target coordinate system.

        Returns
        -------
        The transformation or None if not found

        Raises
        ------
        KeyError
            If target_cs has not been added or if element_name does not belong to a coordinate system.
        """
        if target_cs not in self._graph:
            raise KeyError(f"Target coordinate system '{target_cs.name}' not found.")

        element_cs = self.get_element_coordinate_system(element_name)
        if element_cs is None:
            raise KeyError(f"Element '{element_name}' does not belong to any coordinate system.")

        return self.get_transformation(element_cs, target_cs)

    def build_nx_graph(self) -> nx.MultiDiGraph:
        """
        Build a directed graph where nodes are coordinate systems and edges are transformations.

        Returns
        -------
        nx.MultiDiGraph
            A directed graph representing the coordinate systems and transformations.
        """
        return self._graph.copy()

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
        if self._graph.has_edge(source_cs, target_cs):
            return [self._graph[source_cs][target_cs][0][TRANSFORM_KEY]]

        try:
            path = nx.shortest_path(self._graph, source=source_cs, target=target_cs)

        except nx.NetworkXNoPath as nxe:
            raise ValueError(f"No path found from {source_cs.name} to {target_cs.name}") from nxe

        transformations = []
        for i in range(len(path) - 1):
            edge_data = self._graph[path[i]][path[i + 1]]
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
        paths = list(nx.all_simple_paths(self._graph, source=source_cs, target=target_cs))

        all_sequences = []
        for path in paths:
            sequence = []
            for i in range(len(path) - 1):
                edge_data = self._graph[path[i]][path[i + 1]]
                sequence.append(edge_data[0][TRANSFORM_KEY])
            all_sequences.append(sequence)
        return all_sequences

    def __repr__(self) -> str:
        return (
            f"TransformationManager("
            f"  coordinate_systems={list(self._graph.nodes())}, "
            f"  coordinate_transforms={list(self._graph.edges())}, "
            f"  elements={list(self._element_to_cs_mapping.keys())}"
            f")"
        )
