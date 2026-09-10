from __future__ import annotations

from collections.abc import Sequence
from typing import override

import networkx as nx

from spatialdata._core.transformation_manager.exceptions import (
    CannotRemoveCoordinateSystemError,
    CoordinateSystemAlreadyExistsError,
    CoordinateSystemHasElementsError,
    CoordinateSystemHasTransformationsError,
    CoordinateSystemNotFoundError,
    ElementAlreadyExistsError,
    ElementNotRegisteredToAnyCoordinateSystemError,
    InvalidPathError,
    TransformationNotFoundError,
    TransformationPathAmbiguousError,
    TransformationPathAmbiguousMultipleEdgeExpectedError,
    TransformationPathAmbiguousNoEdgeExpectedError,
    TransformationPathNotFoundError,
    TransformationPathNotSimple,
)
from spatialdata._core.transformation_manager.types import TRANSFORM_KEY
from spatialdata.transformations.graph.edge import BaseTransfEdge, SequenceEdge
from spatialdata.transformations.graph.vert import CoordSystem


class TransformationGraph:
    def __init__(self) -> None:
        """Initialize a TransformationManager with empty graph and mappings."""
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # MultiDiGraph with CoordSystem objects as nodes and BaseTransfEdge as edge attributes
        self._element_to_cs_mapping: dict[str, CoordSystem] = {}
        # mapping element_name to the coordinate system to which the element belongs

    def assert_element_exists(self, element_name: str) -> None:
        """
        Assert that an element exists in the transformation manager.

        Parameters
        ----------
        element_name
            The name of the element to check.

        Raises
        ------
        ElementNotFoundError
            If the element does not exist.
        """
        if element_name not in self._element_to_cs_mapping:
            raise ElementNotRegisteredToAnyCoordinateSystemError(element_name)

    def assert_element_does_not_exist(self, element_name: str) -> None:
        """
        Assert than an element doesn't exist in the transformation manager.

        Parameters
        ----------
        element_name
            The name of the element to check.

        Raises
        ------
        ElementAlreadyExistsError
            If the element already exists.
        """
        if element_name in self._element_to_cs_mapping:
            raise ElementAlreadyExistsError(element_name)

    def assert_coordinate_system_exists(self, cs: CoordSystem) -> None:
        """
        Assert that coordinate system exists in the graph.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system does not exist.
        """
        if cs not in self._graph:
            raise CoordinateSystemNotFoundError(cs.name)

    def assert_coordinate_system_does_not_exist(self, cs: CoordSystem) -> None:
        """
        Assert that coordinate system doesn't exist in the graph.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Raises
        ------
        CoordinateSystemAlreadyExistsError
            If the coordinate system already exists.
        """
        if cs in self._graph:
            raise CoordinateSystemAlreadyExistsError(cs.name)

    def assert_an_edge_exists_between_coordinate_systems(
        self, source_cs: CoordSystem, target_cs: CoordSystem, edge_key: str | None = None
    ) -> None:
        """
        Assert that an edge exists between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
            The output coordinate system.
        edge_key
            If specified, asserts that the specific edge exists between coordinate systems


        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the edge does not exist.
        """
        self.assert_coordinate_system_exists(source_cs)
        self.assert_coordinate_system_exists(target_cs)
        if not self._graph.has_edge(source_cs, target_cs):
            raise TransformationNotFoundError(source_cs.name, target_cs.name, edge_key=edge_key)

    def assert_coordinate_system_has_no_transformations(self, cs: CoordSystem) -> None:
        """
        Assert that coordinate system has no associated transformations.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        CoordinateSystemHasTransformationsError
            If the coordinate system has associated transformations.

        """
        self.assert_coordinate_system_exists(cs)
        has_successors = next(self._graph.successors(cs), None) is not None
        has_predecessors = next(self._graph.predecessors(cs), None) is not None

        if has_successors or has_predecessors:
            raise CoordinateSystemHasTransformationsError(cs.name)

    def assert_coordinate_system_has_no_elements(self, cs: CoordSystem) -> None:
        """
        Assert that a coordinate system doesn't have elements belonging to it.

        Parameters
        ----------
        cs
            The coordinate system to check

        Raises
        ------
        CoordinateSystemHasElementsError
            if the coordinate system has elements belonging to it.
        """
        elements = self._get_elements_belonging_to_cs(cs)
        # also checks if cs exists
        if elements:
            raise CoordinateSystemHasElementsError(cs.name, elements)

    def add_coordinate_system(self, cs: CoordSystem) -> None:
        """
        Add a coordinate system to the transformation manager.

        Parameters
        ----------
        cs
            The coordinate system to add.

        Raises
        ------
        CoordinateSystemAlreadyExistsError
            If the coordinate system already exists.
        """
        self.assert_coordinate_system_does_not_exist(cs)
        self._graph.add_node(cs)

    def remove_coordinate_system(self, cs: CoordSystem) -> None:
        """
        Remove a coordinate system from the transformation manager.

        Parameters
        ----------
        cs
            The coordinate system to remove.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system is not found
        CannotRemoveCoordinateSystemError
            If the coordinate system cannot be removed.
            Caused by
                CoordinateSystemHasTransformationsError or
                CoordinateSystemHasElementsError when the system has dependencies.
        """
        try:
            self.assert_coordinate_system_has_no_transformations(cs)
            # also asserts that cs exists
            self.assert_coordinate_system_has_no_elements(cs)
        except (CoordinateSystemHasTransformationsError, CoordinateSystemHasElementsError) as err:
            raise CannotRemoveCoordinateSystemError(cs.name) from err

        self._graph.remove_node(cs)

    def list_coordinate_systems(self) -> list[CoordSystem]:
        """
        List all registered coordinate systems.

        Returns
        -------
        A list of coordinate system objects.
        """
        return list(self._graph.nodes())

    def add_element(self, element_name: str, coordinate_system: CoordSystem) -> None:
        """
        Register an element and associate it with a coordinate system.

        Parameters
        ----------
        element_name
            The name of the element.
        coordinate_system
            The coordinate system to which the element belongs.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system is not found.
        """
        self.assert_element_does_not_exist(element_name)
        self.assert_coordinate_system_exists(coordinate_system)
        self._element_to_cs_mapping[element_name] = coordinate_system

    def get_element_coordinate_system(self, element_name: str) -> CoordSystem:
        """
        Get the coordinate system to which an element belongs.

        Parameters
        ----------
        element_name
            The name of the element.

        Returns
        -------
        The coordinate system

        Raises
        ------
        ElementNotFoundError
            If the element does not exist.
        """
        self.assert_element_exists(element_name)
        return self._element_to_cs_mapping[element_name]

    def get_outgoing_edges(
        self, element_name: str, target_coordinate_system_names: Sequence[str] | None
    ) -> list[BaseTransfEdge]:

        element_coordinate_system = self.get_element_coordinate_system(element_name)
        outgoing_edges: list[BaseTransfEdge] = []

        successors_to_consider = self._graph.successors(element_coordinate_system)
        if target_coordinate_system_names is not None:
            successors_to_consider = [x for x in successors_to_consider if x.name in target_coordinate_system_names]

        for successor_coordinate_system in successors_to_consider:
            edge_data = self._graph[element_coordinate_system][successor_coordinate_system]
            outgoing_edge_this_successor = [x[TRANSFORM_KEY] for x in edge_data.values()]
            outgoing_edges += outgoing_edge_this_successor

        return outgoing_edges

    def _unset_element(self, element_name: str) -> None:
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
        self.assert_element_exists(element_name)
        del self._element_to_cs_mapping[element_name]

    @staticmethod
    def _get_edge_key(edge: BaseTransfEdge) -> str:
        """
        Encode the definition of an edge into a str to be used as graph-wide edge identifier.

        Parameters
        ----------
        edge

        Returns
        -------
        edge_key
            str, the encoded edge identifier
        """
        return str(id(edge))

    def add_transformation_edge(
        self,
        transformation_edge: BaseTransfEdge,
    ) -> None:
        """
        Add an edge representing a transformation between coordinate systems.

        Parameters
        ----------
        transformation_edge
            The transformation edge to add.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        """
        source_cs, target_cs = transformation_edge.input, transformation_edge.output
        self.assert_coordinate_system_exists(source_cs)
        self.assert_coordinate_system_exists(target_cs)

        edge_key = self._get_edge_key(transformation_edge)
        edge_attributes = {TRANSFORM_KEY: transformation_edge}
        self._graph.add_edge(source_cs, target_cs, key=edge_key, **edge_attributes)

    def get_direct_transformations(self, source_cs: CoordSystem, target_cs: CoordSystem) -> list[BaseTransfEdge]:
        """
        Retrieve transformations directly defined between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
            The output coordinate system.

        Returns
        -------
        List of transformations (empty if there are no direct transformations).

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        """
        self.assert_coordinate_system_exists(source_cs)
        self.assert_coordinate_system_exists(target_cs)

        transforms = []
        if self._graph.has_edge(source_cs, target_cs):
            for edge_data in self._graph[source_cs][target_cs].values():
                transform: BaseTransfEdge = edge_data[TRANSFORM_KEY]
                transforms.append(transform)
        return transforms

    def remove_specific_transformation(
        self,
        transformation_edge: BaseTransfEdge,
    ) -> None:
        """
        Remove a specific transformation between coordinate systems.

        Parameters
        ----------
        transformation_edge
            The transformation edge to remove.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the transformation does not exist.
        """
        source_cs, target_cs = transformation_edge.input, transformation_edge.output
        expected_edge_key = self._get_edge_key(transformation_edge)
        self.assert_an_edge_exists_between_coordinate_systems(source_cs, target_cs, edge_key=expected_edge_key)
        # also checks if source_cs and target_cs exist
        self._graph.remove_edge(source_cs, target_cs, key=expected_edge_key)

    def remove_all_transformations_between_coordinate_systems(
        self,
        source_cs: CoordSystem,
        target_cs: CoordSystem,
    ) -> None:
        """
        Remove all transformation between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
            The output coordinate system.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If no transformation exists between the coordiante systems
        """
        self.assert_coordinate_system_exists(source_cs)
        self.assert_coordinate_system_exists(target_cs)
        if self._graph.has_edge(source_cs, target_cs):
            transform_edges_to_remove = {}
            for edge_key in self._graph[source_cs][target_cs]:
                transform_edges_to_remove[edge_key] = self._graph[source_cs][target_cs][edge_key][TRANSFORM_KEY]

            for edge_key, transform_edge in transform_edges_to_remove.items():
                self._graph.remove_edge(transform_edge.input, transform_edge.output, key=edge_key)

    def _get_transformation_sequences_from_simple_paths_after_disambiguation(
        self,
        paths: list[list[CoordSystem]],
        expected_intermediate_edges: Sequence[BaseTransfEdge] | None,
    ) -> list[SequenceEdge]:
        """
        Traverses paths to form sequence of Transformations.

        In case of ambiguity looks into `expected_intermediate_transformations` to disambiguate.

        Parameters
        ----------
        paths:
            sequence of list of nodes
        expected_intermediate_edges:
            list of edges, for use when multiple edges are found between two coordinate systems in a path

        Returns
        -------
            list of transformation edges, each an instance of the transformation class Sequence

        Raises
        ------
            TransformationPathNotSimple
                if any path in `paths` is not simple, i.e., has recurring coordinate systems
        """
        for path in paths:
            if len(set(path)) != len(path):
                raise TransformationPathNotSimple(path)

        expected_intermediate_transformation_edge_keys = set[str]()
        if expected_intermediate_edges is not None:
            for edge_def in expected_intermediate_edges:
                edge_key = self._get_edge_key(edge_def)
                expected_intermediate_transformation_edge_keys |= {edge_key}

        all_sequences = []
        deduplicated_paths = list({repr(x): x for x in paths}.values())
        for path in deduplicated_paths:
            if len(path) <= 1:
                raise InvalidPathError(path)

            transformation_list: list[BaseTransfEdge] = []
            for i in range(len(path) - 1):
                source_cs_here, target_cs_here = path[i], path[i + 1]
                edge_data = self._graph[source_cs_here][target_cs_here]
                if len(edge_data) > 1:
                    # when there are multiple edges between a pair of coordinate systems in the path
                    # find the expected edge based on key
                    expected_intermediate_transformation_key_here = (
                        expected_intermediate_transformation_edge_keys & set(edge_data.keys())
                    )
                    if len(expected_intermediate_transformation_key_here) == 0:
                        # transformation was not specified in `expected_intermediate_transformations` for disambiguation
                        raise TransformationPathAmbiguousNoEdgeExpectedError(source_cs_here.name, target_cs_here.name)
                    if len(expected_intermediate_transformation_key_here) > 1:
                        # multiple transformations were specified in `expected_intermediate_transformations`
                        # for disambiguation
                        raise TransformationPathAmbiguousMultipleEdgeExpectedError(
                            source_cs_here.name, target_cs_here.name, len(expected_intermediate_transformation_key_here)
                        )
                    edge_key_to_use = list(expected_intermediate_transformation_key_here)[0]
                    transformation_list.append(edge_data[edge_key_to_use][TRANSFORM_KEY])
                else:
                    # Only one edge, no ambiguity
                    edge_key = next(iter(edge_data.keys()))
                    transformation_list.append(edge_data[edge_key][TRANSFORM_KEY])
            all_sequences.append(SequenceEdge(transformations=transformation_list))
        return all_sequences

    def get_all_shortest_transformation_sequences(
        self,
        source_cs: CoordSystem,
        target_cs: CoordSystem,
        expected_intermediate_edges: Sequence[BaseTransfEdge] = (),
    ) -> list[SequenceEdge]:
        """
        Get all shortest sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.
        expected_intermediate_edges
            list of intermediate edges expected.
            Used to choose an edge when multiple edges are found between the same coordinate systems.

        Returns
        -------
        list[Sequence]
            All shortest sequences of transformations from source_cs to target_cs.
            When multiple transformations are defined between the same coordinate systems, only those containing
            a transformation among intermediate transformations are included.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationPathNotFoundError
            If no path exists between the source and target coordinate systems.
        TransformationPathAmbiguousError
            When multiple transformations are defined between the same coordinate systems and transformations
            are not specified in `expeceted_intermediate_transformations` for disambiguation.
        """
        try:
            paths = list(nx.all_shortest_paths(self._graph, source=source_cs, target=target_cs))
        except nx.NetworkXNoPath as nxe:
            raise TransformationPathNotFoundError(source_cs.name, target_cs.name) from nxe

        try:
            return self._get_transformation_sequences_from_simple_paths_after_disambiguation(
                paths, expected_intermediate_edges
            )
        except TransformationPathAmbiguousError as tpae:
            raise TransformationPathAmbiguousError(source_cs.name, target_cs.name) from tpae

    def get_all_transformation_sequences(
        self,
        source_cs: CoordSystem,
        target_cs: CoordSystem,
        expected_intermediate_edges: Sequence[BaseTransfEdge] = (),
    ) -> list[SequenceEdge]:
        """
        Get all existing sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.
        expected_intermediate_edges
            list of intermediate edges expected.
            Used to choose an edge when multiple edges are found between the same coordinate systems.

        Returns
        -------
        list[Sequence]
            All existing sequences of transformations from source_cs to target_cs.
            When multiple transformations are defined between the same coordinate systems, only those containing
            a transformation among intermediate transformations are included.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationPathNotFoundError
            If no path exists between the source and target coordinate systems.
        TransformationPathAmbiguousError
            When multiple transformations are defined between the same coordinate systems and transformations
            are not specified in `expected_intermediate_transformations` for disambiguation.
        """
        try:
            paths = list(nx.all_simple_paths(self._graph, source=source_cs, target=target_cs))
        except nx.NetworkXNoPath as nxe:
            raise TransformationPathNotFoundError(source_cs.name, target_cs.name) from nxe

        try:
            return self._get_transformation_sequences_from_simple_paths_after_disambiguation(
                paths, expected_intermediate_edges
            )
        except TransformationPathAmbiguousError as tpae:
            raise tpae from TransformationPathAmbiguousError(source_cs.name, target_cs.name)

    def _get_elements_belonging_to_cs(self, cs: CoordSystem) -> list[str]:
        """
        Get all elements belonging to a coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Returns
        -------
        List of element names belonging to the coordinate system.

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system does not exist.

        """
        self.assert_coordinate_system_exists(cs)

        elements = []
        for element_name, element_cs in self._element_to_cs_mapping.items():
            if element_cs == cs:
                elements.append(element_name)

        return elements

    @override
    def __repr__(self) -> str:
        """Return a string representation of the TransformationManager."""
        return (
            f"TransformationManager("
            f"  coordinate_systems={list(self._graph.nodes())}, "
            f"  coordinate_transforms={[x[TRANSFORM_KEY] for *_, x in self._graph.edges(data=True)]}, "
            f"  elements={list(self._element_to_cs_mapping.keys())}"
            f")"
        )
