from __future__ import annotations

import warnings

import networkx as nx

from spatialdata._core.transformation_manager.exceptions import (
    CannotRemoveCoordinateSystemError,
    CoordinateSystemAlreadyExistsError,
    CoordinateSystemHasElementsError,
    CoordinateSystemHasTransformationsError,
    CoordinateSystemNotFoundError,
    ElementAlreadyExistsError,
    ElementNotFoundError,
    InternalAttributeAccessWarning,
    TransformationNotFoundError,
    TransformationPathNotFoundError,
    suppress_direct_internal_attribute_access_warning,
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

        Note
        ----
        Direct manipulation of the graph is discouraged. Please use the
        TransformationManager methods whenever possible for better maintainability
        and to ensure proper validation and error handling.
        """
        warnings.warn(
            "Direct access to the internal graph is discouraged. "
            "Please use TransformationManager methods whenever possible for better "
            "maintainability and to ensure proper validation and error handling.",
            InternalAttributeAccessWarning,
            stacklevel=2,
        )

        return self._graph

    @property
    def element_to_cs_mapping(self) -> dict[str, NgffCoordinateSystem]:
        """
        Get the element to coordinate system mapping.

        Returns
        -------
        A dictionary mapping element names to their coordinate systems.

        Note
        ----
        Direct manipulation of internal element_to_cs_mapping is discouraged. Please use the
        TransformationManager methods whenever possible for better maintainability
        and to ensure proper validation and error handling.
        """
        warnings.warn(
            "Direct access to the internal element_to_cs_mapping is discouraged. "
            "Please use TransformationManager methods whenever possible for better "
            "maintainability and to ensure proper validation and error handling",
            InternalAttributeAccessWarning,
            stacklevel=2,
        )
        return self._element_to_cs_mapping

    def check_if_element_exists_else_raise_error(self, element_name: str) -> None:
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
        with suppress_direct_internal_attribute_access_warning():
            if element_name not in self.element_to_cs_mapping:
                raise ElementNotFoundError(element_name)

    def check_if_coordinate_system_exists_else_raise_error(self, cs: NgffCoordinateSystem) -> None:
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
        with suppress_direct_internal_attribute_access_warning():
            if cs not in self.graph:
                raise CoordinateSystemNotFoundError(cs.name)

    def check_if_edge_exists_else_raise_error(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem
    ) -> None:
        """
        Check if an edge exists between coordinate systems.

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
            If the edge does not exist.
        """
        self.check_if_coordinate_system_exists_else_raise_error(source_cs)
        self.check_if_coordinate_system_exists_else_raise_error(target_cs)
        with suppress_direct_internal_attribute_access_warning():
            if not self.graph.has_edge(source_cs, target_cs):
                raise TransformationNotFoundError(source_cs.name, target_cs.name)

    def check_if_coordinate_system_has_no_transformations_else_raise_error(self, cs: NgffCoordinateSystem) -> None:
        """
        Check if a coordinate system has associated transformations.

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
        transformations = self._get_transformations_associated_with_cs(cs)
        # also checks if cs exists
        if transformations:
            raise CoordinateSystemHasTransformationsError(cs.name)

    def check_if_coordinate_system_has_no_elements_else_raise_error(self, cs: NgffCoordinateSystem) -> None:
        """
        Check if a coordinate system has elements that belong to it.

        Parameters
        ----------
        cs
            The coordinate system to check

        Raises
        ------
        Co

        """
        elements = self._get_elements_belonging_to_cs(cs)
        # also checks if cs exists
        if elements:
            raise CoordinateSystemHasElementsError(cs.name, elements)

    def add_coordinate_system(self, cs: NgffCoordinateSystem) -> None:
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
        try:
            self.check_if_coordinate_system_exists_else_raise_error(cs)
        except CoordinateSystemNotFoundError:
            with suppress_direct_internal_attribute_access_warning():
                self.graph.add_node(cs)
                return

        raise CoordinateSystemAlreadyExistsError(cs.name)

    def remove_coordinate_system(self, cs: NgffCoordinateSystem) -> None:
        """
        Remove a coordinate system from the transformation manager.

        Parameters
        ----------
        cs
            The coordinate system to remove.

        Raises
        ------
        CoordinateSystemHasTransformationsError
            If the coordinate system has associated transformations.
        CoordinateSystemHasElementsError
            If the coordinate system has associated elements.
        CoordinateSystemNotFoundError
            If the coordinate system is not found
        """
        try:
            self.check_if_coordinate_system_has_no_transformations_else_raise_error(cs)
            # also checks if cs exists
            self.check_if_coordinate_system_has_no_elements_else_raise_error(cs)
        except (CoordinateSystemHasTransformationsError, CoordinateSystemHasElementsError) as err:
            raise CannotRemoveCoordinateSystemError(cs.name) from err

        with suppress_direct_internal_attribute_access_warning():
            self.graph.remove_node(cs)

    def list_coordinate_systems(self) -> list[NgffCoordinateSystem]:
        """
        List all registered coordinate systems.

        Returns
        -------
        A list of coordinate system objects.
        """
        with suppress_direct_internal_attribute_access_warning():
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

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system is not found.
        """
        try:
            self.check_if_element_exists_else_raise_error(element_name)
        except ElementNotFoundError:
            self.check_if_coordinate_system_exists_else_raise_error(coordinate_system)
            with suppress_direct_internal_attribute_access_warning():
                self.element_to_cs_mapping[element_name] = coordinate_system
                return

        raise ElementAlreadyExistsError(element_name)

    def get_element_coordinate_system(self, element_name: str) -> NgffCoordinateSystem:
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
        self.check_if_element_exists_else_raise_error(element_name)
        with suppress_direct_internal_attribute_access_warning():
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
        self.check_if_element_exists_else_raise_error(element_name)
        with suppress_direct_internal_attribute_access_warning():
            del self.element_to_cs_mapping[element_name]

    def add_transformation(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem, transformation: BaseTransformation
    ) -> None:
        """
        Add a transformation between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
            The output coordinate system.
        transformation
            The transformation to add.

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        """
        self.check_if_coordinate_system_exists_else_raise_error(source_cs)
        self.check_if_coordinate_system_exists_else_raise_error(target_cs)

        with suppress_direct_internal_attribute_access_warning():
            self.graph.add_edge(source_cs, target_cs, **{TRANSFORM_KEY: transformation})

    def get_existing_transformation(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem
    ) -> BaseTransformation:
        """
        Retrieve a transformation defined between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
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
        self.check_if_edge_exists_else_raise_error(source_cs, target_cs)
        # also checks if source_cs and target_cs exist
        with suppress_direct_internal_attribute_access_warning():
            transform: BaseTransformation = self.graph[source_cs][target_cs][0][TRANSFORM_KEY]
            return transform

    def remove_transformation(self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem) -> None:
        """
        Remove a transformation between coordinate systems.

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
            If the transformation does not exist.
        """
        self.check_if_edge_exists_else_raise_error(source_cs, target_cs)
        # also checks if source_cs and target_cs exist
        with suppress_direct_internal_attribute_access_warning():
            self.graph.remove_edge(source_cs, target_cs)

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
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationPathNotFoundError
            If no path exists between the source and target coordinate systems.
        """
        with suppress_direct_internal_attribute_access_warning():
            try:
                return [self.get_existing_transformation(source_cs=source_cs, target_cs=target_cs)]
            except TransformationNotFoundError as _tnfe:
                pass

            try:
                path = nx.shortest_path(self.graph, source=source_cs, target=target_cs)

            except nx.NetworkXNoPath as nxe:
                raise TransformationPathNotFoundError(source_cs.name, target_cs.name) from nxe

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
        with suppress_direct_internal_attribute_access_warning():
            paths = list(nx.all_simple_paths(self.graph, source=source_cs, target=target_cs))

            all_sequences = []
            for path in paths:
                sequence = []
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i + 1]]
                    sequence.append(edge_data[0][TRANSFORM_KEY])
                all_sequences.append(sequence)
            return all_sequences

    def _get_transformations_associated_with_cs(self, cs: NgffCoordinateSystem) -> list[BaseTransformation]:
        """
        Get all transformations associated with a coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to check.

        Returns
        -------
        List of transformations
        """
        self.check_if_coordinate_system_exists_else_raise_error(cs)

        with suppress_direct_internal_attribute_access_warning():
            transformations = []
            # Check outgoing edges (cs -> other)
            for successor in self.graph.successors(cs):
                transformation = self.get_existing_transformation(source_cs=cs, target_cs=successor)
                transformations.append(transformation)
            # Check incoming edges (other -> cs)
            for predecessor in self.graph.predecessors(cs):
                transformation = self.get_existing_transformation(source_cs=predecessor, target_cs=cs)
                transformations.append(transformation)

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

        Raises
        ------
        CoordinateSystemNotFoundError
            If the coordinate system does not exist.

        """
        self.check_if_coordinate_system_exists_else_raise_error(cs)

        with suppress_direct_internal_attribute_access_warning():
            elements = []
            for element_name, element_cs in self.element_to_cs_mapping.items():
                if element_cs == cs:
                    elements.append(element_name)

            return elements

    def __repr__(self) -> str:
        """Return a string representation of the TransformationManager."""
        with suppress_direct_internal_attribute_access_warning():
            return (
                f"TransformationManager("
                f"  coordinate_systems={list(self.graph.nodes())}, "
                f"  coordinate_transforms={[x[TRANSFORM_KEY] for *_, x in self.graph.edges(data=True)]}, "
                f"  elements={list(self.element_to_cs_mapping.keys())}"
                f")"
            )
