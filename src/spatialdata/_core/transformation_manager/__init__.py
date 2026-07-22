from __future__ import annotations

import warnings
from collections.abc import Sequence

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
    TransformationPathAmbiguousError,
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

    def check_if_any_edge_exists_else_raise_error(
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

    @staticmethod
    def _get_edge_key_from_transform(transform: BaseTransformation) -> str:

        return repr(transform)

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
            edge_key = self._get_edge_key_from_transform(transformation)
            edge_attributes = {TRANSFORM_KEY: transformation}
            self.graph.add_edge(source_cs, target_cs, key=edge_key, **edge_attributes)

    def get_existing_direct_transformations(
        self, source_cs: NgffCoordinateSystem, target_cs: NgffCoordinateSystem
    ) -> list[BaseTransformation]:
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
        List of transformations

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the transformation does not exist.
        """
        self.check_if_any_edge_exists_else_raise_error(source_cs, target_cs)
        # also checks if source_cs and target_cs exist
        with suppress_direct_internal_attribute_access_warning():
            transforms = []
            assert target_cs in self.graph[source_cs], TransformationNotFoundError(source_cs.name, target_cs.name)
            for edge_data in self.graph[source_cs][target_cs].values():
                transform: BaseTransformation = edge_data[TRANSFORM_KEY]
                transforms.append(transform)
            return transforms

    def remove_specific_transformation(
        self,
        source_cs: NgffCoordinateSystem,
        target_cs: NgffCoordinateSystem,
        transformation: BaseTransformation,
    ) -> None:
        """
        Remove a specific transformation between coordinate systems.

        Parameters
        ----------
        source_cs
            The input coordinate system.
        target_cs
            The output coordinate system.
        transformation
            The transformation to remove.
            (mainly useful for cases with multiple transformations between the same coordinate systems).

        Raises
        ------
        CoordinateSystemNotFoundError
            If either coordinate system does not exist.
        TransformationNotFoundError
            If the transformation does not exist.
        """
        self.check_if_any_edge_exists_else_raise_error(source_cs, target_cs)
        # also checks if source_cs and target_cs exist
        with suppress_direct_internal_attribute_access_warning():
            expected_edge_key = self._get_edge_key_from_transform(transformation)
            assert expected_edge_key in self.graph[source_cs][target_cs], TransformationNotFoundError(
                source_cs.name, target_cs.name, expected_edge_key
            )
            self.graph.remove_edge(source_cs, target_cs, key=expected_edge_key)

    def remove_all_transformations_between_coordinate_systems(
        self,
        source_cs: NgffCoordinateSystem,
        target_cs: NgffCoordinateSystem,
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
        self.check_if_any_edge_exists_else_raise_error(source_cs, target_cs)
        # also checks if source_cs and target_cs exist
        with suppress_direct_internal_attribute_access_warning():
            assert len(self.graph[source_cs][target_cs]), TransformationNotFoundError(source_cs.name, target_cs.name)
            for edge_key in list(self.graph[source_cs][target_cs].keys()):
                # need to covert keys() to list to freeze it, else it will change during the following removal
                self.graph.remove_edge(source_cs, target_cs, key=edge_key)

    def _get_transformation_sequences_from_path_after_disambiguation(
        self,
        paths: list[list[NgffCoordinateSystem]],
        expected_intermediate_transformations: list[BaseTransformation] | None,
    ) -> list[list[BaseTransformation]]:
        """
        Traverses paths to form sequence of Transformations.

        In case of ambiguity looks into `expected_intermediate_transformations` to disambiguate.

        Parameters
        ----------
        paths:
            sequence of list of nodes
        expected_intermediate_transformations:
            list of transformation objects

        Returns
        -------
            list of sequences of transformations
        """
        intermediate_transformation_edge_keys = set()
        if expected_intermediate_transformations is not None:
            intermediate_transformation_edge_keys |= {
                self._get_edge_key_from_transform(it) for it in expected_intermediate_transformations
            }
        all_sequences = []
        deduplicated_paths = list({repr(x): x for x in paths}.values())
        for path in deduplicated_paths:
            sequence = []
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i + 1]]
                if len(edge_data) > 1:
                    # when there are multiple edges between a pair of coordinate systems in the path
                    intermediate_transformation_key_here = intermediate_transformation_edge_keys & set(edge_data.keys())
                    if len(intermediate_transformation_key_here) == 0:
                        # transformation was not specified in `intermediate_transformations` for disambiguation
                        raise TransformationPathAmbiguousError(path[i].name, path[i + 1].name)

                    edge_key_to_use = list(intermediate_transformation_key_here)[0]
                    # choosing the first one arbitrarily
                    sequence.append(edge_data[edge_key_to_use][TRANSFORM_KEY])
                else:
                    # Only one edge, no ambiguity
                    edge_key = next(iter(edge_data.keys()))
                    sequence.append(edge_data[edge_key][TRANSFORM_KEY])
            all_sequences.append(sequence)
        return all_sequences

    def get_all_shortest_transformation_sequences(
        self,
        source_cs: NgffCoordinateSystem,
        target_cs: NgffCoordinateSystem,
        expected_intermediate_transformations: list[BaseTransformation] | None = None,
    ) -> list[list[BaseTransformation]]:
        """
        Get all shortest sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.
        expected_intermediate_transformations
            list of intermediate transformations.
            Used to choose an edge when multiple edges are found between the same coordinate systems.

        Returns
        -------
        list[list[BaseTransformation]]
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
        with suppress_direct_internal_attribute_access_warning():
            try:
                paths = list(nx.all_shortest_paths(self.graph, source=source_cs, target=target_cs))

            except nx.NetworkXNoPath as nxe:
                raise TransformationPathNotFoundError(source_cs.name, target_cs.name) from nxe

            try:
                return self._get_transformation_sequences_from_path_after_disambiguation(
                    paths, expected_intermediate_transformations
                )
            except TransformationPathAmbiguousError as tpae:
                raise tpae from TransformationPathAmbiguousError(source_cs.name, target_cs.name)

    def get_all_transformation_sequences(
        self,
        source_cs: NgffCoordinateSystem,
        target_cs: NgffCoordinateSystem,
        expected_intermediate_transformations: list[BaseTransformation] | None = None,
    ) -> list[list[BaseTransformation]]:
        """
        Get all existing sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The source coordinate system.
        target_cs
            The target coordinate system.
        expected_intermediate_transformations
            list of intermediate transformations.
            Used to choose an edge when multiple edges are found between the same coordinate systems.

        Returns
        -------
        list[list[BaseTransformation]]
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
        with suppress_direct_internal_attribute_access_warning():
            try:
                paths = list(nx.all_simple_paths(self.graph, source=source_cs, target=target_cs))

            except nx.NetworkXNoPath as nxe:
                raise TransformationPathNotFoundError(source_cs.name, target_cs.name) from nxe

            try:
                return self._get_transformation_sequences_from_path_after_disambiguation(
                    paths, expected_intermediate_transformations
                )
            except TransformationPathAmbiguousError as tpae:
                raise tpae from TransformationPathAmbiguousError(source_cs.name, target_cs.name)

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
                transformations_outgoing = self.get_existing_direct_transformations(cs, successor)
                transformations += transformations_outgoing
            # Check incoming edges (other -> cs)
            for predecessor in self.graph.predecessors(cs):
                transformations_incoming = self.get_existing_direct_transformations(source_cs=predecessor, target_cs=cs)
                transformations += transformations_incoming

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
