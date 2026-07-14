from __future__ import annotations

from typing import Any

from spatialdata._types import ELEMENT_TYPE
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata.transformations.transformations import BaseTransformation


class TransformationManager:
    """Centralized storage for managing coordinate systems (cs), transformations and element-cs mapping."""

    def __init__(self) -> None:
        """Initialize a Scene with empty mappings."""
        self._coordinate_systems: dict[str, NgffCoordinateSystem] = {}
        # mapping names of coordinate system to coordinate systems
        self._coordinate_transforms: dict[tuple[str, str], BaseTransformation] = {}
        # mapping a tuple of coordinate system names, (<source coordinate system name>, <target coordinate system name>)
        # to a transformation object representing the transformation between them
        self._element_to_cs_mapping: dict[tuple[ELEMENT_TYPE, str], str] = {}
        # mapping a tuple, (<element type>, <element key in parent spatialdata object>) to the name of the coordinate
        # system

    def add_coordinate_system(self, cs: NgffCoordinateSystem) -> None:
        """
        Register a new coordinate system.

        Parameters
        ----------
        cs
            The coordinate system to register.

        Raises
        ------
        ValueError
            If a coordinate system with the same name already exists.
        """
        if cs.name in self._coordinate_systems:
            raise ValueError(f"Coordinate system with name '{cs.name}' already exists.")
        self._coordinate_systems[cs.name] = cs

    def _get_transformations_associated_with_cs(self, cs_name: str) -> list[tuple[str, str]]:
        """
        Get all transformations associated with a coordinate system.

        Parameters
        ----------
        cs_name
            The name of the coordinate system.

        Returns
        -------
        list[tuple[str, str]]
            A list of tuples representing the transformations associated with the coordinate system.
        """
        associated_transformations = []
        for input_cs, output_cs in self._coordinate_transforms:
            transformation_key = (input_cs, output_cs)
            if (input_cs == cs_name or output_cs == cs_name) and transformation_key not in associated_transformations:
                associated_transformations.append(transformation_key)
        return associated_transformations

    def _get_elements_associated_with_cs(self, cs_name: str) -> list[tuple[ELEMENT_TYPE, str]]:
        """
        Get all elements associated with a coordinate system.

        Parameters
        ----------
        cs_name
            The name of the coordinate system.

        Returns
        -------
        list[tuple[ELEMENT_TYPE, str]]
            A list of tuples representing the elements associated with the coordinate system.
        """
        associated_elements = []
        for (element_type, element_name), element_cs in self._element_to_cs_mapping.items():
            if element_cs == cs_name:
                associated_elements.append((element_type, element_name))
        return associated_elements

    def remove_coordinate_system(self, cs_name: str) -> None:
        """
        Remove an existing coordinate system.

        Parameters
        ----------
        cs_name
            The name of the coordinate system to remove.

        Raises
        ------
        KeyError
            If the coordinate system does not exist.
        ValueError
            If there are transformations or elements associated with the coordinate system.
        """
        if cs_name not in self._coordinate_systems:
            raise KeyError(f"Coordinate system with name '{cs_name}' not found.")

        associated_transformations = self._get_transformations_associated_with_cs(cs_name)
        associated_elements = self._get_elements_associated_with_cs(cs_name)

        # Raise error if there are associated transformations or elements
        if len(associated_transformations) or len(associated_elements):
            raise ValueError(
                f"Cannot remove coordinate system with name '{cs_name}'. "
                f"{len(associated_elements)} elements and {len(associated_transformations)} transformations"
                f" are associated with it. "
                f"Please remove transformations or disassociate elements from the coordinate system first."
            )

        del self._coordinate_systems[cs_name]

    def list_coordinate_systems(self) -> list[str]:
        """
        List all registered coordinate system names, ordered as they were added.

        Returns
        -------
        A list of coordinate system names.
        """
        return list(self._coordinate_systems.keys())

    def add_element(self, element_type: ELEMENT_TYPE, element_name: str, coordinate_system: str) -> None:
        """
        Register an element and associate it with a coordinate system.

        Parameters
        ----------
        element_type
            The type of the element (e.g., 'images', 'labels', 'points', 'shapes').
        element_name
            The name of the element.
        coordinate_system
            The name of the coordinate system to which the element belongs. If None, a new coordinate system
            will be created with the name "<element_name>_created_at_insert".

        Raises
        ------
        KeyError
            If the coordinate system does not exist.
        """
        if coordinate_system not in self._coordinate_systems:
            raise KeyError(
                f"Cannot set coordinate system ('{coordinate_system}') to element as the "
                f"coordinate system does not exist."
            )

        mapping_key = (element_type, element_name)
        self._element_to_cs_mapping[mapping_key] = coordinate_system

    def get_element_coordinate_system(self, element_type: ELEMENT_TYPE, element_name: str) -> str | None:
        """
        Get the name of the coordinate system to which an element belongs.

        Parameters
        ----------
        element_type
            The type of the element.
        element_name
            The name of the element.

        Returns
        -------
        The name of the coordinate system or None if not found

        """
        mapping_key = (element_type, element_name)
        return self._element_to_cs_mapping.get(mapping_key)

    def add_transformation(self, input_cs: str, output_cs: str, transformation: BaseTransformation) -> None:
        """
        Add a transformation between coordinate systems.

        Parameters
        ----------
        input_cs
            The name of the input coordinate system.
        output_cs
            The name of the output coordinate system.
        transformation
            The transformation to add.

        Raises
        ------
        ValueError
            If either coordinate system does not exist.
        """
        if input_cs not in self._coordinate_systems:
            raise ValueError(
                f"Input coordinate system '{input_cs}' does not exist."
                f" Please create it before adding a transform associated with it"
            )
        if output_cs not in self._coordinate_systems:
            raise ValueError(
                f"Output coordinate system '{output_cs}' does not exist."
                f" Please create it before adding a transform associated with it"
            )

        key = (input_cs, output_cs)
        self._coordinate_transforms[key] = transformation

    def get_existing_transformation(self, input_cs: str, output_cs: str) -> BaseTransformation | None:
        """
        Retrieve a transformation defined between coordinate systems.

        Parameters
        ----------
        input_cs
            The name of the input coordinate system.
        output_cs
            The name of the output coordinate system.

        Returns
        -------
        The transformation

        Raises
        ------
        KeyError:
            if no transformation exists from input_cs to output_cs.
        """
        key = (input_cs, output_cs)
        return self._coordinate_transforms.get(key)

    def remove_transformation(self, input_cs: str, output_cs: str) -> None:
        """
        Remove a transformation between coordinate systems.

        Parameters
        ----------
        input_cs
            The name of the input coordinate system.
        output_cs
            The name of the output coordinate system.

        Raises
        ------
        KeyError
            If the transformation does not exist.
        """
        key = (input_cs, output_cs)
        if key not in self._coordinate_transforms:
            raise KeyError(f"Transformation from '{input_cs}' to '{output_cs}' not found.")
        del self._coordinate_transforms[key]

    def build_nx_graph(self) -> Any:  # type: ignore[unresolved-reference]  # noqa: F821
        # nx lazily imported
        """
        Build a directed graph where nodes are coordinate systems and edges are transformations.

        Returns
        -------
        nx.DiGraph
            A directed graph representing the coordinate systems and transformations.
        """
        import networkx as nx

        g = nx.DiGraph()
        for cs_name in self._coordinate_systems:
            g.add_node(cs_name)
        for (input_cs, output_cs), transformation in self._coordinate_transforms.items():
            g.add_edge(input_cs, output_cs, transformation=transformation)
        return g

    def get_shortest_transformation_sequence(self, source_cs: str, target_cs: str) -> list[BaseTransformation]:
        """
        Get the shortest sequence of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The name of the source coordinate system.
        target_cs
            The name of the target coordinate system.

        Returns
        -------
        list[BaseTransformation]
            The shortest sequence of transformations from source_cs to target_cs.

        Raises
        ------
        ValueError
            If no path exists between the source and target coordinate systems.
        """
        if (source_cs, target_cs) in self._coordinate_transforms:
            return [self._coordinate_transforms[(source_cs, target_cs)]]

        g = self.build_nx_graph()
        import networkx as nx

        try:
            path = nx.shortest_path(g, source=source_cs, target=target_cs)  # type: ignore[name-defined, unresolved-reference]  # noqa: F821
            # nx lazily imported
        except nx.NetworkXNoPath as nxe:  # type: ignore[name-defined, unresolved-reference]  # noqa: F821
            raise ValueError(f"No path found from {source_cs} to {target_cs}") from nxe

        transformations = []
        for i in range(len(path) - 1):
            transformations.append(g[path[i]][path[i + 1]]["transformation"])
        return transformations

    def get_all_transformation_sequences(self, source_cs: str, target_cs: str) -> list[list[BaseTransformation]]:
        """
        Get all existing sequences of transformations between two coordinate systems.

        Parameters
        ----------
        source_cs
            The name of the source coordinate system.
        target_cs
            The name of the target coordinate system.

        Returns
        -------
        list[list[BaseTransformation]]
            All existing sequences of transformations from source_cs to target_cs.
        """
        g = self.build_nx_graph()
        import networkx as nx

        paths = list(nx.all_simple_paths(g, source=source_cs, target=target_cs))

        all_sequences = []
        for path in paths:
            sequence = []
            for i in range(len(path) - 1):
                sequence.append(g[path[i]][path[i + 1]]["transformation"])
            all_sequences.append(sequence)
        return all_sequences

    def __repr__(self) -> str:
        return (
            f"TransformationManager("
            f"  coordinate_systems={list(self._coordinate_systems.keys())}, "
            f"  coordinate_transforms={list(self._coordinate_transforms.keys())}, "
            f"  elements={list(self._element_to_cs_mapping.keys())}"
            f")"
        )
