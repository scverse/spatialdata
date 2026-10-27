from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

from spatialdata.models import SpatialElement
from spatialdata.transformations import BaseTransformation
from spatialdata.transformations._graph import (
    BaseTransformationEdge,
    CoordinateSystemNotFoundError,
    CoordSystem,
    ElementNotRegisteredToAnyCoordinateSystemError,
    ProjectAxisEdge,
    TransformationGraph,
    get_instrinsic_coordinate_system,
)
from spatialdata.transformations._graph.utils import get_intrinsic_coordinate_system_name, get_union_of_axes


class TransformationsFacade:
    def __init__(self) -> None:
        """Initialize a TransformationsFacade with an empty transformation graph."""
        self._graph = TransformationGraph()

    def _get_element_transformation_edges(
        self, element_name: str, target_coordinate_system_names: Sequence[str] | None = None
    ) -> dict[str, BaseTransformationEdge]:
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
        dict, with edge keys as keys and transformations from the element's coordinate system as values.
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
        element_cs = get_instrinsic_coordinate_system(
            element_to_be_added_to_graph=new_element,
            coordinate_system_name=get_intrinsic_coordinate_system_name(new_element_key),
        )

        self._graph.add_coordinate_system(element_cs)
        self._graph.add_element(element_name=new_element_key, coordinate_system=element_cs)

    def _remove_element_from_graph_if_exists(self, element_name: str) -> None:
        """
        Remove element from the graph if it exists, do nothing otherwise

        when removing also removes all incoming and outgoing edges from the coordinate system to which the
        element belongs, then the coordinate system itself as well as the element-coordinate system mapping

        Parameters
        ----------
        element_name
            The name of the element to remove.
        """
        try:
            element_coordinate_system = self._graph.get_element_coordinate_system(element_name)
        except ElementNotRegisteredToAnyCoordinateSystemError as _err:
            return  # element does not exist, do nothing

        number_of_elements_in_element_cs = len(
            self._graph.get_elements_in_coordinate_system(element_coordinate_system.name)
        )

        if number_of_elements_in_element_cs == 1:
            for incoming_edge_key, incoming_edge in self._graph.get_incoming_edges(element_name).items():
                self._graph.remove_specific_transformation(incoming_edge_key, incoming_edge)

            for outgoing_edge_key, outgoing_edge in self._graph.get_outgoing_edges(element_name).items():
                self._graph.remove_specific_transformation(outgoing_edge_key, outgoing_edge)

            self._graph._unset_element(element_name)
            self._graph.remove_coordinate_system(element_coordinate_system)
        else:
            self._graph._unset_element(element_name)

    def set_transformations_from_elements_to_common_cs(
        self,
        target_cs_name: str,
        element_name_transformation_config: dict[str, BaseTransformation],
        make_target_cs_virtual_when_creating: bool = False,
    ) -> None:
        """
        Infer and add the transformation edges needed to align a group of spatial elements to a common coordinate system

        In addition to the transformations specified, adds project axis edges as needed.

        Parameters
        ----------
        target_cs_name
            Name of the target coordinate system. Will be created if it doesn't exist.
        element_name_transformation_config
            Dictionary mapping names of spatial elements to transformations that map the corresponding element to
            the target coordinate system.
        make_target_cs_virtual_when_creating
            whether to make target coordinate system virtual, if and when it gets created.
        """
        intermediate_edge_list: list[BaseTransformationEdge] = []
        for element_name, transformation in element_name_transformation_config.items():
            element_cs = self._graph.get_element_coordinate_system(element_name)
            intermediate_edge = transformation._to_ngff_transformation_edge_creating_output_cs(
                input_coordinate_system=element_cs,
                output_coordinate_system_name=f"{target_cs_name}_{element_name}",
                output_coordinate_system_virtual=True,
            )
            # if a project axis edge is needed for this element, this edge can be used below
            # hence it has a virtual output cs with a specially encoded name.
            intermediate_edge_list.append(intermediate_edge)

        expected_output_axes_flat = tuple(chain.from_iterable(x.output.axes for x in intermediate_edge_list))
        # basically flattens a tuple[tuple[T, ...], ...] into tuple[T]
        expected_output_axes_union = get_union_of_axes(expected_output_axes_flat)

        target_cs = self._graph.get_coordinate_system(target_cs_name)
        if target_cs is None:
            new_target_cs = CoordSystem(
                name=target_cs_name, axes=expected_output_axes_union, virtual=make_target_cs_virtual_when_creating
            )
            self._graph.add_coordinate_system(new_target_cs)
        else:
            new_target_cs_axes = get_union_of_axes(target_cs.axes + expected_output_axes_union)
            created_outputs = set(new_target_cs_axes) - set(target_cs.axes)
            if created_outputs:
                new_target_cs = CoordSystem(
                    name=target_cs_name,
                    axes=new_target_cs_axes,
                    virtual=target_cs.virtual,
                    # same "bag" of coordinate systems, so inherits this attribute
                )
                self._graph.add_coordinate_system(new_target_cs)
                self._graph.add_transformation_edge(
                    ProjectAxisEdge(
                        input=target_cs, output=new_target_cs, created_outputs=created_outputs, dropped_inputs=set()
                    )
                )
            else:
                new_target_cs = target_cs

        for ind, (_element_name, transformation) in enumerate(element_name_transformation_config.items()):
            created_outputs = set(new_target_cs.axes) - set(intermediate_edge_list[ind].output.axes)

            if created_outputs:
                self._graph.add_coordinate_system(intermediate_edge_list[ind].output)
                self._graph.add_transformation_edge(intermediate_edge_list[ind])
                # create intermediate virtual cs, create and add edge between element and this virtual cs

                project_axis_edge = ProjectAxisEdge(
                    input=intermediate_edge_list[ind].output,
                    output=new_target_cs,
                    created_outputs=created_outputs,
                    dropped_inputs=set(),
                )
                self._graph.add_transformation_edge(project_axis_edge)
                # add an extra projection edge from virtual cs to new_target_cs
            else:
                transformation_edge = transformation._to_ngff_transformation_edge_existing_output_cs(
                    input_coordinate_system=intermediate_edge_list[ind].input,
                    output_coordinate_system=new_target_cs,
                )
                self._graph.add_transformation_edge(transformation_edge)
                # create and add edge based on element_name and transformation to new_target_cs

    def remove_transformations(self, transformation_edges: dict[str, BaseTransformationEdge]) -> None:
        """
        Remove specific transformation edges from the graph.

        Parameters
        ----------
        transformation_edges
            dictionary with edge keys as keys and edges as values; those to be removed
        """
        for edge_key, transformation_edge in transformation_edges.items():
            self._graph.remove_specific_transformation(edge_key, transformation_edge)

    def remove_all_outgoing_transformations_of_element(self, element_name: str) -> None:
        """
        Remove all outgoing transformation edges of an element.

        Parameters
        ----------
        element_name
            The name of the element whose transformations should be removed.
        """
        all_outgoing_edges_from_element = self._graph.get_outgoing_edges(element_name)
        self.remove_transformations(all_outgoing_edges_from_element)

    def remove_all_incoming_transformations_of_element(self, element_name: str) -> None:
        """
        Remove all incoming transformation edges of an element.

        Parameters
        ----------
        element_name
            The name of the element whose transformations should be removed.
        """
        all_incoming_edges_from_element = self._graph.get_incoming_edges(element_name)
        self.remove_transformations(all_incoming_edges_from_element)

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
