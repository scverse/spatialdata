#!/usr/bin/env python3

"""
Unit tests for the TransformationManager class in _core/transformation_manager/__init__.py.

This module tests all the functionality of the TransformationManager class to achieve 100% coverage.
"""

from __future__ import annotations

import pytest

from spatialdata import TransformationManager
from spatialdata._types import ELEMENT_TYPE


def test_initialization():
    """Test that TransformationManager initializes correctly."""
    tm = TransformationManager()
    assert tm._coordinate_systems == {}
    assert tm._coordinate_transforms == {}
    assert tm._element_to_cs_mapping == {}


def test_add_coordinate_system(one_point_graph):
    """Test adding a coordinate system."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    assert cs.name in tm._coordinate_systems
    assert tm._coordinate_systems[cs.name] == cs


def test_add_coordinate_system_duplicate(one_point_graph):
    """Test that adding a duplicate coordinate system raises ValueError."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    with pytest.raises(ValueError, match=f"Coordinate system with name '{cs.name}' already exists"):
        tm.add_coordinate_system(cs)


def test_get_transformations_associated_with_cs(fully_connected_two_point_graph):
    """Test getting transformations associated with a coordinate system."""
    tm = TransformationManager()
    coordinate_systems, transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    transform = transformations[0]
    tm.add_transformation(cs1.name, cs2.name, transform)

    associated = tm._get_transformations_associated_with_cs(cs1.name)
    assert associated == [(cs1.name, cs2.name)]


def test_get_elements_associated_with_cs(one_point_graph):
    """Test getting elements associated with a coordinate system."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.add_element(ELEMENT_TYPE.IMAGE, "image1", cs.name)

    associated = tm._get_elements_associated_with_cs(cs.name)
    assert associated == [(ELEMENT_TYPE.IMAGE, "image1")]


def test_remove_coordinate_system(one_point_graph):
    """Test removing a coordinate system."""
    tm = TransformationManager()

    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.remove_coordinate_system(cs.name)
    assert cs.name not in tm._coordinate_systems


def test_remove_coordinate_system_nonexistent():
    """Test that removing a non-existent coordinate system raises KeyError."""
    tm = TransformationManager()
    with pytest.raises(KeyError, match="Coordinate system with name 'cs1' not found"):
        tm.remove_coordinate_system("cs1")


def test_remove_coordinate_system_with_associations(fully_connected_two_point_graph):
    """Test that removing a coordinate system with associations raises ValueError."""
    tm = TransformationManager()
    coordinate_systems, _transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    transform = _transformations[0]
    tm.add_transformation(cs1.name, cs2.name, transform)

    with pytest.raises(ValueError, match="Cannot remove coordinate system"):
        tm.remove_coordinate_system(cs1.name)


def test_remove_coordinate_system_with_element_associations(one_point_graph):
    """Test that removing a coordinate system with element associations raises ValueError."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.add_element(ELEMENT_TYPE.IMAGE, "image1", cs.name)

    with pytest.raises(ValueError, match="Cannot remove coordinate system"):
        tm.remove_coordinate_system(cs.name)


@pytest.mark.parametrize("cs_names", [["cs1", "cs2"]])
def test_list_coordinate_systems(fully_connected_two_point_graph, cs_names):
    """Test listing all coordinate systems."""
    tm = TransformationManager()
    coordinate_systems, _ = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    systems = tm.list_coordinate_systems()
    assert set(systems) == {cs1.name, cs2.name}


def test_add_element(one_point_graph):
    """Test adding an element with an existing coordinate system."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.add_element(ELEMENT_TYPE.IMAGE, "image1", cs.name)

    mapping = tm._element_to_cs_mapping
    assert (ELEMENT_TYPE.IMAGE, "image1") in mapping
    assert mapping[(ELEMENT_TYPE.IMAGE, "image1")] == cs.name


def test_add_element_nonexistent_cs():
    """Test that adding an element with a non-existent coordinate system raises KeyError."""
    tm = TransformationManager()
    with pytest.raises(KeyError, match="Cannot set coordinate system"):
        tm.add_element(ELEMENT_TYPE.IMAGE, "image1", "nonexistent_cs")


def test_get_element_coordinate_system(one_point_graph):
    """Test getting the coordinate system of an element."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.add_element(ELEMENT_TYPE.IMAGE, "image1", cs.name)

    cs_name = tm.get_element_coordinate_system(ELEMENT_TYPE.IMAGE, "image1")
    assert cs_name == cs.name


def test_get_element_coordinate_system_nonexistent():
    """Test getting the coordinate system of a non-existent element."""
    tm = TransformationManager()
    cs_name = tm.get_element_coordinate_system(ELEMENT_TYPE.IMAGE, "nonexistent")
    assert cs_name is None


@pytest.mark.parametrize("cs_names", [["cs1", "cs2"]])
def test_add_transformation(fully_connected_two_point_graph, cs_names):
    """Test adding a transformation between coordinate systems."""
    tm = TransformationManager()
    coordinate_systems, transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    transform = transformations[0]
    tm.add_transformation(cs1.name, cs2.name, transform)

    key = (cs1.name, cs2.name)
    assert key in tm._coordinate_transforms
    assert tm._coordinate_transforms[key] == transform


@pytest.mark.parametrize("cs_names", [["cs1", "cs2"]])
def test_add_transformation_nonexistent_cs(fully_connected_two_point_graph, cs_names):
    """Test that adding a transformation with non-existent coordinate systems raises ValueError."""
    tm = TransformationManager()
    coordinate_systems, transformations = fully_connected_two_point_graph
    transform = transformations[0]
    cs1, cs2 = coordinate_systems

    with pytest.raises(ValueError, match=f"Input coordinate system '{cs1.name}' does not exist"):
        tm.add_transformation(cs1.name, cs2.name, transform)

    # Add one coordinate system
    tm.add_coordinate_system(cs1)

    with pytest.raises(ValueError, match=f"Output coordinate system '{cs2.name}' does not exist"):
        tm.add_transformation(cs1.name, cs2.name, transform)


@pytest.mark.parametrize("cs_names", [["cs1", "cs2"]])
def test_get_existing_transformation(fully_connected_two_point_graph, cs_names):
    """Test getting an existing transformation."""
    tm = TransformationManager()
    coordinate_systems, transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    transform = transformations[0]
    tm.add_transformation(cs1.name, cs2.name, transform)

    retrieved = tm.get_existing_transformation(cs1.name, cs2.name)
    assert retrieved == transform


def test_get_existing_transformation_nonexistent(fully_connected_two_point_graph):
    """Test getting a non-existent transformation."""
    tm = TransformationManager()
    coordinate_systems, _transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    retrieved = tm.get_existing_transformation(cs1.name, cs2.name)
    assert retrieved is None


@pytest.mark.parametrize("cs_names", [["cs1", "cs2"]])
def test_remove_transformation(fully_connected_two_point_graph, cs_names):
    """Test removing a transformation."""
    tm = TransformationManager()
    coordinate_systems, transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    transform = transformations[0]
    tm.add_transformation(cs1.name, cs2.name, transform)

    tm.remove_transformation(cs1.name, cs2.name)
    assert (cs1.name, cs2.name) not in tm._coordinate_transforms


def test_remove_transformation_nonexistent():
    """Test that removing a non-existent transformation raises KeyError."""
    tm = TransformationManager()
    with pytest.raises(KeyError, match="Transformation from 'cs1' to 'cs2' not found"):
        tm.remove_transformation("cs1", "cs2")


def test_build_nx_graph(four_point_graph):
    """Test building a networkx graph from the transformation manager."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3
    transform3 = transformations[2]  # cs3 -> cs4
    transform4 = transformations[3]  # cs1 -> cs3
    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)
    tm.add_transformation(cs3.name, cs4.name, transform3)
    tm.add_transformation(cs1.name, cs3.name, transform4)

    g = tm.build_nx_graph()
    assert g.has_node(cs1.name)
    assert g.has_node(cs2.name)
    assert g.has_node(cs3.name)
    assert g.has_node(cs4.name)
    assert g.has_edge(cs1.name, cs2.name)
    assert g.has_edge(cs2.name, cs3.name)
    assert g.has_edge(cs3.name, cs4.name)
    assert g.has_edge(cs1.name, cs3.name)
    assert g[cs1.name][cs2.name]["transformation"] == transform1
    assert g[cs2.name][cs3.name]["transformation"] == transform2
    assert g[cs3.name][cs4.name]["transformation"] == transform3
    assert g[cs1.name][cs3.name]["transformation"] == transform4


def test_get_shortest_transformation_sequence_direct(four_point_graph):
    """Test getting the shortest transformation sequence for a direct transformation."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3
    transform4 = transformations[3]  # cs1 -> cs3
    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)
    tm.add_transformation(cs1.name, cs3.name, transform4)

    sequence = tm.get_shortest_transformation_sequence(cs1.name, cs3.name)
    assert sequence == [transform4]


def test_get_shortest_transformation_sequence_indirect(four_point_graph):
    """Test getting the shortest transformation sequence for an indirect transformation."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3
    transform3 = transformations[2]  # cs3 -> cs4
    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)
    tm.add_transformation(cs3.name, cs4.name, transform3)

    sequence = tm.get_shortest_transformation_sequence(cs1.name, cs3.name)
    assert sequence == [transform1, transform2]


def test_get_shortest_transformation_sequence_no_path(four_point_graph):
    """Test that getting a transformation sequence with no path raises ValueError."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3

    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)

    with pytest.raises(ValueError, match=f"No path found from {cs1.name} to {cs4.name}"):
        tm.get_shortest_transformation_sequence(cs1.name, cs4.name)


def test_get_all_transformation_sequences(four_point_graph):
    """Test getting all transformation sequences."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3
    transform3 = transformations[2]  # cs3 -> cs4
    transform4 = transformations[3]  # cs1 -> cs3
    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)
    tm.add_transformation(cs3.name, cs4.name, transform3)
    tm.add_transformation(cs1.name, cs3.name, transform4)

    sequences = tm.get_all_transformation_sequences(cs1.name, cs4.name)
    assert len(sequences) == 2
    assert [transform1, transform2, transform3] in sequences
    assert [transform4, transform3] in sequences


def test_get_all_transformation_sequences_no_path(four_point_graph):
    """Test that getting all transformation sequences with no path returns an empty list."""
    tm = TransformationManager()
    coordinate_systems, transformations = four_point_graph
    cs1, cs2, cs3, cs4 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    transform1 = transformations[0]  # cs1 -> cs2
    transform2 = transformations[1]  # cs2 -> cs3

    tm.add_transformation(cs1.name, cs2.name, transform1)
    tm.add_transformation(cs2.name, cs3.name, transform2)

    sequences = tm.get_all_transformation_sequences(cs1.name, cs4.name)
    assert sequences == []


def test_repr(one_point_graph):
    """Test the string representation of TransformationManager."""
    tm = TransformationManager()
    coordinate_systems, _ = one_point_graph
    cs = coordinate_systems[0]
    tm.add_coordinate_system(cs)
    tm.add_element(ELEMENT_TYPE.IMAGE, "image1", cs.name)

    repr_str = repr(tm)
    assert "TransformationManager" in repr_str
    assert cs.name in repr_str
    assert "image1" in repr_str
