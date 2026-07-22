#!/usr/bin/env python3

"""
Unit tests for the TransformationManager class in _core/transformation_manager/__init__.py.

This module tests all the functionality of the TransformationManager class to achieve 100% coverage.
"""

from __future__ import annotations

import pytest

from spatialdata._core.transformation_manager import TRANSFORM_KEY, TransformationManager
from spatialdata._core.transformation_manager.exceptions import (
    CannotRemoveCoordinateSystemError,
    CoordinateSystemAlreadyExistsError,
    CoordinateSystemNotFoundError,
    ElementAlreadyExistsError,
    ElementNotFoundError,
    TransformationNotFoundError,
    TransformationPathAmbiguousError,
    TransformationPathNotFoundError,
    suppress_direct_internal_attribute_access_warning,
)


def test_initialization():
    """Test that TransformationManager initializes correctly."""
    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        assert len(tm.graph.nodes()) == 0
        assert len(tm.graph.edges()) == 0
        assert tm.element_to_cs_mapping == {}


def test_add_coordinate_system(one_point_graph):
    """Test adding a coordinate system."""
    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [cs1], _ = one_point_graph

        tm.add_coordinate_system(cs1)
        assert cs1 in tm.graph.nodes()


def test_add_coordinate_system_duplicate(one_point_graph):
    """Test adding an already existing coordinate system"""
    tm = TransformationManager()
    [cs1], _ = one_point_graph

    tm.add_coordinate_system(cs1)
    with pytest.raises(CoordinateSystemAlreadyExistsError, match=f"Coordinate system '{cs1.name}' already exists"):
        tm.add_coordinate_system(cs1)


def test_remove_coordinate_system(one_point_graph):
    """Test removing a coordinate system."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph

    tm.add_coordinate_system(cs1)
    tm.remove_coordinate_system(cs1)

    with suppress_direct_internal_attribute_access_warning():
        assert cs1 not in tm.graph.nodes()


def test_remove_coordinate_system_nonexistent(one_point_graph):
    """Test that removing a non-existent coordinate system raises CoordinateSystemNotFoundError."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph

    # Add the coordinate system first
    tm.add_coordinate_system(cs1)

    # Remove it
    tm.remove_coordinate_system(cs1)

    # Try to remove it again - should raise CoordinateSystemNotFoundError
    with pytest.raises(CoordinateSystemNotFoundError):
        tm.remove_coordinate_system(cs1)


def test_remove_coordinate_system_with_associated_transformations(fully_connected_two_point_graph):
    """
    Test that removing a coordinate system with associated transformations raises CannotRemoveCoordinateSystemError.
    """
    tm = TransformationManager()
    [cs1, cs2], [transformation] = fully_connected_two_point_graph

    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    tm.add_transformation(cs1, cs2, transformation)

    # Should raise CannotRemoveCoordinateSystem when trying to remove a coordinate system with transformations
    with pytest.raises(CannotRemoveCoordinateSystemError, match="Cannot remove coordinate system"):
        tm.remove_coordinate_system(cs1)

    # Should raise CannotRemoveCoordinateSystem when trying to remove a coordinate system with transformations
    with pytest.raises(CannotRemoveCoordinateSystemError, match="Cannot remove coordinate system"):
        tm.remove_coordinate_system(cs2)


def test_remove_coordinate_system_with_belonging_elements(one_point_graph):
    """Test that removing a coordinate system with element associations raises CannotRemoveCoordinateSystemError."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_element("image1", cs1)

    with pytest.raises(CannotRemoveCoordinateSystemError, match="Cannot remove coordinate system"):
        tm.remove_coordinate_system(cs1)


def test_list_coordinate_systems(fully_connected_two_point_graph):
    """Test listing all coordinate systems."""
    tm = TransformationManager()
    [cs1, cs2], _ = fully_connected_two_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    systems = tm.list_coordinate_systems()
    assert len(systems) == 2
    assert cs1 in systems
    assert cs2 in systems


def test_add_element(one_point_graph):
    """Test adding an element with an existing coordinate system."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_element("image1", cs1)

    with suppress_direct_internal_attribute_access_warning():
        mapping = tm.element_to_cs_mapping
        assert "image1" in mapping
        assert mapping["image1"] == cs1


def test_add_element_duplicate(one_point_graph):
    """Test that adding a duplicate element raises ElementAlreadyExistsError."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    element_name = "image1"
    tm.add_element(element_name, cs1)

    # Try to add the same element again - should raise ElementAlreadyExistsError
    with pytest.raises(
        ElementAlreadyExistsError, match=f"Element '{element_name}' already exists in the transformation manager"
    ):
        tm.add_element(element_name, cs1)


def test_unset_element(one_point_graph):
    """Test unsetting an element."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_element("image1", cs1)
    tm.unset_element("image1")

    with suppress_direct_internal_attribute_access_warning():
        assert "image1" not in tm.element_to_cs_mapping


def test_unset_element_nonexistent(one_point_graph):
    """Test that unsetting a non-existent element raises ElementNotFoundError."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    element_name = "image1"

    # Try to unset non-existent element
    with pytest.raises(
        ElementNotFoundError, match=f"Element '{element_name}' not found in the transformation manager."
    ):
        tm.unset_element(element_name)


def test_add_transformation(fully_connected_two_point_graph):
    """Test adding a transformation between coordinate systems."""
    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [cs1, cs2], [transform] = fully_connected_two_point_graph

        tm.add_coordinate_system(cs1)
        tm.add_coordinate_system(cs2)

        tm.add_transformation(cs1, cs2, transform)

        assert tm.graph.has_edge(cs1, cs2)
        # Get the first edge key and check the transformation
        edge_key = tm._get_edge_key_from_transform(transform)
        assert tm.graph[cs1][cs2][edge_key][TRANSFORM_KEY] == transform


def test_add_transformation_nonexistent_cs(fully_connected_two_point_graph):
    """Test that adding a transformation with non-existent coordinate systems raises CoordinateSystemNotFoundError."""
    tm = TransformationManager()
    [cs1, cs2], [transform] = fully_connected_two_point_graph

    with pytest.raises(
        CoordinateSystemNotFoundError, match=f"Coordinate system '{cs1.name}' not found in the transformation manager"
    ):
        tm.add_transformation(cs1, cs2, transform)

    # Add one coordinate system
    tm.add_coordinate_system(cs1)

    with pytest.raises(
        CoordinateSystemNotFoundError, match=f"Coordinate system '{cs2.name}' not found in the transformation manager"
    ):
        tm.add_transformation(cs1, cs2, transform)


def test_get_existing_transformation(fully_connected_two_point_graph):
    """Test getting an existing transformation."""
    tm = TransformationManager()
    [cs1, cs2], [transform] = fully_connected_two_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    tm.add_transformation(cs1, cs2, transform)

    retrieved = tm.get_existing_direct_transformations(cs1, cs2)
    assert retrieved == [transform]


def test_get_existing_transformation_nonexistent(fully_connected_two_point_graph):
    """Test getting a non-existent transformation."""
    tm = TransformationManager()
    coordinate_systems, _transformations = fully_connected_two_point_graph
    cs1, cs2 = coordinate_systems
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    with pytest.raises(
        TransformationNotFoundError, match=f"Transformation from '{cs1.name}' to '{cs2.name}' not found"
    ):
        tm.get_existing_direct_transformations(cs1, cs2)


def test_remove_all_transformations_between_coordinate_systems(fully_connected_two_point_graph):
    """Test removing all transformations  between coordinate systems."""
    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [cs1, cs2], [transform] = fully_connected_two_point_graph
        tm.add_coordinate_system(cs1)
        tm.add_coordinate_system(cs2)

        tm.add_transformation(cs1, cs2, transform)

        tm.remove_all_transformations_between_coordinate_systems(cs1, cs2)
        assert not tm.graph.has_edge(cs1, cs2)


def test_remove_all_transformation_nonexistent(fully_connected_two_point_graph):
    """Test that removing non-existent transformations between coordinate systems raises TransformationNotFoundError."""
    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [cs1, cs2], _ = fully_connected_two_point_graph
        tm.graph.add_node(cs1)
        tm.graph.add_node(cs2)
        with pytest.raises(
            TransformationNotFoundError, match=f"Transformation from '{cs1.name}' to '{cs2.name}' not found"
        ):
            tm.remove_all_transformations_between_coordinate_systems(cs1, cs2)


def test_remove_specific_transformation_between_coordinate_systems(five_point_graph):
    """Test removing specific transformation between coordinate systems."""

    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [_cs1, _cs2, cs3, _cs4, cs5], [_transform1, _transform2, _transform3, transform4, transform5] = five_point_graph
        tm.add_coordinate_system(cs3)
        tm.add_coordinate_system(cs5)

        tm.add_transformation(cs3, cs5, transform4)
        tm.add_transformation(cs3, cs5, transform5)

        tm.remove_specific_transformation(cs3, cs5, transform4)
        assert not tm.graph.has_edge(cs3, cs5, key=tm._get_edge_key_from_transform(transform4))


def test_remove_specific_transformation_between_coordinate_systems_non_existent(five_point_graph):
    """
    Test that removing non-existent specific transformation between coordinate systems raises
    TransformationNotFoundError.
    """

    with suppress_direct_internal_attribute_access_warning():
        tm = TransformationManager()
        [_cs1, _cs2, cs3, _cs4, cs5], [_transform1, _transform2, _transform3, transform4, transform5] = five_point_graph
        tm.add_coordinate_system(cs3)
        tm.add_coordinate_system(cs5)

        with pytest.raises(
            TransformationNotFoundError, match=f"Transformation from '{cs3.name}' to '{cs5.name}' not found"
        ):
            tm.remove_specific_transformation(cs3, cs5, transform4)


def test_get_all_shortest_transformation_sequences_direct(four_point_graph):
    """Test getting all shortest transformation sequences for a direct transformation path."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4], [transform1, transform2, _transform3, transform4] = four_point_graph

    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs3, transform4)

    sequences = tm.get_all_shortest_transformation_sequences(cs1, cs3)
    assert len(sequences) == 1
    assert [transform4] in sequences


def test_get_all_shortest_transformation_sequences_indirect_one_path(four_point_graph):
    """Test getting all shortest transformation sequences for one indirect transformation path."""
    tm = TransformationManager()
    [cs1, cs2, cs3, _cs4], [transform1, transform2, _transform3, _transform4] = four_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)

    sequences = tm.get_all_shortest_transformation_sequences(cs1, cs3)
    assert len(sequences) == 1
    assert [transform1, transform2] in sequences


def test_get_all_shortest_transformation_sequences_indirect_two_paths(five_point_graph):
    """Test getting all shortest transformation sequences for two indirect transformation paths."""
    tm = TransformationManager()
    ([cs1, cs2, cs3, cs4, _cs5], [transform1, transform2, _transform3, _transform4, _transform5]) = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)

    sequences = tm.get_all_shortest_transformation_sequences(cs1, cs3)
    assert len(sequences) == 2
    assert [transform1, transform2] in sequences
    assert [transform2, transform1] in sequences


def test_get_all_shortest_transformation_sequences_no_path(four_point_graph):
    """Test that getting all shortest transformation sequences with no path raises TransformationPathNotFoundError."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4], [transform1, transform2, _transform3, _transform4] = four_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)

    expected_error_msg = "No transformation path found from"
    with pytest.raises(TransformationPathNotFoundError, match=expected_error_msg):
        tm.get_all_shortest_transformation_sequences(cs1, cs4)


def test_get_all_shortest_transformation_sequences_multiple_paths_multiple_edges_success(five_point_graph):
    """Test getting all shortest transformation sequences, with multiple path and multiple edges between nodes."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4, cs5], [transform1, transform2, transform3, transform4, transform5] = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)
    tm.add_coordinate_system(cs5)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)
    # tm.add_transformation(cs1, cs3, transform3)
    tm.add_transformation(cs3, cs5, transform4)
    tm.add_transformation(cs3, cs5, transform5)

    sequences = tm.get_all_shortest_transformation_sequences(
        cs1, cs5, expected_intermediate_transformations=[transform4]
    )
    assert len(sequences) == 2
    assert [transform1, transform2, transform4] in sequences
    assert [transform2, transform1, transform4] in sequences


def test_get_all_shortest_transformation_sequences_multiple_paths_multiple_edges_failure(five_point_graph):
    """Test getting all shortest transformation sequences, with multiple path and multiple edges between nodes."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4, cs5], [transform1, transform2, transform3, transform4, transform5] = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)
    tm.add_coordinate_system(cs5)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)
    tm.add_transformation(cs1, cs3, transform3)
    tm.add_transformation(cs3, cs5, transform4)
    tm.add_transformation(cs3, cs5, transform5)

    with pytest.raises(TransformationPathAmbiguousError, match="Transformation Path ambiguous"):
        tm.get_all_shortest_transformation_sequences(cs1, cs5)


def test_get_all_transformation_sequences(four_point_graph):
    """Test getting all transformation sequences."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4], [transform1, transform2, transform3, transform4] = four_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs3, cs4, transform3)
    tm.add_transformation(cs1, cs3, transform4)

    sequences = tm.get_all_transformation_sequences(cs1, cs4)
    assert len(sequences) == 2
    assert [transform1, transform2, transform3] in sequences
    assert [transform4, transform3] in sequences


def test_get_all_transformation_sequences_multiple_paths(five_point_graph):
    """Test getting all transformation sequences, with multiple path and multiple edges between nodes."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4, _cs5], [transform1, transform2, transform3, _transform4, _transform5] = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)
    tm.add_transformation(cs1, cs3, transform3)

    sequences = tm.get_all_transformation_sequences(cs1, cs3)
    assert len(sequences) == 3
    assert [transform1, transform2] in sequences
    assert [transform2, transform1] in sequences
    assert [transform3] in sequences


def test_get_all_transformation_sequences_multiple_paths_multiple_edges_success(five_point_graph):
    """Test getting all transformation sequences, with multiple paths and multiple edges between nodes."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4, cs5], [transform1, transform2, transform3, transform4, transform5] = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)
    tm.add_coordinate_system(cs5)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)
    tm.add_transformation(cs1, cs3, transform3)
    tm.add_transformation(cs3, cs5, transform4)
    tm.add_transformation(cs3, cs5, transform5)

    sequences = tm.get_all_transformation_sequences(cs1, cs5, expected_intermediate_transformations=[transform4])
    assert len(sequences) == 3
    assert [transform1, transform2, transform4] in sequences
    assert [transform2, transform1, transform4] in sequences
    assert [transform3, transform4] in sequences


def test_get_all_transformation_sequences_multiple_paths_multiple_edges_failure(five_point_graph):
    """Test getting all transformation sequences, with multiple paths and multiple edges between nodes."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4, cs5], [transform1, transform2, transform3, transform4, transform5] = five_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)
    tm.add_coordinate_system(cs5)

    tm.add_transformation(cs1, cs2, transform1)
    tm.add_transformation(cs2, cs3, transform2)
    tm.add_transformation(cs1, cs4, transform2)
    tm.add_transformation(cs4, cs3, transform1)
    tm.add_transformation(cs1, cs3, transform3)
    tm.add_transformation(cs3, cs5, transform4)
    tm.add_transformation(cs3, cs5, transform5)

    with pytest.raises(TransformationPathAmbiguousError, match="Transformation Path ambiguous"):
        tm.get_all_transformation_sequences(cs1, cs5)


def test_get_all_transformation_sequences_no_path(four_point_graph):
    """Test getting all transformation sequences."""
    tm = TransformationManager()
    [cs1, cs2, cs3, cs4], [transform1, transform2, transform3, transform4] = four_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)
    tm.add_coordinate_system(cs3)
    tm.add_coordinate_system(cs4)

    expected_error_msg = "No transformation path found from"
    with pytest.raises(TransformationPathNotFoundError, match=expected_error_msg):
        tm.get_all_shortest_transformation_sequences(cs1, cs4)


def test_get_element_coordinate_system_success(one_point_graph):
    """Test successfully getting an element's coordinate system."""
    tm = TransformationManager()
    [cs1], _ = one_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_element("image1", cs1)

    # This should successfully return the coordinate system
    result = tm.get_element_coordinate_system("image1")
    assert result == cs1


def test_repr_method_comprehensive(fully_connected_two_point_graph):
    """Test TransformationManager.__repr__ method comprehensively."""

    # Test empty TransformationManager
    tm_empty = TransformationManager()
    repr_empty = repr(tm_empty)
    assert "TransformationManager" in repr_empty
    assert "coordinate_systems=[]" in repr_empty
    assert "coordinate_transforms=[]" in repr_empty
    assert "elements=[]" in repr_empty

    # Test TransformationManager with data
    tm = TransformationManager()
    [cs1, cs2], [transform] = fully_connected_two_point_graph
    tm.add_coordinate_system(cs1)
    tm.add_coordinate_system(cs2)

    tm.add_transformation(cs1, cs2, transform)
    tm.add_element("image1", cs1)
    tm.add_element("image2", cs2)

    repr_str = repr(tm)
    assert "TransformationManager" in repr_str
    assert "cs1" in repr_str
    assert "cs2" in repr_str
    assert repr_str.find(repr(transform))
    assert "image1" in repr_str
    assert "image2" in repr_str
