#!/usr/bin/env python3

"""
Fixtures for transformation manager tests.
"""

from __future__ import annotations

import pytest

from spatialdata.transformations import Affine, Scale, Translation
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffAxis, NgffCoordinateSystem


def get_ngff_coodinate_system(cs_name: str) -> NgffCoordinateSystem:
    return NgffCoordinateSystem(
        name=cs_name,
        axes=[NgffAxis(name="x", type="space", unit="micrometer"), NgffAxis(name="y", type="space", unit="micrometer")],
    )


@pytest.fixture
def one_point_graph() -> tuple[list[NgffCoordinateSystem], list[Translation]]:
    """Fixture providing a single point graph with one coordinate system and one transformation."""
    coordinate_systems = [get_ngff_coodinate_system("cs1")]
    transformations = [Translation(translation=[1.0, 2.0], axes=("x", "y"))]
    return coordinate_systems, transformations


@pytest.fixture
def fully_connected_two_point_graph() -> tuple[list[NgffCoordinateSystem], list[Translation]]:
    """Fixture providing a fully connected two-point graph with two coordinate systems and transformations."""
    coordinate_systems = [get_ngff_coodinate_system("cs1"), get_ngff_coodinate_system("cs2")]
    transformations = [
        Translation(translation=[-1.0, -2.0], axes=("x", "y")),
    ]
    return coordinate_systems, transformations


@pytest.fixture
def four_point_graph() -> tuple[list[NgffCoordinateSystem], list[Scale | Translation]]:
    """Fixture providing a four-point graph with four coordinate systems and transformations."""
    coordinate_systems = [
        get_ngff_coodinate_system("cs1"),
        get_ngff_coodinate_system("cs2"),
        get_ngff_coodinate_system("cs3"),
        get_ngff_coodinate_system("cs4"),
    ]
    transformations = [
        Translation(translation=[1.0, 2.0], axes=("x", "y")),  # cs1 -> cs2
        Translation(translation=[3.0, 4.0], axes=("x", "y")),  # cs2 -> cs3
        Scale(
            scale=[
                2,
                2,
            ],
            axes=("x", "y"),
        ),  # cs3 -> cs4;
        Translation(translation=[4.0, 6.0], axes=("x", "y")),  # cs1 -> cs3 (consistent with cs1->cs2 and cs2->cs3)
    ]
    return coordinate_systems, transformations


@pytest.fixture
def five_point_graph() -> tuple[list[NgffCoordinateSystem], list[Scale | Translation | Affine]]:
    """Fixture providing a five-point graph with five coordinate systems and five transformations."""

    coordinate_systems = [
        get_ngff_coodinate_system("cs1"),
        get_ngff_coodinate_system("cs2"),
        get_ngff_coodinate_system("cs3"),
        get_ngff_coodinate_system("cs4"),
        get_ngff_coodinate_system("cs5"),
    ]
    transformations = [
        Translation(translation=[1.0, 2.0], axes=("x", "y")),  # cs1 -> cs2, cs4 -> cs3
        Translation(translation=[3.0, 4.0], axes=("x", "y")),  # cs2 -> cs3, cs1 -> cs4
        Translation(translation=[4.0, 6.0], axes=("x", "y")),  # cs1 -> cs3 (consistent with cs1->cs2 and cs2->cs3)
        Scale(
            scale=[
                2,
                2,
            ],
            axes=("x", "y"),
        ),  # cs3 -> cs5;
        Affine(
            matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),  # cs3 -> cs5
    ]
    return coordinate_systems, transformations
