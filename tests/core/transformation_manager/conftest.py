#!/usr/bin/env python3

"""
Fixtures for transformation manager tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialdata.transformations.graph.edge import AffineEdge, ScaleEdge, TranslationEdge
from spatialdata.transformations.graph.vert import Axis, CoordSystem


def get_coord_system(cs_name: str) -> CoordSystem:
    return CoordSystem(
        name=cs_name,
        axes=[Axis(name="x", type="space", unit="micrometer"), Axis(name="y", type="space", unit="micrometer")],
    )


@pytest.fixture
def one_point_graph() -> list[CoordSystem]:
    """Fixture providing a single point graph with one coordinate system."""
    return [get_coord_system("cs1")]


@pytest.fixture
def fully_connected_two_point_graph() -> tuple[list[CoordSystem], list[TranslationEdge]]:
    """Fixture providing a fully connected two-point graph with two coordinate systems and transformations."""
    coordinate_systems = [get_coord_system("cs1"), get_coord_system("cs2")]
    cs1, cs2 = coordinate_systems
    transformations = [
        TranslationEdge(translation=np.array([-1.0, -2.0]), input=cs1, output=cs2),
    ]
    return coordinate_systems, transformations


@pytest.fixture
def four_point_graph() -> tuple[list[CoordSystem], list[ScaleEdge | TranslationEdge]]:
    """Fixture providing a four-point graph with four coordinate systems and transformations."""
    coordinate_systems = [
        get_coord_system("cs1"),
        get_coord_system("cs2"),
        get_coord_system("cs3"),
        get_coord_system("cs4"),
    ]
    cs1, cs2, cs3, cs4 = coordinate_systems
    transformations = [
        TranslationEdge(translation=np.array([1.0, 2.0]), input=cs1, output=cs2),  # cs1 -> cs2
        TranslationEdge(translation=np.array([3.0, 4.0]), input=cs2, output=cs3),  # cs2 -> cs3
        ScaleEdge(scale=np.array([2, 2]), input=cs3, output=cs4),  # cs3 -> cs4
        TranslationEdge(translation=np.array([4.0, 6.0]), input=cs1, output=cs3),
        # cs1 -> cs3 (consistent with cs1->cs2 and cs2->cs3)
    ]
    return coordinate_systems, transformations


@pytest.fixture
def five_point_graph() -> tuple[list[CoordSystem], list[ScaleEdge | TranslationEdge | AffineEdge]]:
    """Fixture providing a five-point graph with five coordinate systems and five transformations."""

    coordinate_systems = [
        get_coord_system("cs1"),
        get_coord_system("cs2"),
        get_coord_system("cs3"),
        get_coord_system("cs4"),
        get_coord_system("cs5"),
    ]
    cs1, cs2, cs3, cs4, cs5 = coordinate_systems
    transformations = [
        TranslationEdge(translation=np.array([1.0, 2.0]), input=cs1, output=cs2),  # cs1 -> cs2,
        TranslationEdge(translation=np.array([1.0, 2.0]), input=cs4, output=cs3),  # cs4 -> cs3
        TranslationEdge(translation=np.array([3.0, 4.0]), input=cs2, output=cs3),  # cs2 -> cs3
        TranslationEdge(translation=np.array([3.0, 4.0]), input=cs1, output=cs4),  # cs1 -> cs4
        TranslationEdge(translation=np.array([4.0, 6.0]), input=cs1, output=cs3),
        # cs1 -> cs3 (consistent with cs1->cs2 and cs2->cs3)
        ScaleEdge(scale=np.array([2, 2]), input=cs3, output=cs5),  # cs3 -> cs5;
        AffineEdge(
            linear=np.identity(2),
            translation=np.array([0, 0]),
            input=cs3,
            output=cs5,
        ),  # cs3 -> cs5
    ]
    return coordinate_systems, transformations
