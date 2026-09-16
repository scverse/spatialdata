from __future__ import annotations

from spatialdata.models import SpatialElement, get_axes_names
from spatialdata.transformations._graph.vert import Axis, CoordSystem


def get_default_coordinate_system(
    element_to_be_added_to_graph: SpatialElement, coordinate_system_name: str
) -> CoordSystem:
    """
    Create a default coordinate system object for a spatial element, which can be added to a graph.

    Parameters
    ----------
    element_to_be_added_to_graph
        The spatial element for which to create a coordinate system.
    coordinate_system_name
        The name to use for the coordinate system.

    Returns
    -------
    A CoordSystem object representing the default coordinate system for the element.
    """
    ngff_axes = []
    for axis_name in get_axes_names(element_to_be_added_to_graph):
        ngff_axis = Axis.from_spatialdata_axis_name(axis_name)
        ngff_axes.append(ngff_axis)
    return CoordSystem(name=coordinate_system_name, axes=tuple(ngff_axes), virtual=False)
