from __future__ import annotations

from spatialdata.models import SpatialElement, get_axes_names
from spatialdata.models._utils import get_axes_types
from spatialdata.transformations.graph.vert import Axis, CoordSystem


def get_default_coordinate_system(
    element_to_be_added_to_graph: SpatialElement, coordinate_system_name: str
) -> CoordSystem:

    axes_names = get_axes_names(element_to_be_added_to_graph)
    axes_types = get_axes_types(element_to_be_added_to_graph)
    ngff_axes = []
    for axis_name, axis_type in zip(axes_names, axes_types, strict=True):
        ngff_axis = Axis(name=axis_name, type=axis_type)
        ngff_axes.append(ngff_axis)
    return CoordSystem(name=coordinate_system_name, axes=tuple(ngff_axes), virtual=False)
