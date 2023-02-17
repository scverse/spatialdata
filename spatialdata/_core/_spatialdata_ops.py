from __future__ import annotations

from typing import Optional, Union

import networkx as nx
import numpy as np

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.core_utils import (
    DEFAULT_COORDINATE_SYSTEM,
    SpatialElement,
    _get_transformations,
    _set_transformations,
    has_type_spatial_element,
)
from spatialdata._core.transformations import BaseTransformation, Identity, Sequence

__all__ = [
    "set_transformation",
    "get_transformation",
    "remove_transformation",
    "get_transformation_between_coordinate_systems",
]


def set_transformation(
    element: SpatialElement,
    transformation: Union[BaseTransformation, dict[str, BaseTransformation]],
    to_coordinate_system: Optional[str] = None,
    set_all: bool = False,
    write_to_sdata: Optional[SpatialData] = None,
) -> None:
    """
    Set a transformation/s to an element, in-memory or to disk.

    Parameters
    ----------
    element
        The element to set the transformation/s to.
    transformation
        The transformation/s to set.
    to_coordinate_system
        The coordinate system to set the transformation/s to. This needs to be none if multiple transformations are
        being set.
    set_all
        If True, all transformations are set. If False, only the transformation to the specified coordinate system is set.
        If True, `to_coordinate_system` needs to be None.
    write_to_sdata
        The SpatialData object to set the transformation/s to. If None, the transformation/s are set in-memory. If not
        None, the element needs to belong to the SpatialData object, and the SpatialData object needs to be backed.

    """
    if write_to_sdata is None:
        if set_all is False:
            assert isinstance(transformation, BaseTransformation)
            transformations = _get_transformations(element)
            assert transformations is not None
            if to_coordinate_system is None:
                to_coordinate_system = DEFAULT_COORDINATE_SYSTEM
            transformations[to_coordinate_system] = transformation
            _set_transformations(element, transformations)
        else:
            assert isinstance(transformation, dict)
            assert to_coordinate_system is None
            _set_transformations(element, transformation)
    else:
        if not write_to_sdata.contains_element(element):
            raise ValueError("The element is not part of the SpatialData object.")
        if not write_to_sdata.is_backed():
            raise ValueError(
                "The SpatialData object is not backed. You can either set a transformation to an element "
                "in-memory (write_to_sdata=None), or in-memory and to disk; this last case requires the element "
                "to belong to the SpatialData object that is backed."
            )
        set_transformation(element, transformation, to_coordinate_system, set_all, None)
        write_to_sdata._write_transformations_to_disk(element)


def get_transformation(
    element: SpatialElement, to_coordinate_system: Optional[str] = None, get_all: bool = False
) -> Union[BaseTransformation, dict[str, BaseTransformation]]:
    """
    Get the transformation/s of an element.

    Parameters
    ----------
    element
        The element.
    to_coordinate_system
        The coordinate system to which the transformation should be returned. If None, all transformations are returned.
    get_all
        If True, all transformations are returned. If True, `to_coordinate_system` needs to be None.

    Returns
    -------
    transformation
        The transformation, if `to_coordinate_system` is not None, otherwise a dictionary of transformations to all
        the coordinate systems.
    """
    transformations = _get_transformations(element)
    assert isinstance(transformations, dict)

    if get_all is False:
        if to_coordinate_system is None:
            to_coordinate_system = DEFAULT_COORDINATE_SYSTEM
        # get a specific transformation
        if to_coordinate_system not in transformations:
            raise ValueError(f"Transformation to {to_coordinate_system} not found")
        return transformations[to_coordinate_system]
    else:
        assert to_coordinate_system is None
        # get the dict of all the transformations
        return transformations


def remove_transformation(
    element: SpatialElement,
    to_coordinate_system: Optional[str] = None,
    remove_all: bool = False,
    write_to_sdata: Optional[SpatialData] = None,
) -> None:
    """
    Remove a transformation/s from an element, in-memory or from disk.

    Parameters
    ----------
    element
        The element to remove the transformation/s from.
    to_coordinate_system
        The coordinate system to remove the transformation/s from. If None, all transformations are removed.
    remove_all
        If True, all transformations are removed. If True, `to_coordinate_system` needs to be None.
    write_to_sdata
        The SpatialData object to remove the transformation/s from. If None, the transformation/s are removed in-memory.
        If not None, the element needs to belong to the SpatialData object, and the SpatialData object needs to be backed.
    """
    if write_to_sdata is None:
        if remove_all is False:
            transformations = _get_transformations(element)
            assert transformations is not None
            if to_coordinate_system is None:
                to_coordinate_system = DEFAULT_COORDINATE_SYSTEM
            del transformations[to_coordinate_system]
            _set_transformations(element, transformations)
        else:
            assert to_coordinate_system is None
            _set_transformations(element, {})
    else:
        if not write_to_sdata.contains_element(element):
            raise ValueError("The element is not part of the SpatialData object.")
        if not write_to_sdata.is_backed():
            raise ValueError(
                "The SpatialData object is not backed. You can either remove a transformation from an "
                "element in-memory (write_to_sdata=None), or in-memory and from disk; this last case requires the "
                "element to belong to the SpatialData object that is backed."
            )
        remove_transformation(element, to_coordinate_system, remove_all, None)
        write_to_sdata._write_transformations_to_disk(element)


def _build_transformations_graph(sdata: SpatialData) -> nx.Graph:
    g = nx.DiGraph()
    gen = sdata._gen_elements_values()
    for cs in sdata.coordinate_systems:
        g.add_node(cs)
    for e in gen:
        g.add_node(id(e))
        transformations = get_transformation(e, get_all=True)
        assert isinstance(transformations, dict)
        for cs, t in transformations.items():
            g.add_edge(id(e), cs, transformation=t)
            try:
                g.add_edge(cs, id(e), transformation=t.inverse())
            except np.linalg.LinAlgError:
                pass
    return g


def get_transformation_between_coordinate_systems(
    sdata: SpatialData,
    source_coordinate_system: Union[SpatialElement, str],
    target_coordinate_system: Union[SpatialElement, str],
    intermediate_coordinate_systems: Optional[Union[SpatialElement, str]] = None,
) -> BaseTransformation:
    """
    Get the transformation to map a coordinate system (intrinsic or extrinsic) to another one.

    Parameters
    ----------
    source_coordinate_system
        The source coordinate system. Can be a SpatialElement (intrinsic coordinate system) or a string (extrinsic
        coordinate system).
    target_coordinate_system
        The target coordinate system. Can be a SpatialElement (intrinsic coordinate system) or a string (extrinsic
        coordinate system).

    Returns
    -------
    The transformation to map the source coordinate system to the target coordinate system.
    """

    def _describe_paths(paths: list[list[Union[int, str]]]) -> str:
        paths_str = ""
        for p in paths:
            components = []
            for c in p:
                if isinstance(c, str):
                    components.append(f"{c!r}")
                else:
                    ss = [
                        f"<sdata>.{element_type}[{element_name!r}]"
                        for element_type, element_name, e in sdata._gen_elements()
                        if id(e) == c
                    ]
                    assert len(ss) == 1
                    components.append(ss[0])
            paths_str += "\n    " + " -> ".join(components)
        return paths_str

    if (
        isinstance(source_coordinate_system, str)
        and isinstance(target_coordinate_system, str)
        and source_coordinate_system == target_coordinate_system
        or id(source_coordinate_system) == id(target_coordinate_system)
    ):
        return Identity()
    else:
        g = _build_transformations_graph(sdata)
        src_node: Union[int, str]
        if has_type_spatial_element(source_coordinate_system):
            src_node = id(source_coordinate_system)
        else:
            assert isinstance(source_coordinate_system, str)
            src_node = source_coordinate_system
        tgt_node: Union[int, str]
        if has_type_spatial_element(target_coordinate_system):
            tgt_node = id(target_coordinate_system)
        else:
            assert isinstance(target_coordinate_system, str)
            tgt_node = target_coordinate_system
        paths = list(nx.all_simple_paths(g, source=src_node, target=tgt_node))
        if len(paths) == 0:
            # error 0 (we refer to this in the tests)
            raise RuntimeError("No path found between the two coordinate systems")
        elif len(paths) > 1:
            if intermediate_coordinate_systems is None:
                # if one and only one of the paths has lenght 1, we choose it straight away, otherwise we raise
                # an expection and ask the user to be more specific
                paths_with_length_1 = [p for p in paths if len(p) == 2]
                if len(paths_with_length_1) == 1:
                    path = paths_with_length_1[0]
                else:
                    # error 1
                    s = _describe_paths(paths)
                    raise RuntimeError(
                        "Multiple paths found between the two coordinate systems. Please specify an intermediate "
                        f"coordinate system. Available paths are:{s}"
                    )
            else:
                if has_type_spatial_element(intermediate_coordinate_systems):
                    intermediate_coordinate_systems = id(intermediate_coordinate_systems)
                paths = [p for p in paths if intermediate_coordinate_systems in p]
                if len(paths) == 0:
                    # error 2
                    raise RuntimeError(
                        "No path found between the two coordinate systems passing through the intermediate"
                    )
                elif len(paths) > 1:
                    # error 3
                    s = _describe_paths(paths)
                    raise RuntimeError(
                        "Multiple paths found between the two coordinate systems passing through the intermediate. "
                        f"Avaliable paths are:{s}"
                    )
                else:
                    path = paths[0]
        else:
            path = paths[0]
        transformations = []
        for i in range(len(path) - 1):
            transformations.append(g[path[i]][path[i + 1]]["transformation"])
        sequence = Sequence(transformations)
        return sequence
