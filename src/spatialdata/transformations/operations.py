from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional, Union

import networkx as nx
import numpy
import numpy as np
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from skimage.transform import estimate_transform

from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData
    from spatialdata.models import SpatialElement
    from spatialdata.transformations import Affine, BaseTransformation


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
        The coordinate system to set the transformation/s to.

        * If None and `set_all=False` sets the transformation to the 'global' coordinate system (default system).
        * If None and `set_all=True` sets all transformations.

    set_all
        If True, all transformations are set. If False, only the transformation
        to the specified coordinate system is set. If True, `to_coordinate_system` needs to be None.
    write_to_sdata
        The SpatialData object to set the transformation/s to.
        If None, the transformation/s are set in-memory. If not None, the element needs to belong
        to the SpatialData object, and the SpatialData object needs to be backed.

    """
    from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
    from spatialdata.transformations import BaseTransformation

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
        if not write_to_sdata.contains_element(element, raise_exception=True):
            raise RuntimeError("contains_element() failed without raising an exception.")
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
        The coordinate system to which the transformation should be returned.

        * If None and `get_all=False` returns the transformation from the 'global' coordinate system (default system).
        * If None and `get_all=True` returns all transformations.

    get_all
        If True, all transformations are returned. If True, `to_coordinate_system` needs to be None.

    Returns
    -------
    The transformation, if `to_coordinate_system` is not None, otherwise a dictionary of transformations to all
    the coordinate systems.
    """
    from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

    transformations = _get_transformations(element)
    assert isinstance(transformations, dict)

    if get_all is False:
        if to_coordinate_system is None:
            to_coordinate_system = DEFAULT_COORDINATE_SYSTEM
        # get a specific transformation
        if to_coordinate_system not in transformations:
            raise ValueError(f"Transformation to {to_coordinate_system} not found in element {element}.")
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

        * If None and `remove_all=False` removes the transformation from the 'global' coordinate system
            (default system).
        * If None and `remove_all=True` removes all transformations.

    remove_all
        If True, all transformations are removed. If True, `to_coordinate_system` needs to be None.
    write_to_sdata
        The SpatialData object to remove the transformation/s from.
        If None, the transformation/s are removed in-memory.
        If not None, the element needs to belong to the SpatialData object,
        and the SpatialData object needs to be backed.
    """
    from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

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
        if not write_to_sdata.contains_element(element, raise_exception=True):
            raise RuntimeError("contains_element() failed without raising an exception.")
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
            with contextlib.suppress(np.linalg.LinAlgError):
                g.add_edge(cs, id(e), transformation=t.inverse())
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
    from spatialdata.models._utils import has_type_spatial_element
    from spatialdata.transformations import Identity, Sequence

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


def get_transformation_between_landmarks(
    references_coords: Union[GeoDataFrame, DaskDataFrame],
    moving_coords: Union[GeoDataFrame, DaskDataFrame],
) -> Affine:
    """
    Get a similarity transformation between two lists of (n >= 3) landmarks.

    Note that landmarks are assumed to be in the same space.

    Parameters
    ----------
    references_coords
        landmarks annotating the reference element. Must be a valid element describing points or circles.
    moving_coords
        landmarks annotating the moving element. Must be a valid element describing points or circles.

    Returns
    -------
    The Affine transformation that maps the moving element to the reference element.

    Examples
    --------
    If you save the landmark points using napari_spatialdata, they will be alredy saved as circles. Here is an
    example on how to call this function on two sets of numpy arrays describing x, y coordinates.
    >>> import numpy as np
    >>> from spatialdata.models import PointsModel
    >>> from spatialdata.transform import get_transformation_between_landmarks
    >>> points_moving = np.array([[0, 0], [1, 1], [2, 2]])
    >>> points_reference = np.array([[0, 0], [10, 10], [20, 20]])
    >>> moving_coords = PointsModel(points_moving)
    >>> references_coords = PointsModel(points_reference)
    >>> transformation = get_transformation_between_landmarks(references_coords, moving_coords)
    """
    from spatialdata import transform
    from spatialdata.models import get_axes_names
    from spatialdata.transformations.transformations import (
        Affine,
        BaseTransformation,
        Sequence,
    )

    assert get_axes_names(references_coords) == ("x", "y")
    assert get_axes_names(moving_coords) == ("x", "y")

    if isinstance(references_coords, GeoDataFrame):
        references_xy = np.stack([references_coords.geometry.x, references_coords.geometry.y], axis=1)
        moving_xy = np.stack([moving_coords.geometry.x, moving_coords.geometry.y], axis=1)
    elif isinstance(references_coords, DaskDataFrame):
        references_xy = references_coords[["x", "y"]].to_dask_array().compute()
        moving_xy = moving_coords[["x", "y"]].to_dask_array().compute()
    else:
        raise TypeError("references_coords must be either an GeoDataFrame or a DaskDataFrame")

    model = estimate_transform("affine", src=moving_xy, dst=references_xy)
    transform_matrix = model.params
    a = transform_matrix[:2, :2]
    d = np.linalg.det(a)
    final: BaseTransformation
    if d < 0:
        m = (moving_xy[:, 0].max() - moving_xy[:, 0].min()) / 2
        flip = Affine(
            np.array(
                [
                    [-1, 0, 2 * m],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        flipped_moving = transform(moving_coords, flip, maintain_positioning=False)
        if isinstance(flipped_moving, GeoDataFrame):
            flipped_moving_xy = np.stack([flipped_moving.geometry.x, flipped_moving.geometry.y], axis=1)
        elif isinstance(flipped_moving, DaskDataFrame):
            flipped_moving_xy = flipped_moving[["x", "y"]].to_dask_array().compute()
        else:
            raise TypeError("flipped_moving must be either an GeoDataFrame or a DaskDataFrame")
        model = estimate_transform("similarity", src=flipped_moving_xy, dst=references_xy)
        final = Sequence([flip, Affine(model.params, input_axes=("x", "y"), output_axes=("x", "y"))])
    else:
        model = estimate_transform("similarity", src=moving_xy, dst=references_xy)
        final = Affine(model.params, input_axes=("x", "y"), output_axes=("x", "y"))

    affine = Affine(
        final.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    return affine


def align_elements_using_landmarks(
    references_coords: Union[GeoDataFrame | DaskDataFrame],
    moving_coords: Union[GeoDataFrame | DaskDataFrame],
    reference_element: SpatialElement,
    moving_element: SpatialElement,
    reference_coordinate_system: str = "global",
    moving_coordinate_system: str = "global",
    new_coordinate_system: Optional[str] = None,
    write_to_sdata: Optional[SpatialData] = None,
) -> BaseTransformation:
    """
    Maps a moving object into a reference object using two lists of (n >= 3) landmarks.

    This returns the transformations that enable this mapping and optionally saves them,
    to map to a new shared coordinate system.

    Parameters
    ----------
    references_coords
        landmarks annotating the reference element. Must be a valid element describing points or circles.
    moving_coords
        landmarks annotating the moving element. Must be a valid element describing points or circles.
    reference_element
        the reference element.
    moving_element
        the moving element.
    reference_coordinate_system
        the coordinate system of the reference element that have been used to annotate the landmarks.
    moving_coordinate_system
        the coordinate system of the moving element that have been used to annotate the landmarks.
    new_coordinate_system
        If provided, both elements will be mapped to this new coordinate system with the new transformations just
        computed.
    write_to_sdata
        If provided, the transformations will be saved to disk in the specified SpatialData object. The SpatialData
        object must be backed and must contain both the reference and moving elements.

    Returns
    -------
    A similarity transformation that maps the moving element to the same coordinate of reference element in the
    coordinate system specified by reference_coordinate_system.
    """
    from spatialdata.transformations.transformations import BaseTransformation, Sequence

    affine = get_transformation_between_landmarks(references_coords, moving_coords)

    # get the old transformations of the visium and xenium data
    old_moving_transformation = get_transformation(moving_element, moving_coordinate_system)
    old_reference_transformation = get_transformation(reference_element, reference_coordinate_system)
    assert isinstance(old_moving_transformation, BaseTransformation)
    assert isinstance(old_reference_transformation, BaseTransformation)

    # compute the new transformations
    new_moving_transformation = Sequence([old_moving_transformation, affine])
    new_reference_transformation = old_reference_transformation

    if new_coordinate_system is not None:
        # this allows to work on singleton objects, not embedded in a SpatialData object
        set_transformation(
            moving_element, new_moving_transformation, new_coordinate_system, write_to_sdata=write_to_sdata
        )
        set_transformation(
            reference_element, new_reference_transformation, new_coordinate_system, write_to_sdata=write_to_sdata
        )
    return new_moving_transformation
