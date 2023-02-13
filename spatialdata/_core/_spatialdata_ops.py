from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING
from anndata import AnnData
from spatialdata._core.models import TableModel

import networkx as nx
import numpy as np

if TYPE_CHECKING:
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
    "concatenate_tables",
    "concatenate",
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


##
def concatenate_tables(tables: list[AnnData]) -> Optional[AnnData]:
    """
    Concatenate a list of tables using AnnData.concatenate() and preserving the validity of region, region_key and instance_key

    Parameters
    ----------
    tables
        A list of tables to concatenate

    Returns
    -------
    A table with all the tables concatenated. If the list of tables is empty, None is returned.

    Notes
    -----
    Not all tables can be merged, they need to have compatible region, region_key and instance_key values. This function
    checks this and merges if possible.

    """
    if len(tables) == 0:
        return None
    if len(tables) == 1:
        return tables[0]

    MERGED_TABLES_REGION_KEY = "annotated_element_merged"
    MAX_CONCATENTAION_TABLES = 100
    for i in range(MAX_CONCATENTAION_TABLES):
        if i == 0:
            key = MERGED_TABLES_REGION_KEY
        else:
            key = f"{MERGED_TABLES_REGION_KEY}_{i}"

        all_without = True
        for table in tables:
            if key in table.obs:
                all_without = False
                break
        if all_without:
            MERGED_TABLES_REGION_KEY = key
            break

    spatialdata_attrs_found = [TableModel.ATTRS_KEY in table.uns for table in tables]
    assert all(spatialdata_attrs_found) or not any(spatialdata_attrs_found)
    if not any(spatialdata_attrs_found):
        merged_region = None
        merged_region_key = None
        merged_instance_key = None
    else:
        all_instance_keys = [table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] for table in tables]
        assert all(all_instance_keys[0] == instance_key for instance_key in all_instance_keys)
        merged_instance_key = all_instance_keys[0]

        all_region_keys = []
        for table in tables:
            TableModel().validate(table)
            region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
            if not (
                isinstance(region, str)
                and region_key is None
                or isinstance(region, list)
                and isinstance(region_key, str)
            ):
                # this code should be never reached because the validate function should have raised an exception, but
                # let's be extra safe
                raise RuntimeError("Tables have incompatible region keys")
            if isinstance(region, list):
                all_region_keys.append(region_key)
            else:
                assert (
                    len(all_region_keys) == 0
                    or len(set(all_region_keys)) == 1
                    and all_region_keys[0] == MERGED_TABLES_REGION_KEY
                )
                table.obs[MERGED_TABLES_REGION_KEY] = region
                all_region_keys.append(MERGED_TABLES_REGION_KEY)
        if len(set(all_region_keys)) != 1:
            raise RuntimeError("Tables have incompatible region keys")
        merged_region_key = all_region_keys[0]

        all_regions = []
        for table in tables:
            region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            if isinstance(region, str):
                all_regions.append(region)
            else:
                all_regions.extend(region)
        all_regions = list(set(all_regions))
        merged_region = all_regions

    attr = {"region": merged_region, "region_key": merged_region_key, "instance_key": merged_instance_key}
    merged_table = AnnData.concatenate(*tables, join="outer", uns_merge="same")

    # remove the MERGED_TABLES_REGION_KEY column if it has been added (the code above either adds that column
    # to all the tables, either it doesn't add it at all)
    for table in tables:
        if MERGED_TABLES_REGION_KEY in table.obs:
            del table.obs[MERGED_TABLES_REGION_KEY]

    merged_table.uns[TableModel.ATTRS_KEY] = attr
    merged_table.obs[merged_region_key] = merged_table.obs[merged_region_key].astype("category")
    TableModel().validate(merged_table)
    return merged_table


##


def concatenate(sdatas: list[SpatialData], omit_table: bool = False) -> SpatialData:
    """Concatenate a list of spatial data objects.

    Parameters
    ----------
    sdatas
        The spatial data objects to concatenate.

    Returns
    -------
    SpatialData
        The concatenated spatial data object.
    """
    from spatialdata._core._spatialdata import SpatialData

    assert type(sdatas) == list
    assert len(sdatas) > 0
    if len(sdatas) == 1:
        return sdatas[0]

    if not omit_table:
        list_of_tables = [sdata.table for sdata in sdatas if sdata.table is not None]
        merged_table = concatenate_tables(list_of_tables)
    else:
        merged_table = None

    merged_images = {**{k: v for sdata in sdatas for k, v in sdata.images.items()}}
    if len(merged_images) != np.sum([len(sdata.images) for sdata in sdatas]):
        raise RuntimeError("Images must have unique names across the SpatialData objects to concatenate")
    merged_labels = {**{k: v for sdata in sdatas for k, v in sdata.labels.items()}}
    if len(merged_labels) != np.sum([len(sdata.labels) for sdata in sdatas]):
        raise RuntimeError("Labels must have unique names across the SpatialData objects to concatenate")
    merged_points = {**{k: v for sdata in sdatas for k, v in sdata.points.items()}}
    if len(merged_points) != np.sum([len(sdata.points) for sdata in sdatas]):
        raise RuntimeError("Points must have unique names across the SpatialData objects to concatenate")
    merged_polygons = {**{k: v for sdata in sdatas for k, v in sdata.polygons.items()}}
    if len(merged_polygons) != np.sum([len(sdata.polygons) for sdata in sdatas]):
        raise RuntimeError("Polygons must have unique names across the SpatialData objects to concatenate")
    merged_shapes = {**{k: v for sdata in sdatas for k, v in sdata.shapes.items()}}
    if len(merged_shapes) != np.sum([len(sdata.shapes) for sdata in sdatas]):
        raise RuntimeError("Shapes must have unique names across the SpatialData objects to concatenate")

    sdata = SpatialData(
        images=merged_images,
        labels=merged_labels,
        points=merged_points,
        polygons=merged_polygons,
        shapes=merged_shapes,
        table=merged_table,
    )
    return sdata
