from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import get_coordinate_dimension
from xarray import DataArray, DataTree

from spatialdata.transformations._graph import C, X, Y, Z
from spatialdata.transformations._graph.types import ValidAxis_t
from spatialdata.transformations._graph.vert import Axis, CoordSystem

if TYPE_CHECKING:
    from spatialdata.models import SpatialElement


def get_intrinsic_coordinate_system_name(element_key: str) -> str:

    return f"{element_key}_intrinsic"


def get_instrinsic_coordinate_system(
    element_to_be_added_to_graph: SpatialElement, coordinate_system_name: str
) -> CoordSystem:
    """
    Create a coordinate system object from a spatial element, which can be added to a graph.

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


def validate_axis_name(axis: ValidAxis_t) -> None:
    """
    Check if the axis name is valid.

    Parameters
    ----------
    axis
        The axis name

    Raises
    ------
    TypeError
        If the axis name not in ["c", "x", "y", "z"]
    """
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
    """
    Check if the names of the axes are valid.

    Parameters
    ----------
    axis
        The names of the axes

    Raises
    ------
    TypeError
        If not in ["c", "x", "y", "z"]
    """
    for ax in axes:
        validate_axis_name(ax)
    if len(axes) != len(set(axes)):
        raise ValueError("Axes must be unique.")


def get_spatial_axes(axes: tuple[ValidAxis_t, ...]) -> tuple[ValidAxis_t, ...]:
    """
    Get the spatial axes of interest.

    Parameters
    ----------
    axes
        Should be a subset of ['x', 'y', 'z', 'c']

    Returns
    -------
    The spatial axes, i.e. the input axes but without 'c'
    """
    validate_axes(axes)
    return tuple(ax for ax in axes if ax in [X, Y, Z])


@singledispatch
def get_axes_names(e: SpatialElement | pd.DataFrame) -> tuple[str, ...]:
    """
    Get the dimensions of a SpatialElement.

    Parameters
    ----------
    e
        SpatialElement

    Returns
    -------
    Dimensions of the SpatialElement (e.g. ("z", "y", "x"))
    """
    raise TypeError(f"Unsupported type: {type(e)}")


@get_axes_names.register(DataTree)
def _(e: DataTree) -> tuple[str, ...]:
    if "scale0" in e:
        # dims_coordinates = tuple(i for i in e["scale0"].dims.keys())

        scale0 = e["scale0"]
        if not isinstance(scale0, DataTree):
            raise TypeError(f"Expected `scale0` to be a DataTree node, got {type(scale0).__name__}.")
        variables = list(scale0.values())
        assert len(variables) == 1
        xdata = variables[0]
        dims_data = tuple(str(dim) for dim in xdata.dims)

        _validate_dims(dims_data)
        return dims_data
    raise ValueError("Spatialdata DataTree does not contain the scale0 key")


@get_axes_names.register(GeoDataFrame)
def _(e: GeoDataFrame) -> tuple[str, ...]:
    all_dims = (X, Y, Z)
    n = get_coordinate_dimension(e.geometry.iloc[0])
    dims = all_dims[:n]
    if Z not in dims and Z in e.columns:
        dims += (Z,)
    _validate_dims(dims)
    return dims


@get_axes_names.register(DaskDataFrame)
@get_axes_names.register(pd.DataFrame)
def _(e: DaskDataFrame | pd.DataFrame) -> tuple[str, ...]:
    valid_dims = (X, Y, Z)
    dims = tuple([c for c in valid_dims if c in e.columns])
    _validate_dims(dims)
    return dims


@get_axes_names.register(DataArray)
def _(e: DataArray) -> tuple[str, ...]:
    dims = tuple(str(dim) for dim in e.dims)
    _validate_dims(dims)
    return dims


def _validate_dims(dims: tuple[str, ...]) -> None:
    for c in dims:
        if c not in (X, Y, Z, C):
            raise ValueError(f"Invalid dimension: {c}")
    if dims not in [(X,), (Y,), (Z,), (C,), (X, Y), (X, Y, Z), (Y, X), (Z, Y, X), (C, Y, X), (C, Z, Y, X)]:
        raise ValueError(f"Invalid dimensions: {dims}")


def get_union_of_axes(axes: tuple[Axis, ...]) -> tuple[Axis, ...]:
    """
    Produce a union of ``Axis`` objects, dropping duplicates and ordering reverse alphabetically.

    The ordering favors non‑None values for ``unit`` and ``long_name``.
    Axes that share the same ``name`` **and** ``type`` are considered duplicates;
    only the first occurrence (according to the sorted order) is retained.
    """
    # ------------------------------------------------------------------
    # Build a DataFrame that holds the original Axis objects together with
    # their attributes and auxiliary “non‑null” flags.
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "axis": pd.Series(axes, dtype="object"),  # original Axis objects
            "name": pd.Series([ax.name for ax in axes], dtype="string"),
            "type": pd.Series([ax.type for ax in axes], dtype="object"),
            "unit": pd.Series([ax.unit for ax in axes], dtype="string"),
            "long_name": pd.Series([ax.long_name for ax in axes], dtype="string"),
        }
    )

    # Boolean columns that indicate the presence of a value.
    df["unit_not_none"] = df["unit"].notna()
    df["long_name_not_none"] = df["long_name"].notna()

    # Sort axes in descending (alphabetical) order
    df = df.sort_values(
        by=[
            "type",
            "name",
            "unit_not_none",
            "unit",
            "long_name_not_none",
            "long_name",
        ],
        ascending=False,
        kind="mergesort",  # stable sort to preserve order of equal keys
    )

    # Remove duplicates based on the logical key (name, type, unit, long_name).
    df = df.drop_duplicates(subset=["name", "type", "unit", "long_name"], keep="first")

    return tuple(df["axis"])
