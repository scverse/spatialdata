from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

__all__ = ["NgffCoordinateSystem", "NgffAxis"]

Axis_t = dict[str, str]
CoordSystem_t = dict[str, Union[str, list[dict[str, str]]]]
AXIS_ORDER = ["t", "c", "z", "y", "x"]


class NgffAxis:
    """
    Representation of an axis, following the NGFF specification.

    Attributes
    ----------
    name
        name of the axis.
    type
        type of the axis. Should be in ["type", "channel", "space"].
    unit
        unit of the axis. For a set of valid options see https://ngff.openmicroscopy.org/
    """

    name: str
    type: str
    unit: str | None

    def __init__(self, name: str, type: str, unit: str | None = None):
        self.name = name
        self.type = type
        self.unit = unit

    def __repr__(self) -> str:
        inner = ", ".join(f"{v!r}" for v in self.to_dict().values())
        return f"NgffAxis({inner})"

    def to_dict(self) -> Axis_t:
        d = {"name": self.name, "type": self.type}
        if self.unit is not None:
            d["unit"] = self.unit
        return d

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NgffAxis):
            return False
        return self.to_dict() == other.to_dict()


class NgffCoordinateSystem:
    """
    Representation of a coordinate system, following the NGFF specification.

    Parameters
    ----------
    name
        name of the coordinate system
    axes
        names of the axes of the coordinate system
    """

    def __init__(self, name: str, axes: Optional[list[NgffAxis]] = None):
        self.name = name
        self._axes = axes if axes is not None else []
        if len(self._axes) != len({axis.name for axis in self._axes}):
            raise ValueError("Axes names must be unique")

    def __repr__(self) -> str:
        return f"NgffCoordinateSystem({self.name!r}, {self._axes})"

    @staticmethod
    def from_dict(coord_sys: CoordSystem_t) -> NgffCoordinateSystem:
        if "name" not in coord_sys:
            raise ValueError("`coordinate_system` MUST have a name.")
        if "axes" not in coord_sys:
            raise ValueError("`coordinate_system` MUST have axes.")

        if TYPE_CHECKING:
            assert isinstance(coord_sys["name"], str)
            assert isinstance(coord_sys["axes"], list)
        name = coord_sys["name"]

        # sorted_axes = sorted(coord_sys["axes"], key=lambda x: AXIS_ORDER.index(x["name"]))
        axes = []
        for axis in coord_sys["axes"]:
            # for axis in sorted_axes:
            if "name" not in axis:
                raise ValueError("Each axis MUST have a name.")
            if "type" not in axis:
                raise ValueError("Each axis MUST have a type.")
            if "unit" not in axis and axis["type"] not in ["channel", "array"]:
                raise ValueError("Each axis is either of type channel either MUST have a unit.")
            kw = {}
            if "unit" in axis:
                kw = {"unit": axis["unit"]}
            axes.append(NgffAxis(name=axis["name"], type=axis["type"], **kw))
        return NgffCoordinateSystem(name=name, axes=axes)

    def to_dict(self) -> CoordSystem_t:
        out: dict[str, Any] = {"name": self.name, "axes": [axis.to_dict() for axis in self._axes]}
        # if TYPE_CHECKING:
        #     assert isinstance(out["axes"], list)
        return out

    @staticmethod
    def from_json(data: Union[str, bytes]) -> NgffCoordinateSystem:
        """Initialize a coordinate system from it's json representation."""
        coord_sys = json.loads(data)
        return NgffCoordinateSystem.from_dict(coord_sys)

    def to_json(self, **kwargs: Any) -> str:
        """Give the json representation of the coordinate system."""
        out = self.to_dict()
        return json.dumps(out, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NgffCoordinateSystem):
            return False
        return self.to_dict() == other.to_dict()

    def equal_up_to_the_units(self, other: NgffCoordinateSystem) -> bool:
        """Check if two coordinate systems are the same based on the axes' names and types (ignoring the units)."""
        if self.name != other.name:
            return False
        if len(self._axes) != len(other._axes):
            return False
        for i, axis in enumerate(self._axes):
            if axis.name != other._axes[i].name:
                return False
            if axis.type != other._axes[i].type:
                return False
        return True

    def equal_up_to_the_name(self, other: NgffCoordinateSystem) -> bool:
        """Checks if two coordinate systems are the same based on the axes (ignoring the coordinate systems names)."""
        return self._axes == other._axes

    def subset(self, axes_names: list[str], new_name: str | None = None) -> NgffCoordinateSystem:
        """
        Return a new coordinate system subsetting the axes.

        Parameters
        ----------
        axes_names
            the axes to keep.

        new_name
            name of the new CoordinateSystem

        Returns
        -------
        a new CoordinateSystem with the subset axes
        """
        axes = [copy.deepcopy(axis) for axis in self._axes if axis.name in axes_names]
        if new_name is None:
            new_name = self.name + "_subset " + str(axes_names)
        return NgffCoordinateSystem(name=new_name, axes=axes)

    @property
    def axes_names(self) -> tuple[str, ...]:
        """Get axes' names"""
        return tuple([ax.name for ax in self._axes])

    @property
    def axes_types(self) -> tuple[str, ...]:
        """Get axes' types"""
        return tuple([ax.type for ax in self._axes])

    def __hash__(self) -> int:
        """compute a hash the object"""
        return hash(frozenset(self.to_dict()))

    def has_axis(self, name: str) -> bool:
        """
        Check the coordinate system has an axis of the given name.

        Parameters
        ----------
        name
            name of the axis.
        """
        return any(axis.name == name for axis in self._axes)

    def get_axis(self, name: str) -> NgffAxis:
        """Get the axis by name"""
        for axis in self._axes:
            if axis.name == name:
                return axis
        raise ValueError(f"NgffAxis {name} not found in {self.name} coordinate system.")

    @staticmethod
    def merge(
        coord_sys1: NgffCoordinateSystem, coord_sys2: NgffCoordinateSystem, new_name: str | None = None
    ) -> NgffCoordinateSystem:
        """
        Merge two coordinate systems

        Parameters
        ----------
        coord_sys1

        coord_sys2

        new_name
            name of the new coordinate system
        """
        # common axes need to be the identical otherwise no merge is made
        common_axes = set(coord_sys1.axes_names).intersection(coord_sys2.axes_names)
        for axis_name in common_axes:
            if coord_sys1.get_axis(axis_name) != coord_sys2.get_axis(axis_name):
                raise ValueError("Common axes are not identical")
        axes = copy.deepcopy(coord_sys1._axes)
        for axis in coord_sys2._axes:
            if axis.name not in common_axes:
                axes.append(axis)
        if new_name is None:
            new_name = coord_sys1.name + "_merged_" + coord_sys2.name
        return NgffCoordinateSystem(name=new_name, axes=axes)

    def set_unit(self, axis_name: str, unit: str) -> None:
        """
        set new units for an axis

        Parameters
        ----------
        axis_name
            name of the axis
        unit
            new units of the axis
        """
        for axis in self._axes:
            if axis.name == axis_name:
                axis.unit = unit
                return
        raise ValueError(f"Axis {axis_name} not found in {self.name} coordinate system.")


def _get_spatial_axes(
    coordinate_system: NgffCoordinateSystem,
) -> list[str]:
    """
    Get the names of the spatial axes (type = 'space') in a coordinate system.

    Parameters
    ----------
    coordinate_system
        The coordinate system to get the spatial axes from.

    Returns
    -------
    The names of the spatial axes.
    """
    return [axis.name for axis in coordinate_system._axes if axis.type == "space"]


def _make_cs(ndim: Literal[2, 3], name: str | None = None, unit: str | None = None) -> NgffCoordinateSystem:
    """helper function to make a yx or zyx coordinate system"""
    if ndim == 2:
        axes = [
            NgffAxis(name="y", type="space", unit=unit),
            NgffAxis(name="x", type="space", unit=unit),
        ]
        if name is None:
            name = "yx"
    elif ndim == 3:
        axes = [
            NgffAxis(name="z", type="space", unit=unit),
            NgffAxis(name="y", type="space", unit=unit),
            NgffAxis(name="x", type="space", unit=unit),
        ]
        if name is None:
            name = "zyx"
    else:
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    return NgffCoordinateSystem(name=name, axes=axes)


def yx_cs(name: str | None = None, unit: str | None = None) -> NgffCoordinateSystem:
    """
    Helper function to create a 2D yx coordinate system.

    Parameters
    ----------
    name
        The name of the coordinate system. A default value of None leads to the name being set to "yx".
    unit
        The unit of the spatial axes. A default value of None leads to the unit being set to "unit".

    Returns
    -------
    The coordinate system.
    """
    return _make_cs(name=name, ndim=2, unit=unit)


def zyx_cs(name: str | None = None, unit: str | None = None) -> NgffCoordinateSystem:
    """
    Helper function to create a 3D zyx coordinate system.

    Parameters
    ----------
    name
        The name of the coordinate system. A default value of None leads to the name being set to "zyx".
    unit
        The unit of the spatial axes. A default value of None leads to the unit being set to "unit".

    Returns
    -------
    The coordinate system.
    """
    return _make_cs(name=name, ndim=3, unit=unit)
