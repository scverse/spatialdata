from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

__all__ = ["NgffCoordinateSystem", "NgffAxis"]

Axis_t = dict[str, str]
CoordSystem_t = dict[str, Union[str, list[dict[str, str]]]]
AXIS_ORDER = ["t", "c", "z", "y", "x"]


class NgffAxis:

    """A class for Ngff-format axes
    
    Attributes
    ----------
    name : str
        name of the axis. Should be in ["t", "c", "z", "y", "x"].
    type : str
        type of the axis. Should be in ["type", "channel", "space"].
    unit : TYPE
        unit of the axis. For set of options see https://ngff.openmicroscopy.org/
    """
    
    name: str
    type: str
    unit: Optional[str]

    def __init__(self, name: str, type: str, unit: Optional[str] = None):
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

    """An Ngff-format coordinate system
        Parameters
        ----------
        name : str
            name of the coordinate system
        axes : Optional[list[NgffAxis]], optional
            names of the axes
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
        if "name" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have a name.")
        if "axes" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have axes.")

        if TYPE_CHECKING:
            assert isinstance(coord_sys["name"], str)
            assert isinstance(coord_sys["axes"], list)
        name = coord_sys["name"]

        # sorted_axes = sorted(coord_sys["axes"], key=lambda x: AXIS_ORDER.index(x["name"]))
        axes = []
        for axis in coord_sys["axes"]:
            # for axis in sorted_axes:
            if "name" not in axis.keys():
                raise ValueError("Each axis MUST have a name.")
            if "type" not in axis.keys():
                raise ValueError("Each axis MUST have a type.")
            if "unit" not in axis.keys():
                if not axis["type"] in ["channel", "array"]:
                    raise ValueError("Each axis is either of type channel either MUST have a unit.")
            kw = {}
            if "unit" in axis.keys():
                kw = {"unit": axis["unit"]}
            axes.append(NgffAxis(name=axis["name"], type=axis["type"], **kw))
        return NgffCoordinateSystem(name=name, axes=axes)

    def to_dict(self) -> CoordSystem_t:
        out: dict[str, Any] = {"name": self.name, "axes": [axis.to_dict() for axis in self._axes]}
        # if TYPE_CHECKING:
        #     assert isinstance(out["axes"], list)
        return out

    def from_array(self, array: Any) -> None:
        """Reading form array"""
        raise NotImplementedError()

    @staticmethod
    def from_json(data: Union[str, bytes]) -> NgffCoordinateSystem:
        """Reading form json"""
        coord_sys = json.loads(data)
        return NgffCoordinateSystem.from_dict(coord_sys)

    def to_json(self, **kwargs: Any) -> str:
        """Writing into json"""
        out = self.to_dict()
        return json.dumps(out, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NgffCoordinateSystem):
            return False
        return self.to_dict() == other.to_dict()

    def equal_up_to_the_units(self, other: NgffCoordinateSystem) -> bool:
        """Checks if two CS are the same based on the axes' names and types (not units)"""
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
        """Checks if two CS are the same based on the axes"""
        return self._axes == other._axes

    def subset(self, axes_names: list[str], new_name: Optional[str] = None) -> NgffCoordinateSystem:
        """Querys a subset of axes by axes' names
        
        Parameters
        ----------
        axes_names : list[str]
            
        new_name : Optional[str], optional
            name of the new CoordinateSystem
        
        Returns
        -------
        NgffCoordinateSystem
            a new CoordinateSystem with the subset axes
        """
        axes = [copy.deepcopy(axis) for axis in self._axes if axis.name in axes_names]
        if new_name is None:
            new_name = self.name + "_subset " + str(axes_names)
        return NgffCoordinateSystem(name=new_name, axes=axes)

    @property
    def axes_names(self) -> tuple[str, ...]:
        """Gets axe's names"""
        return tuple([ax.name for ax in self._axes])

    @property
    def axes_types(self) -> tuple[str, ...]:
        """Gets axes' types"""
        return tuple([ax.type for ax in self._axes])

    def __hash__(self) -> int:
        """hashes the object"""
        return hash(frozenset(self.to_dict()))

    def has_axis(self, name: str) -> bool:
        """Check if the axis exists in the Coordinate system by name
        
        Parameters
        ----------
        name : str
            Name of the axis
        """
        for axis in self._axes:
            if axis.name == name:
                return True
        return False

    def get_axis(self, name: str) -> NgffAxis:
        """Get axis by name"""
        for axis in self._axes:
            if axis.name == name:
                return axis
        raise ValueError(f"NgffAxis {name} not found in {self.name} coordinate system.")

    @staticmethod
    def merge(
        coord_sys1: NgffCoordinateSystem, coord_sys2: NgffCoordinateSystem, new_name: Optional[str] = None
    ) -> NgffCoordinateSystem:
        """Merges two coordinate systems
        
        Parameters
        ----------
        coord_sys1 : NgffCoordinateSystem
            
        coord_sys2 : NgffCoordinateSystem

        new_name : Optional[str], optional
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
        """sets units of an axis
        
        Parameters
        ----------
        axis_name : str
            name of the axis
        unit : str
            unit of the axis
        """
        for axis in self._axes:
            if axis.name == axis_name:
                axis.unit = unit
                return
        raise ValueError(f"Axis {axis_name} not found in {self.name} coordinate system.")


def _get_spatial_axes(
    coordinate_system: NgffCoordinateSystem,
) -> list[str]:
    """Get the names of the spatial axes in a coordinate system.
    
    Parameters
    ----------
    coordinate_system : NgffCoordinateSystem
        The coordinate system to get the spatial axes from.
    
    No Longer Returned
    ------------------
    spatial_axis_names : List[str]
        The names of the spatial axes.
    """
    return [axis.name for axis in coordinate_system._axes if axis.type == "space"]


def _make_cs(ndim: Literal[2, 3], name: Optional[str] = None, unit: Optional[str] = None) -> NgffCoordinateSystem:
    """makes a coordinate system"""
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


def yx_cs(name: Optional[str] = None, unit: Optional[str] = None) -> NgffCoordinateSystem:
    """Create a 2D yx coordinate system.
    
    Parameters
    ----------
    name : Optional[str], optional
        The name of the coordinate system. A default value of None leads to the name being set to "yx".
    unit : Optional[str], optional
        The unit of the spatial axes. A default value of None leads to the unit being set to "unit".
    
    Returns
    -------
    NgffCoordinateSystem
        The coordinate system.
    """
    return _make_cs(name=name, ndim=2, unit=unit)


def zyx_cs(name: Optional[str] = None, unit: Optional[str] = None) -> NgffCoordinateSystem:
    """Create a 3D zyx coordinate system.
    
    Parameters
    ----------
    name : Optional[str], optional
        The name of the coordinate system. A default value of None leads to the name being set to "zyx".
    unit : Optional[str], optional
        The unit of the spatial axes. A default value of None leads to the unit being set to "unit".
    
    Returns
    -------
    NgffCoordinateSystem
        The coordinate system.
    """
    return _make_cs(name=name, ndim=3, unit=unit)
