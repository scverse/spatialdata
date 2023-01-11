from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, Optional, Union

__all__ = ["CoordinateSystem", "Axis"]

Axis_t = dict[str, str]
CoordSystem_t = dict[str, Union[str, list[dict[str, str]]]]
AXIS_ORDER = ["t", "c", "z", "y", "x"]


class Axis:
    name: str
    type: str
    unit: Optional[str]

    def __init__(self, name: str, type: str, unit: Optional[str] = None):
        self.name = name
        self.type = type
        self.unit = unit

    def __repr__(self) -> str:
        inner = ", ".join(f"'{v}'" for v in self.to_dict().values())
        return f"Axis({inner})"

    def to_dict(self) -> Axis_t:
        d = {"name": self.name, "type": self.type}
        if self.unit is not None:
            d["unit"] = self.unit
        return d

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Axis):
            return False
        return self.to_dict() == other.to_dict()


class CoordinateSystem:
    def __init__(self, name: str, axes: Optional[list[Axis]] = None):
        self._name = name
        self._axes = axes if axes is not None else []
        if len(self._axes) != len({axis.name for axis in self._axes}):
            raise ValueError("Axes names must be unique")

    def __repr__(self) -> str:
        return f"CoordinateSystem('{self.name}', {self._axes})"

    @staticmethod
    def from_dict(coord_sys: CoordSystem_t) -> CoordinateSystem:
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
            axes.append(Axis(name=axis["name"], type=axis["type"], **kw))
        return CoordinateSystem(name=name, axes=axes)

    def to_dict(self) -> CoordSystem_t:
        out: dict[str, Any] = {"name": self.name, "axes": [axis.to_dict() for axis in self._axes]}
        # if TYPE_CHECKING:
        #     assert isinstance(out["axes"], list)
        return out

    def from_array(self, array: Any) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_json(data: Union[str, bytes]) -> CoordinateSystem:
        coord_sys = json.loads(data)
        return CoordinateSystem.from_dict(coord_sys)

    def to_json(self, **kwargs: Any) -> str:
        out = self.to_dict()
        return json.dumps(out, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CoordinateSystem):
            return False
        return self.to_dict() == other.to_dict()

    def equal_up_to_the_units(self, other: CoordinateSystem) -> bool:
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

    def equal_up_to_the_name(self, other: CoordinateSystem) -> bool:
        return self._axes == other._axes

    def subset(self, axes_names: list[str], new_name: Optional[str] = None) -> CoordinateSystem:
        axes = [copy.deepcopy(axis) for axis in self._axes if axis.name in axes_names]
        if new_name is None:
            new_name = self.name + "_subset " + str(axes_names)
        return CoordinateSystem(name=new_name, axes=axes)

    @property
    def name(self) -> str:
        return self._name

    @property
    def axes_names(self) -> tuple[str, ...]:
        return tuple([ax.name for ax in self._axes])

    @property
    def axes_types(self) -> tuple[str, ...]:
        return tuple([ax.type for ax in self._axes])

    def __hash__(self) -> int:
        return hash(frozenset(self.to_dict()))

    def has_axis(self, name: str) -> bool:
        for axis in self._axes:
            if axis.name == name:
                return True
        return False

    def get_axis(self, name: str) -> Axis:
        for axis in self._axes:
            if axis.name == name:
                return axis
        raise ValueError(f"Axis {name} not found in {self.name} coordinate system.")

    @staticmethod
    def merge(
        coord_sys1: CoordinateSystem, coord_sys2: CoordinateSystem, new_name: Optional[str] = None
    ) -> CoordinateSystem:
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
        return CoordinateSystem(name=new_name, axes=axes)


def _get_spatial_axes(
    coordinate_system: CoordinateSystem,
) -> list[str]:
    """Get the names of the spatial axes in a coordinate system.

    Parameters
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system to get the spatial axes from.

    Returns
    -------
    spatial_axis_names : List[str]
        The names of the spatial axes.
    """
    return [axis.name for axis in coordinate_system._axes if axis.type == "space"]
