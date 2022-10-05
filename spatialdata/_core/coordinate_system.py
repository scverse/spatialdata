import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

__all__ = ["CoordinateSystem"]

CoordSystem_t = Dict[str, Union[str, List[Dict[str, str]]]]
AXIS_ORDER = ["t", "c", "z", "y", "x"]


class CoordinateSystem:
    def __init__(
        self,
        name: Optional[str] = None,
        axes: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        units: Optional[List[str]] = None,
    ):
        self._name = name
        self._axes = axes if axes is not None else []
        self._types = types if types is not None else []
        self._units = units if units is not None else []

    def from_dict(self, coord_sys: CoordSystem_t) -> None:
        if "name" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have a name.")
        if "axes" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have axes.")

        if TYPE_CHECKING:
            assert isinstance(coord_sys["name"], str)
            assert isinstance(coord_sys["axes"], list)
        self._name = coord_sys["name"]

        for axis in coord_sys["axes"]:
            if "name" not in axis.keys():
                raise ValueError("Each axis MUST have a name.")
            if "type" not in axis.keys():
                raise ValueError("Each axis MUST have a type.")
            if "unit" not in axis.keys():
                raise ValueError("Each axis MUST have a unit.")
            self._axes.append(axis["name"])
            self._types.append(axis["type"])
            self._units.append(axis["unit"])

        if len(self._axes) != len(self._types) != len(self._units):
            raise ValueError("Axes, types, and units MUST be the same length.")

    def to_dict(self) -> CoordSystem_t:
        unsorted_axes = []
        for axis, axis_type, axis_unit in zip(self.axes, self.types, self.units):
            unsorted_axes.append({"name": axis, "type": axis_type, "unit": axis_unit})
        sorted_axes = sorted(unsorted_axes, key=lambda x: AXIS_ORDER.index(x["name"]))
        out: Dict[str, Any] = {"name": self.name, "axes": sorted_axes}
        # if TYPE_CHECKING:
        #     assert isinstance(out["axes"], list)
        return out

    def from_array(self, array: Any) -> None:
        raise NotImplementedError()

    def from_json(self, data: Union[str, bytes]) -> None:
        coord_sys = json.loads(data)
        self.from_dict(coord_sys)

    def to_json(self, **kwargs: Any) -> str:
        out = self.to_dict()
        return json.dumps(out, **kwargs)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CoordinateSystem):
            return False
        return self.to_dict() == other.to_dict()

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def axes(self) -> List[str]:
        return self._axes

    @property
    def types(self) -> List[str]:
        return self._types

    @property
    def units(self) -> List[str]:
        return self._units

    def __hash__(self):
        return hash(frozenset(self.to_dict()))
