import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

__all__ = ["CoordinateSystem"]

CoordSystem_t = Dict[str, Union[str, List[Dict[str, str]]]]
AXIS_ORDER = ["t", "c", "z", "y", "x"]


class Axis:
    name: str
    type: str
    unit: Optional[str]

    def __init__(self, name: str, type: str, unit: Optional[str] = None):
        self.name = name
        self.type = type
        self.unit = unit

    def to_dict(self):
        d = {"name": self.name, "type": self.type}
        if self.unit is not None:
            d["unit"] = self.unit
        return d


class CoordinateSystem:
    def __init__(self, name: Optional[str] = None, axes: Optional[List[Axis]] = None):
        self._name = name
        self._axes = axes if axes is not None else []

    def from_dict(self, coord_sys: CoordSystem_t) -> None:
        if "name" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have a name.")
        if "axes" not in coord_sys.keys():
            raise ValueError("`coordinate_system` MUST have axes.")

        if TYPE_CHECKING:
            assert isinstance(coord_sys["name"], str)
            assert isinstance(coord_sys["axes"], list)
        self._name = coord_sys["name"]

        # sorted_axes = sorted(coord_sys["axes"], key=lambda x: AXIS_ORDER.index(x["name"]))
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
            self._axes.append(Axis(name=axis["name"], type=axis["type"], **kw))

    def to_dict(self) -> CoordSystem_t:
        out: Dict[str, Any] = {"name": self.name, "axes": [axis.to_dict() for axis in self.axes]}
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

    def __hash__(self):
        return hash(frozenset(self.to_dict()))
