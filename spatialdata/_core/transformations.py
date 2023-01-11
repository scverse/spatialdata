from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._types import ArrayLike

__all__ = [
    "BaseTransformation",
    "Identity",
    # "MapIndex",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "Sequence",
    # "Displacements",
    # "Coordinates",
    # "VectorField",
    # "InverseOf",
    # "Bijection",
    # "ByDimension",
]
# link pointing to the latest specs from John Bogovic (from his fork of the repo)
# TODO: update this link when the specs are finalized
# http://api.csswg.org/bikeshed/?url=https://raw.githubusercontent.com/bogovicj/ngff/coord-transforms/latest/index.bs
# Transformation_t = Dict[str, Union[str, List[int], List[str], List[Dict[str, Any]]]]
Transformation_t = dict[str, Any]
TRANSFORMATIONS: dict[str, type[BaseTransformation]] = {}


class BaseTransformation(ABC):
    """Base class for all transformations."""

    input_coordinate_system: Optional[CoordinateSystem] = None
    output_coordinate_system: Optional[CoordinateSystem] = None

    def __init__(
        self,
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        self.input_coordinate_system = input_coordinate_system
        self.output_coordinate_system = output_coordinate_system

    def _indent(self, indent: int) -> str:
        return " " * indent * 4

    def _repr_transformation_signature(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{type(self).__name__} ({', '.join(self.input_coordinate_system.axes_names)} -> {', '.join(self.output_coordinate_system.axes_names)})"

    @abstractmethod
    def _repr_transformation_description(self, indent: int = 0) -> str:
        pass

    def _repr_indent(self, indent: int = 0) -> str:
        if isinstance(self, Identity):
            return f"{self._repr_transformation_signature(indent)}"
        else:
            return f"{self._repr_transformation_signature(indent)}\n{self._repr_transformation_description(indent + 1)}"

    def __repr__(self) -> str:
        return self._repr_indent(0)

    @classmethod
    @abstractmethod
    def _from_dict(cls, d: Transformation_t) -> BaseTransformation:
        pass

    @classmethod
    def from_dict(cls, d: Transformation_t) -> BaseTransformation:
        pass

        # d = MappingProxyType(d)
        type = d["type"]
        # MappingProxyType is readonly
        transformation = TRANSFORMATIONS[type]._from_dict(d)
        if "input" in d:
            input_coordinate_system = d["input"]
            if isinstance(input_coordinate_system, dict):
                input_coordinate_system = CoordinateSystem.from_dict(input_coordinate_system)
            transformation.input_coordinate_system = input_coordinate_system
        if "output" in d:
            output_coordinate_system = d["output"]
            if isinstance(output_coordinate_system, dict):
                output_coordinate_system = CoordinateSystem.from_dict(output_coordinate_system)
            transformation.output_coordinate_system = output_coordinate_system
        return transformation

    @abstractmethod
    def to_dict(self) -> Transformation_t:
        pass

    def _update_dict_with_input_output_cs(self, d: Transformation_t) -> None:
        if self.input_coordinate_system is not None:
            d["input"] = self.input_coordinate_system
            if isinstance(d["input"], CoordinateSystem):
                d["input"] = d["input"].to_dict()
        if self.output_coordinate_system is not None:
            d["output"] = self.output_coordinate_system
            if isinstance(d["output"], CoordinateSystem):
                d["output"] = d["output"].to_dict()

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    @abstractmethod
    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        pass

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def to_affine(self) -> Affine:
        pass

    def _validate_transform_points_shapes(self, input_size: int, points_shape: tuple[int, ...]) -> None:
        if len(points_shape) != 2 or points_shape[1] != input_size:
            raise ValueError(
                f"points must be a tensor of shape (n, d), where n is the number of points and d is the "
                f"the number of spatial dimensions. Points shape: {points_shape}, input size: {input_size}"
            )

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformation: BaseTransformation) -> BaseTransformation:
        return Sequence([self, transformation])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseTransformation):
            raise NotImplementedError("Cannot compare BaseTransformation with other types")
        return self.to_dict() == other.to_dict()

    def _get_axes_from_coordinate_systems(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not isinstance(self.input_coordinate_system, CoordinateSystem):
            raise ValueError("Input coordinate system not specified")
        if not isinstance(self.output_coordinate_system, CoordinateSystem):
            raise ValueError("Output coordinate system not specified")
        input_axes = self.input_coordinate_system.axes_names
        output_axes = self.output_coordinate_system.axes_names
        return input_axes, output_axes

    @staticmethod
    def _parse_list_into_array(array: Union[list[Number], ArrayLike]) -> ArrayLike:
        if isinstance(array, list):
            array = np.array(array)
        if array.dtype != float:
            array = array.astype(float)
        return array


# A note on affine transformations and their matrix representation.
# Some transformations can be interpreted as (n-dimensional) affine transformations; explicitly these transformations
# are:
# - identity
# - mapIndex
# - mapAxis
# - translation
# - scale
# - affine
# - rotation
# - sequence (if the components are of the types described here)
# - inverseOf (if the components are of the types described here)
# - bijection (if the components are of the types described here)
# - byDimension (if the components are of the types described here)
# In general the conversion between one of these classes to (or from) its matrix form requires the knowledge of the
# input and output axes.
#
# An example.
# The relationship between the input/ouput axes and the matrix form is shown in the following
# example, of an affine transformation between the input axes (x, y) and the output axes (c, y, x), which simply maps
# the input axes to the output axes respecting the axis names, and then translates by the vector (x, y) = (3, 2):
#   x  y
# c 0  0 0
# y 0  1 2
# x 1  0 3
#   0  0 1
# to apply the above affine transformation A to a point, say the point (x, y) = (a, b), you multiply the matrix by the
# vector (a, b, 1); you always need to add 1 as the last element.
#   x  y
# c 0  0 0   *   a
# y 0  1 2       b
# x 1  0 3       1
#   0  0 1
# Notice that if the input space had been y, x, the point (x, y) = (a, b) would have led to the column vector (b, a, 1)
#
# Notice that:
# - you can think of the input axes as labeling the columns of the matrix
# - you can think of the output axes as labeling the rows of the matrix
# - the last rows is always the vector [0, 0, ..., 0, 1]
# - the last column is always a translation vector, plus the element 1 in the last position
#
# A more theoretical note.
# The matrix representation of an affine transformation has the above form thanks to the fact an affine
# transformation is a particular case of a projective transformation, and the affine space can be seen as the
# projective space without the "line at infinity". The vector representation of a point that belongs to the line
# at the infinity has the last coordinate equal to 0. This is the reason why when applying an affine transformation
# to a point we set the last element of the point to 1, and in this way the affine transformation leaves the affine
# space invariant (i.e. it does not map finite points to the line at the infinity).
# For a primer you can look here: https://en.wikipedia.org/wiki/Affine_space#Relation_to_projective_spaces
# For more information please consult a linear algebra textbook.
class Identity(BaseTransformation):
    def __init__(
        self,
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        super().__init__(input_coordinate_system, output_coordinate_system)

    # TODO: remove type: ignore[valid-type] when https://github.com/python/mypy/pull/14041 is merged
    @classmethod
    def _from_dict(cls, _: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls()

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "identity",
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return ""

    def inverse(self) -> BaseTransformation:
        return Identity(
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError("Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points

    def to_affine(self) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        return Affine(
            np.eye(len(input_axes) + 1),
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


# # maybe this transformation will not make it to the final specs, waiting before implementing this
# class MapIndex(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
class MapAxis(BaseTransformation):
    def __init__(
        self,
        map_axis: dict[str, str],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        super().__init__(input_coordinate_system, output_coordinate_system)
        self.map_axis = map_axis

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls(d["mapAxis"])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "mapAxis",
            "mapAxis": self.map_axis,
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for k, v in self.map_axis.items():
            s += f"{self._indent(indent)}{k} -> {v}\n"
        s = s[:-1]
        return s

    def inverse(self) -> BaseTransformation:
        if len(self.map_axis.keys()) != len(set(self.map_axis.values())):
            raise ValueError("Cannot invert a map axis transformation with different number of input and output axes")
        else:
            return MapAxis(
                {v: k for k, v in self.map_axis.items()},
                input_coordinate_system=self.output_coordinate_system,
                output_coordinate_system=self.input_coordinate_system,
            )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if not set(input_axes).issuperset(set(self.map_axis.values())):
            raise ValueError(
                "Each value of the dict specifying a MapAxis transformation must be an axis of the input "
                "coordinate system"
            )
        if set(output_axes) != set(self.map_axis.keys()):
            raise ValueError(
                "The set of output axes must be the same as the set of keys the dict specifying a "
                "MapAxis transformation"
            )
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        new_indices = [input_axes.index(self.map_axis[ax]) for ax in output_axes]
        assert len(new_indices) == len(output_axes)
        mapped = points[:, new_indices]
        assert type(mapped) == np.ndarray
        return mapped

    def to_affine(self) -> Affine:
        input_axes, output_axes = self._get_and_validate_axes()
        matrix: ArrayLike = np.zeros((len(output_axes) + 1, len(input_axes) + 1), dtype=float)
        matrix[-1, -1] = 1
        for i, des_axis in enumerate(output_axes):
            for j, src_axis in enumerate(input_axes):
                if src_axis == self.map_axis[des_axis]:
                    matrix[i, j] = 1
        affine = Affine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )
        return affine


class Translation(BaseTransformation):
    def __init__(
        self,
        translation: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        """
        class for storing translation transformations.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        self.translation = self._parse_list_into_array(translation)

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls(d["translation"])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "translation",
            "translation": self.translation.tolist(),
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.translation}"

    def inverse(self) -> BaseTransformation:
        return Translation(
            -self.translation,
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError("Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points + self.translation

    def to_affine(self, ndims_input: Optional[int] = None, ndims_output: Optional[int] = None) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, -1] = self.translation
        return Affine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class Scale(BaseTransformation):
    def __init__(
        self,
        scale: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        """
        class for storing scale transformations.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        self.scale = self._parse_list_into_array(scale)

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls(d["scale"])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "scale",
            "scale": self.scale.tolist(),
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"{self._indent(indent)}{self.scale}"

    def inverse(self) -> BaseTransformation:
        new_scale = np.zeros_like(self.scale)
        new_scale[np.nonzero(self.scale)] = 1 / self.scale[np.nonzero(self.scale)]
        return Scale(
            new_scale,
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError("Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points * self.scale

    def to_affine(self) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, :-1] = np.diag(self.scale)
        return Affine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class Affine(BaseTransformation):
    def __init__(
        self,
        affine: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        """
        class for storing affine transformations.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        self.affine = self._parse_list_into_array(affine)

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        assert isinstance(d["affine"], list)
        last_row = [[0.0] * (len(d["affine"][0]) - 1) + [1.0]]
        return cls(d["affine"] + last_row)

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "affine",
            "affine": self.affine[:-1, :].tolist(),
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for row in self.affine:
            s += f"{self._indent(indent)}{row}\n"
        s = s[:-1]
        return s

    def inverse(self) -> BaseTransformation:
        inv = np.linalg.inv(self.affine)
        return Affine(
            inv,
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        return q[: len(output_axes), :].T  # type: ignore[no-any-return]

    def to_affine(self) -> Affine:
        return Affine(
            self.affine,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )

    @classmethod
    def _affine_matrix_from_input_and_output_axes(
        cls, input_axes: tuple[str, ...], output_axes: tuple[str, ...]
    ) -> ArrayLike:
        from spatialdata._core.core_utils import C, X, Y, Z

        assert all([ax in (X, Y, Z, C) for ax in input_axes])
        assert all([ax in (X, Y, Z, C) for ax in output_axes])
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        for output_ax in output_axes:
            for input_ax in input_axes:
                if output_ax == input_ax:
                    m[output_axes.index(output_ax), input_axes.index(input_ax)] = 1
        m[-1, -1] = 1
        return m

    @classmethod
    def from_input_output_coordinate_systems(
        cls,
        input_coordinate_system: CoordinateSystem,
        output_coordinate_system: CoordinateSystem,
    ) -> Affine:
        input_axes = input_coordinate_system.axes_names
        output_axes = output_coordinate_system.axes_names
        m = cls._affine_matrix_from_input_and_output_axes(input_axes, output_axes)
        return cls(
            affine=m, input_coordinate_system=input_coordinate_system, output_coordinate_system=output_coordinate_system
        )


class Rotation(BaseTransformation):
    def __init__(
        self,
        rotation: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        """
        class for storing rotation transformations.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        self.rotation = self._parse_list_into_array(rotation)

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        x = d["rotation"]
        n = len(x)
        r = math.sqrt(n)
        assert n == int(r * r)
        m = np.array(x).reshape((int(r), int(r))).tolist()
        return cls(m)

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "rotation",
            "rotation": self.rotation.ravel().tolist(),
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for row in self.rotation:
            s += f"{self._indent(indent)}{row}\n"
        s = s[:-1]
        return s

    def inverse(self) -> BaseTransformation:
        return Rotation(
            self.rotation.T,
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError("Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return (self.rotation @ points.T).T

    def to_affine(self) -> Affine:
        m = np.eye(len(self.rotation) + 1)
        m[:-1, :-1] = self.rotation
        return Affine(
            m,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class Sequence(BaseTransformation):
    def __init__(
        self,
        transformations: list[BaseTransformation],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        super().__init__(input_coordinate_system, output_coordinate_system)
        # we can decide to treat an empty sequence as an Identity if we need to
        assert len(transformations) > 0
        self.transformations = transformations

        if (cs := self.transformations[0].input_coordinate_system) is not None:
            if self.input_coordinate_system is not None:
                assert cs == self.input_coordinate_system
            else:
                self.input_coordinate_system = cs
        if (cs := self.transformations[-1].output_coordinate_system) is not None:
            if self.output_coordinate_system is not None:
                assert cs == self.output_coordinate_system
            else:
                self.output_coordinate_system = cs

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls([BaseTransformation.from_dict(t) for t in d["transformations"]])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "sequence",
            "transformations": [t.to_dict() for t in self.transformations],
        }
        self._update_dict_with_input_output_cs(d)
        return d

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for t in self.transformations:
            s += f"{t._repr_indent(indent=indent)}\n"
        s = s[:-1]
        return s

    def inverse(self) -> BaseTransformation:
        return Sequence(
            [t.inverse() for t in reversed(self.transformations)],
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        return input_axes, output_axes

    @staticmethod
    def _inferring_cs_infer_output_coordinate_system(
        t: BaseTransformation,
    ) -> Optional[CoordinateSystem]:
        assert isinstance(t.input_coordinate_system, CoordinateSystem)
        if isinstance(t, Affine):
            return None
        elif isinstance(t, Translation) or isinstance(t, Scale) or isinstance(t, Rotation) or isinstance(t, Identity):
            return t.input_coordinate_system
        elif isinstance(t, MapAxis):
            return None
        elif isinstance(t, Sequence):
            t.check_and_infer_coordinate_systems()
            return t.output_coordinate_system
        else:
            return None

    @staticmethod
    def _inferring_cs_pre_action(t: BaseTransformation, latest_output_cs: CoordinateSystem) -> BaseTransformation:
        adjusted: Optional[BaseTransformation] = None
        input_cs = t.input_coordinate_system
        if input_cs is None:
            t.input_coordinate_system = latest_output_cs
        else:
            assert isinstance(input_cs, CoordinateSystem), input_cs
            if not input_cs.equal_up_to_the_units(latest_output_cs):
                # fix the mismatched coordinate systems by adding an affine that permutes/add/remove axes
                identity = Identity(input_coordinate_system=input_cs, output_coordinate_system=input_cs)
                adjusted = _adjust_transformation_between_mismatching_coordinate_systems(identity, latest_output_cs)
        output_cs = t.output_coordinate_system
        expected_output_cs = Sequence._inferring_cs_infer_output_coordinate_system(t)
        if output_cs is None:
            if expected_output_cs is None:
                raise ValueError(
                    f"Cannot infer output coordinate system for {t}, this could happen for instance if "
                    f"passing an Affine transformation as a component of a Sequence transformation "
                    f"without specifying the input and output coordinate system for the Affine "
                    f"transformation."
                )
            t.output_coordinate_system = expected_output_cs
        else:
            assert isinstance(output_cs, CoordinateSystem)
            # if it is not possible to infer the output, like for Affine, we skip this check
            if expected_output_cs is not None:
                assert t.output_coordinate_system == expected_output_cs
        if adjusted is None:
            return t
        else:
            return adjusted

    def check_and_infer_coordinate_systems(self) -> None:
        """
        Check that the coordinate systems of the components are consistent or infer them when possible.

        Notes
        -----
        The NGFF specs allow for Sequence transformations to be specified even without making all the coordinate
        systems of their component explicit. This reduces verbosity but can create some inconsistencies. This method
        infers missing coordinate systems when possible and throws an error otherwise. Furthermore, we allow the
        composition of transformations with different coordinate systems up to a reordering of the axes, by inserting
        opportune Affine transformations.

        This method is called automatically when a transformation:
        - is applied (transform_points())
        - is inverted (inverse())
        - needs to be saved/converted (to_dict(), to_affine())
        """
        assert type(self.input_coordinate_system) == CoordinateSystem
        latest_output_cs: CoordinateSystem = self.input_coordinate_system
        new_transformations = []
        for t in self.transformations:
            adjusted = Sequence._inferring_cs_pre_action(t, latest_output_cs)
            new_transformations.append(adjusted)
            assert adjusted.output_coordinate_system is not None
            latest_output_cs = adjusted.output_coordinate_system
        if self.output_coordinate_system is not None:
            if not self.output_coordinate_system.equal_up_to_the_units(latest_output_cs):
                # fix the mismatched coordinate systems by adding an affine that permutes/add/remove axes
                identity = Identity(
                    input_coordinate_system=self.output_coordinate_system,
                    output_coordinate_system=self.output_coordinate_system,
                )
                adjusted = _adjust_transformation_between_mismatching_coordinate_systems(identity, latest_output_cs)
                new_transformations.append(adjusted)
                # raise ValueError(
                #     "The output coordinate system of the Sequence transformation is not consistent with the "
                #     "output coordinate system of the last component transformation."
                # )
        else:
            self.output_coordinate_system = self.transformations[-1].output_coordinate_system

        # final check
        self.transformations = new_transformations
        assert self.input_coordinate_system == self.transformations[0].input_coordinate_system
        for i in range(len(self.transformations)):
            assert self.transformations[i].output_coordinate_system is not None
            if i < len(self.transformations) - 1:
                assert (
                    self.transformations[i].output_coordinate_system
                    == self.transformations[i + 1].input_coordinate_system
                )
        assert self.output_coordinate_system == self.transformations[-1].output_coordinate_system

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        # the specs allow to compose transformations without specifying the input and output coordinate systems of
        # every transformation. Since in order to apply a transformation we need to know the input and output coordinate
        # systems, we allow for on-the-fly computation by inferring this in real-time. The inferred information is
        # then removed, so to leave the object as it was found before. To be precise, some transformation don't
        # strictly require the input and output coordinate system to be specified when applying the transformation,
        # because they "trust" the input and output to be correct (for example Identity). But requiring the
        # coordinate systems to be specified we can enforce a check that could help catch errors, so the extra
        # complexity in these Sequence class will be rewarded in the long term.
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        assert type(self.input_coordinate_system) == CoordinateSystem
        self.check_and_infer_coordinate_systems()
        for t in self.transformations:
            points = t.transform_points(points)
        return points

    def to_affine(self) -> Affine:
        # the same comment on the coordinate systems of the various transformations, made on the transform_points()
        # method, applies also here
        input_axes, output_axes = self._get_and_validate_axes()
        composed = np.eye(len(input_axes) + 1)
        self.check_and_infer_coordinate_systems()
        for t in self.transformations:
            a = t.to_affine()
            composed = a.affine @ composed
        return Affine(
            composed,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


# class Displacements(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
#
# # this class is not in the ngff transform specification and is a prototype
# class VectorField(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
#
# class Coordinates(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
#
# class InverseOf(BaseTransformation):
#     def __init__(self, transformation: Union[Dict[str, Any], BaseTransformation]) -> None:
#         if isinstance(transformation, BaseTransformation):
#             self.transformation = transformation
#         else:
#             self.transformation = BaseTransformation.from_dict(transformation)
#         self._ndim = self.transformation.ndim
#
#     @property
#     def src_dim(self) -> Optional[int]:
#         return self._ndim
#
#     @property
#     def des_dim(self) -> Optional[int]:
#         return self._ndim
#
#     @property
#     def ndim(self) -> Optional[int]:
#         # support mixed ndim and remove this property
#         return self._ndim
#
#     def to_dict(self) -> Transformation_t:
#         return {
#             "type": "inverseOf",
#             "transformation": self.transformation.to_dict(),
#         }
#
#     def transform_points(self, points: ArrayLike) -> ArrayLike:
#         return self.transformation.inverse().transform_points(points)
#
#     def inverse(self) -> BaseTransformation:
#         return self.transformation
#
#
# class Bijection(BaseTransformation):
#     def __init__(
#         self, forward: Union[Dict[str, Any], BaseTransformation], inverse: Union[Dict[str, Any], BaseTransformation]
#     ) -> None:
#         if isinstance(forward, BaseTransformation):
#             self.forward = forward
#         else:
#             self.forward = BaseTransformation.from_dict(forward)
#
#         if isinstance(inverse, BaseTransformation):
#             self._inverse = inverse
#         else:
#             self._inverse = BaseTransformation.from_dict(inverse)
#         assert self.forward.ndim == self._inverse.ndim
#         self._ndim = self.forward.ndim
#
#     @property
#     def src_dim(self) -> Optional[int]:
#         return self._ndim
#
#     @property
#     def des_dim(self) -> Optional[int]:
#         return self._ndim
#
#     @property
#     def ndim(self) -> Optional[int]:
#         return self._ndim
#
#     def to_dict(self) -> Transformation_t:
#         return {
#             "type": "bijection",
#             "forward": self.forward.to_dict(),
#             "inverse": self._inverse.to_dict(),
#         }
#
#     def transform_points(self, points: ArrayLike) -> ArrayLike:
#         return self.forward.transform_points(points)
#
#     def inverse(self) -> BaseTransformation:
#         return self._inverse
class ByDimension(BaseTransformation):
    def __init__(
        self,
        transformations: list[BaseTransformation],
        input_coordinate_system: Optional[CoordinateSystem] = None,
        output_coordinate_system: Optional[CoordinateSystem] = None,
    ) -> None:
        super().__init__(input_coordinate_system, output_coordinate_system)
        assert len(transformations) > 0
        self.transformations = transformations

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls([BaseTransformation.from_dict(t) for t in d["transformations"]])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "byDimension",
            "transformations": [t.to_dict() for t in self.transformations],
        }
        self._update_dict_with_input_output_cs(d)
        return d

    # same code as in Sequence
    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for t in self.transformations:
            s += f"{t._repr_indent(indent=indent)}\n"
        s = s[:-1]
        return s

    def inverse(self) -> BaseTransformation:
        inverse_transformations = [t.inverse() for t in self.transformations]
        return ByDimension(
            inverse_transformations,
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        # we check that:
        # 1. each input from each transformation in self.transformation must appear in the set of input axes
        # 2. each output from each transformation in self.transformation must appear at most once in the set of output
        # axes
        defined_output_axes: set[str] = set()
        for t in self.transformations:
            assert isinstance(t.input_coordinate_system, CoordinateSystem)
            assert isinstance(t.output_coordinate_system, CoordinateSystem)
            for ax in t.input_coordinate_system.axes_names:
                assert ax in input_axes
            for ax in t.output_coordinate_system.axes_names:
                assert ax not in defined_output_axes
                defined_output_axes.add(ax)
        assert defined_output_axes == set(output_axes)
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        output_columns: dict[str, ArrayLike] = {}
        for t in self.transformations:
            assert isinstance(t.input_coordinate_system, CoordinateSystem)
            assert isinstance(t.output_coordinate_system, CoordinateSystem)
            input_columns = [points[:, input_axes.index(ax)] for ax in t.input_coordinate_system.axes_names]
            input_columns_stacked: ArrayLike = np.stack(input_columns, axis=1)
            output_columns_t = t.transform_points(input_columns_stacked)
            for ax, col in zip(t.output_coordinate_system.axes_names, output_columns_t.T):
                output_columns[ax] = col
        output: ArrayLike = np.stack([output_columns[ax] for ax in output_axes], axis=1)
        return output

    def to_affine(self) -> Affine:
        input_axes, output_axes = self._get_and_validate_axes()
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        m[-1, -1] = 1
        for t in self.transformations:
            assert isinstance(t.input_coordinate_system, CoordinateSystem)
            assert isinstance(t.output_coordinate_system, CoordinateSystem)
            t_affine = t.to_affine()
            target_output_indices = [output_axes.index(ax) for ax in t.output_coordinate_system.axes_names]
            target_input_indices = [input_axes.index(ax) for ax in t.input_coordinate_system.axes_names] + [-1]
            m[np.ix_(target_output_indices, target_input_indices)] = t_affine.affine[:-1, :]
        return Affine(
            m,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )

    # function proposed by isaac, see https://github.com/scverse/spatialdata/issues/39
    # I started implementing it but I am not sure it's needed since now I use: _adjust_transformation_between_mismatching_coordinate_systems()
    # def pass_axes_through(self, input_coordinate_system: CoordinateSystem) -> ByDimension:
    #     """
    #     Returns a new ByDimension transformation that passes the axes through without transforming them.
    #     """
    #     output_transformations = deepcopy(self.transformations)
    #
    #     transformed_dims = []
    #     for sub_t in self.transformations:
    #         transformed_dims.extend(sub_t.input_coordinate_system.axes_names)
    #     untransformed = list(set(input_coordinate_system.axes_names).difference(transformed_dims))
    #     # TODO: is Identify good or do we need to use an affine?
    #     output_transformations.append(Identity(untransformed, untransformed))
    #
    #     # TODO: join the coordinate systems
    #     # TODO: add tests for joining coordinate systems
    #     # TODO: add tests for passing axes through
    #     output_coordinate_system = ...  # compute from ByDimensions
    #
    #     return ByDimension(
    #         input_coordinate_space=input_coordinate_system,
    #         output_coordinate_space=output_coordinate_system,
    #         transforms=output_transformations,
    #     )


TRANSFORMATIONS["identity"] = Identity
TRANSFORMATIONS["mapAxis"] = MapAxis
TRANSFORMATIONS["translation"] = Translation
TRANSFORMATIONS["scale"] = Scale
TRANSFORMATIONS["affine"] = Affine
TRANSFORMATIONS["rotation"] = Rotation
TRANSFORMATIONS["sequence"] = Sequence
TRANSFORMATIONS["byDimension"] = ByDimension


def _adjust_transformation_between_mismatching_coordinate_systems(
    t: BaseTransformation, input_coordinate_system: CoordinateSystem
) -> BaseTransformation:
    """
    Adjusts a transformation (t) whose input coordinate system does not match the on of the one of the element it should be applied to (input_coordinate_system).

    Parameters
    ----------
    t
        The transformation to adjust.
    input_coordinate_system
        The coordinate system of the element the transformation should be applied to.

    Returns
    -------
    The adjusted transformation.

    Notes
    -----
    The function behaves as follows:
    - if input_coordinate_system coincides with t.input_coordinate_system, the function returns t
    - if input_coordinate_system coincides with t.input_coordinate_system up to a permutation of the axes, the function
        returns a composition a permutation of the axes and t, and then a permutation back to to the original
    - if input_coordinate_system is a subset of t.input_coordinate_system, the function returns a new transformation
    which only transforms the coordinates in input_coordinate_system, eventually permuting the axes, and then permuting them back after the transfomation
    - if input_coordinate_system is a superset of t.input_coordinate_system, the function returns a new
    transformation which passes the coordinates in input_coordinate_system - t.input_coordinate_system through
    without transforming them, and applies the transformation t to the coordinates in t.input_coordinate_system as in. Finally it permutes the axes back to the original order, eventually passing through eventual axes that have been added by the transformation
    - in input_coordinate_system is neither a subset nor a superset of t.input_coordinate_system, the functions behaves like in the previous two cases on the relevant axes
    """
    cs_left = copy.deepcopy(input_coordinate_system)
    cs_right = copy.deepcopy(t.input_coordinate_system)
    common = set(cs_left.axes_names).intersection(set(cs_right.axes_names))
    only_left = set(cs_left.axes_names).difference(set(cs_right.axes_names))
    only_right = set(cs_right.axes_names).difference(set(cs_left.axes_names))
    if len(only_left) == 0 and len(only_right) == 0:
        return t
    else:
        # order of axes and units don't matter, this transformation will be bundled in a sequence whose final
        # coordinate system will have the right order and correct units
        cs_only_left = cs_left.subset(list(only_left))
        cs_common = cs_left.subset(list(common))
        cs_only_right = cs_right.subset(list(only_right))
        cs_merged = CoordinateSystem.merge(cs_left, cs_right)

        map_axis = MapAxis(
            {ax: ax for ax in cs_left.axes_names},
            input_coordinate_system=cs_only_left,
            output_coordinate_system=cs_only_left,
        )

        pass_through_only_left = Identity(cs_only_left, cs_only_left)

        m = np.zeros((len(only_right) + 1, 2))
        m[-1, -1] = 1
        if len(common) > 0:
            any_axis = cs_common._axes[0]
        else:
            assert len(only_left) > 0
            any_axis = cs_only_left._axes[0]
        cs_any_axis = cs_left.subset([any_axis])
        add_empty_only_right = Affine(
            m,
            input_coordinate_system=cs_any_axis,
            output_coordinate_system=cs_only_right,
        )
        by_dimension = ByDimension(
            [map_axis, pass_through_only_left, add_empty_only_right],
            input_coordinate_system=cs_left,
            output_coordinate_system=cs_merged,
        )
        sequence = Sequence(
            [by_dimension, t],
        )
        print(sequence)
        print(sequence)
        print(sequence)
        print(sequence.to_affine().affine)
        return sequence

    # if t.input_coordinate_system is not None and not t.input_coordinate_system.equal_up_to_the_units(element_cs):
    #     # for mypy so that it doesn't complain in the logger.info() below
    #     assert adjusted.input_coordinate_system is not None
    #     assert adjusted.output_coordinate_system is not None
    #     logger.info(
    #         f"Adding a transformation ({adjusted.input_coordinate_system.axes_names} -> "
    #         f"{adjusted.output_coordinate_system.axes_names}) to adjust for mismatched coordinate systems in the "
    #         "Sequence object"
    #     )
    #     new_t = adjusted
    # else:
    #     new_t = t
    # pass
