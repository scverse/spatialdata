import math
from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
from typing_extensions import Self

from spatialdata._types import ArrayLike
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem

__all__ = [
    "NgffBaseTransformation",
    "NgffIdentity",
    # "MapIndex",
    "NgffMapAxis",
    "NgffTranslation",
    "NgffScale",
    "NgffAffine",
    "NgffRotation",
    "NgffSequence",
    # "Displacements",
    # "Coordinates",
    # "InverseOf",
    # "Bijection",
    "NgffByDimension",
]
# link pointing to the latest specs from John Bogovic (from his fork of the repo)
# TODO: update this link when the specs are finalized
# http://api.csswg.org/bikeshed/?url=https://raw.githubusercontent.com/bogovicj/ngff/coord-transforms/latest/index.bs
# Transformation_t = Dict[str, Union[str, List[int], List[str], List[Dict[str, Any]]]]
Transformation_t = dict[str, Any]


class NgffBaseTransformation(ABC):
    """Base class for all the transformations defined by the NGFF specification."""

    input_coordinate_system: Optional[NgffCoordinateSystem] = None
    output_coordinate_system: Optional[NgffCoordinateSystem] = None

    def __init__(
        self,
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        self.input_coordinate_system = input_coordinate_system
        self.output_coordinate_system = output_coordinate_system

    def _indent(self, indent: int) -> str:
        return " " * indent * 4

    def _repr_transformation_signature(self, indent: int = 0) -> str:
        domain = ", ".join(self.input_coordinate_system.axes_names) if self.input_coordinate_system is not None else ""
        if self.output_coordinate_system is not None:
            codomain = ", ".join(self.output_coordinate_system.axes_names)
        else:
            codomain = ""
        return f"{self._indent(indent)}{type(self).__name__} ({domain} -> {codomain})"

    @abstractmethod
    def _repr_transformation_description(self, indent: int = 0) -> str:
        """Helper function to compute the __repr__ of the transformation."""

    def _repr_indent(self, indent: int = 0) -> str:
        if isinstance(self, NgffIdentity):
            return f"{self._repr_transformation_signature(indent)}"
        else:
            return f"{self._repr_transformation_signature(indent)}\n{self._repr_transformation_description(indent + 1)}"

    def __repr__(self) -> str:
        return self._repr_indent(0)

    @classmethod
    @abstractmethod
    def _from_dict(cls, d: Transformation_t) -> "NgffBaseTransformation":
        pass

    @classmethod
    def from_dict(cls, d: Transformation_t) -> "NgffBaseTransformation":
        """
        Initialize a transformation from the Python dict of its json representation.

        Parameters
        ----------
        d
            the Python dict representing the transformation.
        """
        # d = MappingProxyType(d)
        type = d["type"]
        # MappingProxyType is readonly
        transformation = NGFF_TRANSFORMATIONS[type]._from_dict(d)
        if "input" in d:
            input_coordinate_system = d["input"]
            if isinstance(input_coordinate_system, dict):
                input_coordinate_system = NgffCoordinateSystem.from_dict(input_coordinate_system)
            transformation.input_coordinate_system = input_coordinate_system
        if "output" in d:
            output_coordinate_system = d["output"]
            if isinstance(output_coordinate_system, dict):
                output_coordinate_system = NgffCoordinateSystem.from_dict(output_coordinate_system)
            transformation.output_coordinate_system = output_coordinate_system
        return transformation

    @abstractmethod
    def to_dict(self) -> Transformation_t:
        """
        Return the Python dict representation of the transformation.
        """

    def _update_dict_with_input_output_cs(self, d: Transformation_t) -> None:
        """
        Update a transformation dictionary with the transformation's input and output coordinate systems.

        Parameters
        ----------
        d
            The dictionary to be updated.

        Notes
        -------
        Use of this helper function.
        A transformation is saved to disk as a json string. When loaded this is represented by a dictionary.
        We allow transformations to be defined even when the input and output coordinate systems are not
        specified (for instance the transformations inside a Sequence transformation don't necessarily
        specity the input and output coordinate system). This helper function puts the input and output
        coordinate system back into the dict.
        """
        if self.input_coordinate_system is not None:
            d["input"] = self.input_coordinate_system
            if isinstance(d["input"], NgffCoordinateSystem):
                d["input"] = d["input"].to_dict()
        if self.output_coordinate_system is not None:
            d["output"] = self.output_coordinate_system
            if isinstance(d["output"], NgffCoordinateSystem):
                d["output"] = d["output"].to_dict()

    @abstractmethod
    def inverse(self) -> "NgffBaseTransformation":
        """Return the inverse of the transformation."""

    @abstractmethod
    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get the input and output axes of the coordinate systems specified for the transformation, and check if they are
        compatible with the transformation.
        """

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        """
        Transform points (coordinates).

        Notes
        -------
        This function will check if the dimensionality of the input and output coordinate systems of the
        transformation are compatible with the given points.
        """

    @abstractmethod
    def to_affine(self) -> "NgffAffine":
        """Convert the transformation to an affine transformation, whenever the conversion can be made."""

    def _validate_transform_points_shapes(self, input_size: int, points_shape: tuple[int, ...]) -> None:
        """
        Validate if the shape of the points (coordinats to be transformed) are consistent with the input size of the transformation.

        Parameters
        ----------
        input_size
            The input size.
        points_shape
            The points' shape
        """
        if len(points_shape) != 2 or points_shape[1] != input_size:
            raise ValueError(
                f"points must be a tensor of shape (n, d), where n is the number of points and d is the "
                f"the number of spatial dimensions. Points shape: {points_shape}, input size: {input_size}"
            )

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformation: "NgffBaseTransformation") -> "NgffBaseTransformation":
        """
        Compose the transfomation object with another transformation

        Parameters
        ----------
        transformation
            The transformation to compose with.

        Returns
        -------
        The compoesed transformation.

        Notes
        -------
        Self is applied first, then the transformation passed as argument.
        """
        return NgffSequence([self, transformation])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NgffBaseTransformation):
            raise NotImplementedError("Cannot compare NgffBaseTransformation with other types")
        return self.to_dict() == other.to_dict()

    def _get_axes_from_coordinate_systems(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get the input and output coordinate systems' axes"""
        if not isinstance(self.input_coordinate_system, NgffCoordinateSystem):
            raise ValueError("Input coordinate system not specified")
        if not isinstance(self.output_coordinate_system, NgffCoordinateSystem):
            raise ValueError("Output coordinate system not specified")
        input_axes = self.input_coordinate_system.axes_names
        output_axes = self.output_coordinate_system.axes_names
        return input_axes, output_axes

    @staticmethod
    def _parse_list_into_array(array: Union[list[Number], list[list[Number]], ArrayLike]) -> ArrayLike:
        """Parse a list or numbers, or a list of lists of numbers, into a float numpy array."""
        if isinstance(array, list):
            array = np.array(array)
        if array.dtype != float:
            array = array.astype(float)
        return array


NGFF_TRANSFORMATIONS: dict[str, type[NgffBaseTransformation]] = {}


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


class NgffAffine(NgffBaseTransformation):
    """The Affine transformation from the NGFF specification."""

    def __init__(
        self,
        affine: Union[ArrayLike, list[list[Number]]],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffAffine object.
        Parameters
        ----------
        affine
            A list of lists of numbers or a matrix specifying the affine transformation.
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
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

    def inverse(self) -> NgffBaseTransformation:
        inv = np.linalg.inv(self.affine)
        return NgffAffine(
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

    def to_affine(self) -> "NgffAffine":
        return NgffAffine(
            self.affine,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )

    @classmethod
    def _affine_matrix_from_input_and_output_axes(
        cls, input_axes: tuple[str, ...], output_axes: tuple[str, ...]
    ) -> ArrayLike:
        """
        computes a permutation matrix to match the input and output axes.

        Parameters
        ----------
        input_axes
            the input axes.
        output_axes
            the output axes.
        """
        from spatialdata.models import C, X, Y, Z

        assert all(ax in (X, Y, Z, C) for ax in input_axes)
        assert all(ax in (X, Y, Z, C) for ax in output_axes)
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
        input_coordinate_system: NgffCoordinateSystem,
        output_coordinate_system: NgffCoordinateSystem,
    ) -> "NgffAffine":
        input_axes = input_coordinate_system.axes_names
        output_axes = output_coordinate_system.axes_names
        m = cls._affine_matrix_from_input_and_output_axes(input_axes, output_axes)
        return cls(
            affine=m, input_coordinate_system=input_coordinate_system, output_coordinate_system=output_coordinate_system
        )


class NgffIdentity(NgffBaseTransformation):
    """The Identity transformation from the NGFF specification."""

    def __init__(
        self,
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffIdentity object.

        Parameters
        ----------
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
        """
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

    def inverse(self) -> NgffBaseTransformation:
        return NgffIdentity(
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

    def to_affine(self) -> NgffAffine:
        input_axes, _ = self._get_and_validate_axes()
        return NgffAffine(
            np.eye(len(input_axes) + 1),
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


# # maybe this transformation will not make it to the final specs, waiting before implementing this
# class MapIndex(NgffBaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
class NgffMapAxis(NgffBaseTransformation):
    """The MapAxis transformation from the NGFF specification."""

    def __init__(
        self,
        map_axis: dict[str, str],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffMapAxis object.
        Parameters
        ----------
        map_axis
            A dictionary mapping the output axes (keys) to the input axes (values).
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
        """
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
            s += f"{self._indent(indent)}{k} <- {v}\n"
        s = s[:-1]
        return s

    def inverse(self) -> NgffBaseTransformation:
        if len(self.map_axis.keys()) != len(set(self.map_axis.values())):
            raise ValueError("Cannot invert a map axis transformation with different number of input and output axes")
        else:
            return NgffMapAxis(
                {v: k for k, v in self.map_axis.items()},
                input_coordinate_system=self.output_coordinate_system,
                output_coordinate_system=self.input_coordinate_system,
            )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if not set(input_axes).issuperset(set(self.map_axis.values())):
            raise ValueError(
                "Each value of the dict specifying a NgffMapAxis transformation must be an axis of the input "
                "coordinate system"
            )
        if set(output_axes) != set(self.map_axis.keys()):
            raise ValueError(
                "The set of output axes must be the same as the set of keys the dict specifying a "
                "NgffMapAxis transformation"
            )
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        new_indices = [input_axes.index(self.map_axis[ax]) for ax in output_axes]
        assert len(new_indices) == len(output_axes)
        mapped = points[:, new_indices]
        assert isinstance(mapped, np.ndarray)
        return mapped

    def to_affine(self) -> NgffAffine:
        input_axes, output_axes = self._get_and_validate_axes()
        matrix: ArrayLike = np.zeros((len(output_axes) + 1, len(input_axes) + 1), dtype=float)
        matrix[-1, -1] = 1
        for i, des_axis in enumerate(output_axes):
            for j, src_axis in enumerate(input_axes):
                if src_axis == self.map_axis[des_axis]:
                    matrix[i, j] = 1
        affine = NgffAffine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )
        return affine


class NgffTranslation(NgffBaseTransformation):
    """The Translation transformation from the NGFF specification."""

    def __init__(
        self,
        translation: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffTranslation object.
        Parameters
        ----------
        translation
            A list of numbers or a vector specifying the translation along each axis.
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
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

    def inverse(self) -> NgffBaseTransformation:
        return NgffTranslation(
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

    def to_affine(self, ndims_input: Optional[int] = None, ndims_output: Optional[int] = None) -> NgffAffine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, -1] = self.translation
        return NgffAffine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class NgffScale(NgffBaseTransformation):
    """The Scale transformation from the NGFF specification."""

    def __init__(
        self,
        scale: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffScale object.
        Parameters
        ----------
        scale
            A list of numbers or a vector specifying the scale along each axis.
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
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

    def inverse(self) -> NgffBaseTransformation:
        new_scale = np.zeros_like(self.scale)
        new_scale[np.nonzero(self.scale)] = 1 / self.scale[np.nonzero(self.scale)]
        return NgffScale(
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

    def to_affine(self) -> NgffAffine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, :-1] = np.diag(self.scale)
        return NgffAffine(
            matrix,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class NgffRotation(NgffBaseTransformation):
    """The Rotation transformation from the NGFF specification."""

    def __init__(
        self,
        rotation: Union[ArrayLike, list[Number]],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffRotation object.
        Parameters
        ----------
        rotation
            A list of numbers specifying the rotation transformation.
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
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

    def inverse(self) -> NgffBaseTransformation:
        return NgffRotation(
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

    def to_affine(self) -> NgffAffine:
        m = np.eye(len(self.rotation) + 1)
        m[:-1, :-1] = self.rotation
        return NgffAffine(
            m,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


class NgffSequence(NgffBaseTransformation):
    """The Sequence transformation from the NGFF specification."""

    def __init__(
        self,
        transformations: list[NgffBaseTransformation],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the NgffSequence object.

        Parameters
        ----------
        transformations
            The transformations which compose the sequence.
        input_coordinate_system
            Input coordinate system of the transformation.
        output_coordinate_system
            Output coordinate system of the transformation.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        # we can decide to treat an empty sequence as an NgffIdentity if we need to
        assert len(transformations) > 0
        self.transformations = transformations

        if (cs := self.transformations[0].input_coordinate_system) is not None:
            if self.input_coordinate_system is not None:
                pass
                # if cs != input_coordinate_system:
                #     raise ValueError(
                #         "Input coordinate system do not match the input coordinate system of the first transformation"
                #     )
            else:
                self.input_coordinate_system = cs
        if (cs := self.transformations[-1].output_coordinate_system) is not None:
            if self.output_coordinate_system is not None:
                pass
                # if cs != self.output_coordinate_system:
                #     raise ValueError(
                #         "Output coordinate system do not match the output coordinate system of the last transformation"
                #     )
            else:
                self.output_coordinate_system = cs

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls([NgffBaseTransformation.from_dict(t) for t in d["transformations"]])

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

    def inverse(self) -> NgffBaseTransformation:
        return NgffSequence(
            [t.inverse() for t in reversed(self.transformations)],
            input_coordinate_system=self.output_coordinate_system,
            output_coordinate_system=self.input_coordinate_system,
        )

    def _get_and_validate_axes(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        return input_axes, output_axes

    @staticmethod
    def _inferring_cs_infer_output_coordinate_system(
        t: NgffBaseTransformation,
    ) -> Optional[NgffCoordinateSystem]:
        """
        One of a series of helper functions to infer the input and output axes of the transformations composing a Sequence NGFF transformation.

        The NGFF specification don't require all the input and output coordinate systems of a sequence transformation to be specified.
        To make the handling of Sequence transformations easier, this function populate such values. When not needed anymore (e.g. right after
        applying the transformation) these set of functions remove the added coordinate systems, thus leaving the initial object invariate.
        """
        assert isinstance(t.input_coordinate_system, NgffCoordinateSystem)
        if isinstance(t, NgffAffine):
            return None
        elif isinstance(t, NgffTranslation | NgffScale | NgffRotation | NgffIdentity):
            return t.input_coordinate_system
        elif isinstance(t, NgffMapAxis):
            return None
        elif isinstance(t, NgffSequence):
            latest_output_cs = t.input_coordinate_system
            for tt in t.transformations:
                (
                    latest_output_cs,
                    input_cs,
                    output_cs,
                ) = NgffSequence._inferring_cs_pre_action(tt, latest_output_cs)
                NgffSequence._inferring_cs_post_action(tt, input_cs, output_cs)
            return latest_output_cs
        else:
            return None

    @staticmethod
    def _inferring_cs_pre_action(
        t: NgffBaseTransformation, latest_output_cs: NgffCoordinateSystem
    ) -> tuple[NgffCoordinateSystem, Optional[NgffCoordinateSystem], Optional[NgffCoordinateSystem]]:
        """See _inferring_cs_infer_output_coordinate_system()"""
        input_cs = t.input_coordinate_system
        if input_cs is None:
            t.input_coordinate_system = latest_output_cs
        else:
            assert isinstance(input_cs, NgffCoordinateSystem)
            assert input_cs == latest_output_cs
        output_cs = t.output_coordinate_system
        expected_output_cs = NgffSequence._inferring_cs_infer_output_coordinate_system(t)
        if output_cs is None:
            if expected_output_cs is None:
                raise ValueError(
                    f"Cannot infer output coordinate system for {t}, this could happen for instance if "
                    f"passing an NgffAffine transformation as a component of a NgffSequence transformation "
                    f"without specifying the input and output coordinate system for the NgffAffine "
                    f"transformation."
                )
            t.output_coordinate_system = expected_output_cs
        else:
            assert isinstance(output_cs, NgffCoordinateSystem)
            # if it is not possible to infer the output, like for NgffAffine, we skip this check
            if expected_output_cs is not None:
                assert t.output_coordinate_system == expected_output_cs
        new_latest_output_cs = t.output_coordinate_system
        assert isinstance(new_latest_output_cs, NgffCoordinateSystem)
        return new_latest_output_cs, input_cs, output_cs

    @staticmethod
    def _inferring_cs_post_action(
        t: NgffBaseTransformation,
        input_cs: Optional[NgffCoordinateSystem],
        output_cs: Optional[NgffCoordinateSystem],
    ) -> None:
        """See _inferring_cs_infer_output_coordinate_system()"""
        # if the transformation t was passed without input or output coordinate systems (and so we had to infer
        # them), we now restore the original state of the transformation
        if input_cs is None:
            t.input_coordinate_system = None
        if output_cs is None:
            t.output_coordinate_system = None

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        # the specs allow to compose transformations without specifying the input and output coordinate systems of
        # every transformation. Since in order to apply a transformation we need to know the input and output coordinate
        # systems, we allow for on-the-fly computation by inferring this in real-time. The inferred information is
        # then removed, so to leave the object as it was found before. To be precise, some transformation don't
        # strictly require the input and output coordinate system to be specified when applying the transformation,
        # because they "trust" the input and output to be correct (for example NgffIdentity). But requiring the
        # coordinate systems to be specified we can enforce a check that could help catch errors, so the extra
        # complexity in these NgffSequence class will be rewarded in the long term.
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        assert type(self.input_coordinate_system) is NgffCoordinateSystem
        latest_output_cs: NgffCoordinateSystem = self.input_coordinate_system
        for t in self.transformations:
            latest_output_cs, input_cs, output_cs = NgffSequence._inferring_cs_pre_action(t, latest_output_cs)
            points = t.transform_points(points)
            NgffSequence._inferring_cs_post_action(t, input_cs, output_cs)
        if output_axes != latest_output_cs.axes_names:
            raise ValueError(
                "Inferred output axes of the sequence of transformations do not match the expected output "
                "coordinate system."
            )
        return points

    def to_affine(self) -> NgffAffine:
        # the same comment on the coordinate systems of the various transformations, made on the transform_points()
        # method, applies also here
        input_axes, output_axes = self._get_and_validate_axes()
        composed = np.eye(len(input_axes) + 1)
        assert type(self.input_coordinate_system) is NgffCoordinateSystem
        latest_output_cs: NgffCoordinateSystem = self.input_coordinate_system
        for t in self.transformations:
            latest_output_cs, input_cs, output_cs = NgffSequence._inferring_cs_pre_action(t, latest_output_cs)
            a = t.to_affine()
            composed = a.affine @ composed
            NgffSequence._inferring_cs_post_action(t, input_cs, output_cs)
        if output_axes != latest_output_cs.axes_names:
            raise ValueError(
                "Inferred output axes of the sequence of transformations do not match the expected output "
                "coordinate system."
            )
        return NgffAffine(
            composed,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


# class Displacements(NgffBaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
#
# class Coordinates(NgffBaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
#
# class InverseOf(NgffBaseTransformation):
#     def __init__(self, transformation: Union[Dict[str, Any], NgffBaseTransformation]) -> None:
#         if isinstance(transformation, NgffBaseTransformation):
#             self.transformation = transformation
#         else:
#             self.transformation = NgffBaseTransformation.from_dict(transformation)
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
#     def inverse(self) -> NgffBaseTransformation:
#         return self.transformation
#
#
# class Bijection(NgffBaseTransformation):
#     def __init__(
#         self, forward: Union[Dict[str, Any], NgffBaseTransformation], inverse: Union[Dict[str, Any], NgffBaseTransformation]
#     ) -> None:
#         if isinstance(forward, NgffBaseTransformation):
#             self.forward = forward
#         else:
#             self.forward = NgffBaseTransformation.from_dict(forward)
#
#         if isinstance(inverse, NgffBaseTransformation):
#             self._inverse = inverse
#         else:
#             self._inverse = NgffBaseTransformation.from_dict(inverse)
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
#     def inverse(self) -> NgffBaseTransformation:
#         return self._inverse
class NgffByDimension(NgffBaseTransformation):
    """The ByDimension transformation from the NGFF specification."""

    def __init__(
        self,
        transformations: list[NgffBaseTransformation],
        input_coordinate_system: Optional[NgffCoordinateSystem] = None,
        output_coordinate_system: Optional[NgffCoordinateSystem] = None,
    ) -> None:
        """
        Init the ByDimension object.

        Parameters
        ----------
        transformations
            A list of transformations, whose set of output coordinate systems partition the output coordinate system of
            the ByDimension transformation.
        input_coordinate_system
            The input coordinate system of the transformation.
        output_coordinate_system
            The output coordinate system of the transformation.
        """
        super().__init__(input_coordinate_system, output_coordinate_system)
        assert len(transformations) > 0
        self.transformations = transformations

    @classmethod
    def _from_dict(cls, d: Transformation_t) -> Self:  # type: ignore[valid-type]
        return cls([NgffBaseTransformation.from_dict(t) for t in d["transformations"]])

    def to_dict(self) -> Transformation_t:
        d = {
            "type": "byDimension",
            "transformations": [t.to_dict() for t in self.transformations],
        }
        self._update_dict_with_input_output_cs(d)
        return d

    # same code as in NgffSequence
    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = ""
        for t in self.transformations:
            s += f"{t._repr_indent(indent=indent)}\n"
        s = s[:-1]
        return s

    def inverse(self) -> NgffBaseTransformation:
        inverse_transformations = [t.inverse() for t in self.transformations]
        return NgffByDimension(
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
            assert isinstance(t.input_coordinate_system, NgffCoordinateSystem)
            assert isinstance(t.output_coordinate_system, NgffCoordinateSystem)
            for ax in t.input_coordinate_system.axes_names:
                assert ax in input_axes
            for ax in t.output_coordinate_system.axes_names:
                # assert ax not in defined_output_axes
                if ax not in defined_output_axes:
                    defined_output_axes.add(ax)
                else:
                    raise ValueError(f"Output axis {ax} is defined more than once")
        assert defined_output_axes.issuperset(set(output_axes))
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        output_columns: dict[str, ArrayLike] = {}
        for t in self.transformations:
            assert isinstance(t.input_coordinate_system, NgffCoordinateSystem)
            assert isinstance(t.output_coordinate_system, NgffCoordinateSystem)
            input_columns = [points[:, input_axes.index(ax)] for ax in t.input_coordinate_system.axes_names]
            input_columns_stacked: ArrayLike = np.stack(input_columns, axis=1)
            output_columns_t = t.transform_points(input_columns_stacked)
            for ax, col in zip(t.output_coordinate_system.axes_names, output_columns_t.T, strict=True):
                output_columns[ax] = col
        output: ArrayLike = np.stack([output_columns[ax] for ax in output_axes], axis=1)
        return output

    def to_affine(self) -> NgffAffine:
        input_axes, output_axes = self._get_and_validate_axes()
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        m[-1, -1] = 1
        for t in self.transformations:
            assert isinstance(t.input_coordinate_system, NgffCoordinateSystem)
            assert isinstance(t.output_coordinate_system, NgffCoordinateSystem)
            t_affine = t.to_affine()
            target_output_indices = [
                output_axes.index(ax) for ax in t.output_coordinate_system.axes_names if ax in output_axes
            ]
            source_output_indices = [
                t.output_coordinate_system.axes_names.index(ax)
                for ax in t.output_coordinate_system.axes_names
                if ax in output_axes
            ]
            target_input_indices = [input_axes.index(ax) for ax in t.input_coordinate_system.axes_names] + [-1]
            m[np.ix_(target_output_indices, target_input_indices)] = t_affine.affine[source_output_indices, :]
        return NgffAffine(
            m,
            input_coordinate_system=self.input_coordinate_system,
            output_coordinate_system=self.output_coordinate_system,
        )


NGFF_TRANSFORMATIONS["identity"] = NgffIdentity
NGFF_TRANSFORMATIONS["mapAxis"] = NgffMapAxis
NGFF_TRANSFORMATIONS["translation"] = NgffTranslation
NGFF_TRANSFORMATIONS["scale"] = NgffScale
NGFF_TRANSFORMATIONS["affine"] = NgffAffine
NGFF_TRANSFORMATIONS["rotation"] = NgffRotation
NGFF_TRANSFORMATIONS["sequence"] = NgffSequence
NGFF_TRANSFORMATIONS["byDimension"] = NgffByDimension
