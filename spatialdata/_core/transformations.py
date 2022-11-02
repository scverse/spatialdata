from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

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


Transformation_t = Dict[str, Union[str, List[int], List[str], Dict[str, Any]]]


class BaseTransformation(ABC):
    """Base class for all transformations."""

    # the general json description of a transformation contains just the name of the input and output space,
    # (coordinate systems are specified outside the transformation), and sometimes the name is even not specified (like
    # for transformation that define a "Sequence" transformation). For this reason the following two variables can be
    # None or strings. Anyway, in order to be able to apply a transformation to a DataArray, we need to know the name of
    # the input and output axes (information contained in the CoordinateSystem object). Therefore, the following
    # variables will be populated with CoordinateSystem objects when both the coordinate_system and the transformation
    # are known.
    # Example: as a user you have an Image (cyx) and a Point (xy) elements, and you want to contruct a SpatialData
    # object containing the two of them. You also want to apply a Scale transformation to the Image. You can simply
    # assign the Scale transformation to the Image element, and the SpatialData object will take care of assiging to
    # "_input_coordinate_system" the intrinsitc coordinate system of the Image element,
    # and to "_output_coordinate_system" the global coordinate system (cyx) that will be created when calling the
    # SpatialData constructor
    _input_coordinate_system: Optional[Union[str, CoordinateSystem]] = None
    _output_coordinate_system: Optional[Union[str, CoordinateSystem]] = None

    @property
    def input_coordinate_system(self) -> Optional[str]:
        return self._input_coordinate_system

    @input_coordinate_system.setter
    def input_coordinate_system(self, coordinate_system: Optional[Union[str, CoordinateSystem]]) -> None:
        self._input_coordinate_system = coordinate_system

    @property
    def output_coordinate_system(self) -> Optional[str]:
        return self._output_coordinate_system

    @output_coordinate_system.setter
    def output_coordinate_system(self, coordinate_system: Optional[Union[str, CoordinateSystem]]) -> None:
        self._output_coordinate_system = coordinate_system

    @classmethod
    def from_dict(cls, d: Transformation_t) -> BaseTransformation:
        kw = d.copy()
        type = kw["type"]
        my_class: Type[BaseTransformation]
        if type == "identity":
            my_class = Identity
        # elif type == "mapIndex":
        #     my_class = MapIndex  # type: ignore
        elif type == "mapAxis":
            my_class = MapAxis
            kw["map_axis"] = kw["mapAxis"]
            del kw["mapAxis"]
        elif type == "translation":
            my_class = Translation
        elif type == "scale":
            my_class = Scale
        elif type == "affine":
            my_class = Affine
        elif type == "rotation":
            my_class = Rotation
        elif type == "sequence":
            my_class = Sequence
        # elif type == "displacements":
        #     my_class = Displacements  # type: ignore
        # elif type == "coordinates":
        #     my_class = Coordinates  # type: ignore
        # elif type == "vectorField":
        #     my_class = VectorField  # type: ignore
        # elif type == "inverseOf":
        #     my_class = InverseOf
        # elif type == "bijection":
        #     my_class = Bijection
        # elif type == "byDimension":
        #     my_class = ByDimension  # type: ignore
        else:
            raise ValueError(f"Unknown transformation type: {type}")
        del kw["type"]
        input_coordinate_system = None
        if "input" in kw:
            input_coordinate_system = kw["input"]
            del kw["input"]
        output_coordinate_system = None
        if "output" in kw:
            output_coordinate_system = kw["output"]
            del kw["output"]
        transformation = my_class(**kw)
        transformation.input_coordinate_system = input_coordinate_system
        transformation.output_coordinate_system = output_coordinate_system
        return transformation

    @abstractmethod
    def to_dict(self) -> Transformation_t:
        pass

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    @abstractmethod
    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        pass

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def to_affine(self) -> Affine:
        pass

    def _validate_transform_points_shapes(self, input_size: int, points_shape: Tuple[int]):
        if len(points_shape) != 2 or points_shape[1] != input_size:
            raise ValueError(
                f"points must be a tensor of shape (n, d), where n is the number of points and d is the "
                f"the number of spatial dimensions. Points shape: {points_shape}, input size: {input_size}"
            )

    @classmethod
    def from_json(cls, data: Union[str, bytes]) -> BaseTransformation:
        d = json.loads(data)
        return get_transformation_from_dict(d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformation: BaseTransformation) -> BaseTransformation:
        return Sequence([self, transformation])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseTransformation):
            return False
        return self.to_dict() == other.to_dict()

    def _get_axes_from_coordinate_systems(self) -> Tuple[Tuple[str], Tuple[str]]:
        if not isinstance(self._input_coordinate_system, CoordinateSystem):
            raise ValueError(f"Input coordinate system not specified")
        if not isinstance(self.output_coordinate_system, CoordinateSystem):
            raise ValueError(f"Output coordinate system not specified")
        input_axes = self.input_coordinate_system.axes_names
        output_axes = self.output_coordinate_system.axes_names
        return input_axes, output_axes


def get_transformation_from_json(s: str) -> BaseTransformation:
    return BaseTransformation.from_json(s)


def get_transformation_from_dict(d: Transformation_t) -> BaseTransformation:
    return BaseTransformation.from_dict(d)


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
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "identity",
        }

    def inverse(self) -> BaseTransformation:
        t = Identity()
        t.input_coordinate_system = self.output_coordinate_system
        t.output_coordinate_system = self.input_coordinate_system
        return t

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError(f"Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points

    def to_affine(self) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        affine = Affine(np.eye(len(input_axes) + 1))
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


# # maybe this transformation will not make it to the final specs, waiting before implementing this
# class MapIndex(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()


class MapAxis(BaseTransformation):
    def __init__(self, map_axis: Dict[str, str]) -> None:
        self.map_axis = map_axis

    def to_dict(self) -> Transformation_t:
        return {
            "type": "mapAxis",
            "mapAxis": self.map_axis,
        }

    def inverse(self) -> BaseTransformation:
        if len(self.map_axis.keys()) != len(set(self.map_axis.values())):
            raise ValueError("Cannot invert a map axis transformation with different number of input and output axes")
        else:
            t = MapAxis({v: k for k, v in self.map_axis.items()})
            t.input_coordinate_system = self.output_coordinate_system
            t.output_coordinate_system = self.input_coordinate_system
            return t

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if set(input_axes) != set(self.map_axis.keys()):
            raise ValueError(f"The set of input axes must be the same as the set of keys of mapAxis")
        if set(output_axes) != set(self.map_axis.values()):
            raise ValueError(f"The set of output axes must be the same as the set of values of mapAxis")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        raise NotImplementedError()

    def to_affine(self) -> Affine:
        input_axes, output_axes = self._get_and_validate_axes()
        matrix = np.zeros((len(output_axes) + 1, len(input_axes) + 1), dtype=float)
        matrix[-1, -1] = 1
        for i, des_axis in enumerate(output_axes):
            for j, src_axis in enumerate(input_axes):
                if des_axis == self.map_axis[src_axis]:
                    matrix[i, j] = 1
        affine = Affine(matrix)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


class Translation(BaseTransformation):
    def __init__(self, translation: ArrayLike) -> None:
        """
        class for storing translation transformations.
        """
        self.translation = translation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "translation",
            "translation": self.translation.tolist(),
        }

    def inverse(self) -> BaseTransformation:
        t = Translation(-self.translation)
        t.input_coordinate_system = self.output_coordinate_system
        t.output_coordinate_system = self.input_coordinate_system
        return t

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError(f"Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points + self.translation

    def to_affine(self, ndims_input: Optional[int] = None, ndims_output: Optional[int] = None) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, -1] = self.translation
        affine = Affine(matrix)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


class Scale(BaseTransformation):
    def __init__(self, scale: ArrayLike = None) -> None:
        """
        class for storing scale transformations.
        """
        self.scale = scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "scale",
            "scale": self.scale.tolist(),
        }

    def inverse(self) -> BaseTransformation:
        new_scale = np.zeros_like(self.scale)
        new_scale[np.nonzero(self.scale)] = 1 / self.scale[np.nonzero(self.scale)]
        t = Scale(new_scale)
        t.input_coordinate_system = self.output_coordinate_system
        t.output_coordinate_system = self.input_coordinate_system
        return t

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError(f"Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return points * self.scale

    def to_affine(self) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        matrix = np.eye(len(input_axes) + 1)
        matrix[:-1, :-1] = np.diag(self.scale)
        affine = Affine(matrix)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


class Affine(BaseTransformation):
    def __init__(self, affine: ArrayLike) -> None:
        """
        class for storing scale transformations.
        """
        self.affine = affine

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "affine",
            "affine": self.affine[:-1, :].tolist(),
        }

    def inverse(self) -> BaseTransformation:
        inv = np.lingalg.inv(self.affine)
        t = Affine(inv)
        t.input_coordinate_system = self.output_coordinate_system
        t.output_coordinate_system = self.input_coordinate_system
        return t

        # old code, manually inverting a 2d (3x3) affine matrix
        # a = self.affine[0, 0]
        # b = self.affine[0, 1]
        # m = self.affine[0, 2]
        # c = self.affine[1, 0]
        # d = self.affine[1, 1]
        # n = self.affine[1, 2]
        # det = a * d - b * c
        # closed_form = np.array([[d, -c, 0], [-b, a, 0], [b * n - d * m, c * m - a * n, det]])
        # return Affine(affine=closed_form)

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        return q[: len(output_axes), :].T  # type: ignore[no-any-return]

    def to_affine(self) -> Affine:
        affine = Affine(self.affine)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


class Rotation(BaseTransformation):
    def __init__(self, rotation: ArrayLike) -> None:
        """
        class for storing scale transformations.
        """
        self.rotation = rotation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "rotation",
            "rotation": self.rotation.ravel().tolist(),
        }

    def inverse(self) -> BaseTransformation:
        t = Rotation(self.rotation.T)
        t.input_coordinate_system = self.output_coordinate_system
        t.output_coordinate_system = self.input_coordinate_system
        return t

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError(f"Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes, _ = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        return (self.rotation @ points.T).T

    def to_affine(self) -> Affine:
        m = np.eye(len(self.rotation) + 1)
        m[:-1, :-1] = self.rotation
        affine = Affine(m)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


class Sequence(BaseTransformation):
    def __init__(self, transformations: List[BaseTransformation]) -> None:
        assert len(transformations) > 0
        self.transformations = transformations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "sequence",
            "transformations": [t.to_dict() for t in self.transformations],
        }

    def inverse(self) -> BaseTransformation:
        inv = Sequence([t.inverse() for t in reversed(self.transformations)])
        inv.input_coordinate_system = self.output_coordinate_system
        inv.output_coordinate_system = self.input_coordinate_system
        return inv

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        return input_axes, output_axes

    @staticmethod
    def _inferring_cs_infer_output_coordinate_system(t: BaseTransformation) -> Optional[CoordinateSystem]:
        assert isinstance(t.input_coordinate_system, CoordinateSystem)
        if isinstance(t, Affine):
            return None
        elif isinstance(t, Translation) or isinstance(t, Scale) or isinstance(t, Rotation) or isinstance(t, Identity):
            return t.input_coordinate_system
        elif isinstance(t, MapAxis):
            raise NotImplementedError()
        elif isinstance(t, Sequence):
            previous_output = t.input_coordinate_system
            for t in t.transformations:
                previous_output, input_cs, output_cs = Sequence._inferring_cs_pre_action(t, previous_output)
                current_output_axes = Sequence._inferring_cs_post_action(t, input_cs, output_cs)
            return current_output_axes
        else:
            return None

    @staticmethod
    def _inferring_cs_pre_action(
        t: BaseTransformation, previous_output: CoordinateSystem
    ) -> Tuple[CoordinateSystem, Optional[CoordinateSystem], Optional[CoordinateSystem]]:
        input_cs = t.input_coordinate_system
        if input_cs is None:
            t.input_coordinate_system = previous_output
        elif isinstance(input_cs, str):
            raise ValueError(
                f"Input coordinate system for {t} is a string, not a CoordinateSystem. It should be "
                f"replaced by the CoordinateSystem named after the string before calling this function."
            )
        else:
            assert isinstance(input_cs, CoordinateSystem)
            assert input_cs == previous_output
        output_cs = t.output_coordinate_system
        expected_output_cs = Sequence._inferring_cs_infer_output_coordinate_system(t)
        if output_cs is None:
            t.output_coordinate_system = expected_output_cs
        elif isinstance(output_cs, str):
            raise ValueError(
                f"Output coordinate system for {t} is a string, not a CoordinateSystem. It should be "
                f"replaced by the CoordinateSystem named after the string before calling this function."
            )
        else:
            assert isinstance(output_cs, CoordinateSystem)
            # if it is not possible to infer the output, like for Affine, we skip this check
            if expected_output_cs is not None:
                assert t.output_coordinate_system == expected_output_cs
        previous_output = t.output_coordinate_system
        return previous_output, input_cs, output_cs

    @staticmethod
    def _inferring_cs_post_action(
        t: BaseTransformation, input_cs: Optional[CoordinateSystem], output_cs: Optional[CoordinateSystem]
    ) -> Tuple[str]:
        assert isinstance(t.output_coordinate_system, CoordinateSystem)
        output_axes = t.output_coordinate_system.axes_names
        if input_cs is None:
            t.input_coordinate_system = None
        if output_cs is None:
            t.output_coordinate_system = None
        return output_axes

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        # the specs allow to compose transformations without specifying the input and output coordinate systems of
        # every transformation. Since in order to apply a transformation we need to know the input and output coordinate
        # systems, we allow for on-the-fly computation by inferring this in real-time. The inferred information is
        # then removed, so to leave the object as it was found before
        input_axes, output_axes = self._get_and_validate_axes()
        self._validate_transform_points_shapes(len(input_axes), points.shape)
        previous_output = input_axes
        current_output_axes: Optional[Tuple[str]] = None
        for t in self.transformations:
            previous_output, input_cs, output_cs = Sequence._inferring_cs_pre_action(t, previous_output)
            points = t.transform_points(points)
            current_output_axes = Sequence._inferring_cs_post_action(t, input_cs, output_cs)
        # as long as len(self.transformations) > 0, current_output_axes is not None
        assert current_output_axes is not None
        assert output_axes == current_output_axes
        return points

    def to_affine(self) -> Affine:
        # the same comment on the coordinate systems of the various transformations, made on the transform_points()
        # method, applies also here
        input_axes, output_axes = self._get_and_validate_axes()
        composed = np.eye(len(input_axes) + 1)
        previous_output = input_axes
        for t in self.transformations:
            previous_output, input_cs, output_cs = Sequence._inferring_cs_pre_action(t, previous_output)
            a = t.to_affine()
            composed = a.affine @ composed
            current_output_axes = Sequence._inferring_cs_post_action(t, input_cs, output_cs)
        assert output_axes == current_output_axes
        affine = Affine(composed)
        affine.input_coordinate_system = self.input_coordinate_system
        affine.output_coordinate_system = self.output_coordinate_system
        return affine


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
#             self.transformation = get_transformation_from_dict(transformation)
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
#     def to_dict(self) -> Dict[str, Any]:
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
#             self.forward = get_transformation_from_dict(forward)
#
#         if isinstance(inverse, BaseTransformation):
#             self._inverse = inverse
#         else:
#             self._inverse = get_transformation_from_dict(inverse)
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
#     def to_dict(self) -> Dict[str, Any]:
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
#
#
# class ByDimension(BaseTransformation):
#     def __init__(self) -> None:
#         raise NotImplementedError()
#
#     # @property
#     # def ndim(self) -> Optional[int]:
#     #     return self._ndim
#
