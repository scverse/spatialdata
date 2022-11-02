from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from spatialdata._core.coordinate_system import CoordinateSystem

import numpy as np

from xarray import DataArray

__all__ = [
    "BaseTransformation",
    "Identity",
    "MapIndex",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "Sequence",
    "Displacements",
    "Coordinates",
    "VectorField",
    "InverseOf",
    "Bijection",
    "ByDimension",
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
    _input_coordinate_system: Optional[str, CoordinateSystem] = None
    _output_coordinate_system: Optional[str, CoordinateSystem] = None

    @property
    def input_coordinate_system(self) -> Optional[str]:
        return self._input_coordinate_system

    @property
    def output_coordinate_system(self) -> Optional[str]:
        return self._output_coordinate_system

    @classmethod
    def from_dict(cls, d: Transformation_t) -> BaseTransformation:
        kw = d.copy()
        type = kw["type"]
        my_class: Type[BaseTransformation]
        if type == "identity":
            my_class = Identity
        elif type == "mapIndex":
            my_class = MapIndex  # type: ignore
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
        elif type == "displacements":
            my_class = Displacements  # type: ignore
        elif type == "coordinates":
            my_class = Coordinates  # type: ignore
        elif type == "vectorField":
            my_class = VectorField  # type: ignore
        elif type == "inverseOf":
            my_class = InverseOf
        elif type == "bijection":
            my_class = Bijection
        elif type == "byDimension":
            my_class = ByDimension  # type: ignore
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
        transformation._input_coordinate_system = input_coordinate_system
        transformation._output_coordinate_system = output_coordinate_system
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
    def transform_points(self, points: DataArray) -> DataArray:
        pass

    @abstractmethod
    def to_affine(self) -> Affine:
        pass

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
        if not isinstance(self._output_coordinate_system, CoordinateSystem):
            raise ValueError(f"Output coordinate system not specified")
        input_axes = self._input_coordinate_system.axes_names
        output_axes = self._output_coordinate_system.axes_names
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
    def __init__(self) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "identity",
        }

    def inverse(self) -> BaseTransformation:
        return copy.deepcopy(self)

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if input_axes != output_axes:
            raise ValueError(f"Input and output axes must be the same")
        return input_axes, output_axes

    def transform_points(self, points: DataArray) -> DataArray:
        _, _ = self._get_and_validate_axes()
        return points

    def to_affine(self) -> Affine:
        input_axes, _ = self._get_and_validate_axes()
        return Affine(np.eye(len(input_axes) + 1))


# TODO: maybe this transformation will not make it to the final specs, waiting before implementing this
class MapIndex(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()


class MapAxis(BaseTransformation):
    def __init__(self, map_axis: Dict[str, str]) -> None:
        self._map_axis = map_axis

    def to_dict(self) -> Transformation_t:
        return {
            "type": "mapAxis",
            "mapAxis": self._map_axis,
        }

    def inverse(self) -> BaseTransformation:
        if len(self._map_axis.keys()) != len(set(self._map_axis.values())):
            raise ValueError("Cannot invert a map axis transformation with different number of input and output axes")
        else:
            return MapAxis({v: k for k, v in self._map_axis.items()})

    def _get_and_validate_axes(self) -> Tuple[Tuple[str], Tuple[str]]:
        input_axes, output_axes = self._get_axes_from_coordinate_systems()
        if set(input_axes) != set(self._map_axis.keys()):
            raise ValueError(f"The set of input axes must be the same as the set of keys of mapAxis")
        if set(output_axes) != set(self._map_axis.values()):
            raise ValueError(f"The set of output axes must be the same as the set of values of mapAxis")
        return input_axes, output_axes

    def transform_points(self, points: DataArray) -> DataArray:
        input_axes, output_axes = self._get_and_validate_axes()
        raise NotImplementedError()

    def to_affine(self) -> Affine:
        input_axes, output_axes = self._get_and_validate_axes()
        matrix = np.zeros((len(output_axes) + 1, len(input_axes) + 1), dtype=float)
        matrix[-1, -1] = 1
        for i, des_axis in enumerate(output_axes):
            for j, src_axis in enumerate(input_axes):
                if des_axis == self._map_axis[src_axis]:
                    matrix[i, j] = 1
        affine = Affine(matrix)
        return affine


class Translation(BaseTransformation):
    def __init__(
        self,
        translation: Optional[Union[ArrayLike, List[Any]]] = None,
        ndim: Optional[int] = None,
    ) -> None:
        """
        class for storing translation transformations.
        """
        if translation is None:
            assert ndim is not None
            translation = np.ones(ndim, dtype=float)

        if ndim is None:
            assert translation is not None
            if isinstance(translation, list):
                translation = np.array(translation)
            ndim = len(translation)

        assert type(translation) == np.ndarray
        assert len(translation) == ndim
        self.translation = translation
        self._ndim = ndim

    @property
    def src_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def des_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def ndim(self) -> int:
        return self._ndim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "translation",
            "translation": self.translation.tolist(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        return points + self.translation

    def inverse(self) -> BaseTransformation:
        return Translation(translation=-self.translation)

    def to_affine(self, ndims_input: Optional[int] = None, ndims_output: Optional[int] = None) -> Affine:
        m: ArrayLike = np.hstack((np.eye(len(self.translation)), self.translation.reshape(len(self.translation), 1)))
        return Affine(affine=m)


class Scale(BaseTransformation):
    def __init__(
        self,
        scale: Optional[Union[ArrayLike, List[Any]]] = None,
        ndim: Optional[int] = None,
    ) -> None:
        """
        class for storing scale transformations.
        """
        if scale is None:
            assert ndim is not None
            scale = np.ones(ndim, dtype=float)

        if ndim is None:
            assert scale is not None
            if isinstance(scale, list):
                scale = np.array(scale)
            ndim = len(scale)

        assert type(scale) == np.ndarray
        assert len(scale) == ndim
        self.scale = scale
        self._ndim = ndim

    @property
    def src_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def des_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def ndim(self) -> Optional[int]:
        return self._ndim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "scale",
            "scale": self.scale.tolist(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        return points * self.scale

    def inverse(self) -> BaseTransformation:
        new_scale = np.zeros_like(self.scale)
        new_scale[np.nonzero(self.scale)] = 1 / self.scale[np.nonzero(self.scale)]
        return Scale(scale=new_scale)

    def to_affine(self) -> Affine:
        m: ArrayLike = np.hstack((np.diag(self.scale), np.zeros((len(self.scale), 1))))
        return Affine(affine=m)


class Affine(BaseTransformation):
    def __init__(
        self,
        affine: Optional[Union[ArrayLike, List[Any]]] = None,
        src_dim: Optional[int] = None,
        des_dim: Optional[int] = None,
    ) -> None:
        """
        class for storing scale transformations.
        """
        if affine is None:
            assert src_dim is not None and des_dim is not None
            raise NotImplementedError()
        else:
            if isinstance(affine, list):
                affine = np.array(affine)
            if len(affine.shape) == 1:
                raise ValueError(
                    "The specification of affine transformations as 1D arrays/lists is not "
                    "supported. Please be explicit about the dimensions of the transformation."
                )
            des_dim = affine.shape[0]
            src_dim = affine.shape[1] - 1
            last_row = np.zeros(src_dim + 1, dtype=float).reshape(1, -1)
            last_row[-1, -1] = 1
            affine = np.vstack((affine, last_row))
            assert affine.shape == (des_dim + 1, src_dim + 1)

        self.affine = affine
        self._src_dim = src_dim
        self._des_dim = des_dim

    @property
    def src_dim(self) -> Optional[int]:
        return self._src_dim

    @property
    def des_dim(self) -> Optional[int]:
        return self._des_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "affine",
            "affine": self.affine[:-1, :].tolist(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        return q[:2, :].T  # type: ignore[no-any-return]

    def inverse(self) -> BaseTransformation:
        # naive check for numerical stability
        a = self.affine[0, 0]
        b = self.affine[0, 1]
        m = self.affine[0, 2]
        c = self.affine[1, 0]
        d = self.affine[1, 1]
        n = self.affine[1, 2]
        det = a * d - b * c
        closed_form = np.array([[d, -c, 0], [-b, a, 0], [b * n - d * m, c * m - a * n, det]])
        return Affine(affine=closed_form)

    def to_affine(self) -> Affine:
        return copy.deepcopy(self)


class Rotation(BaseTransformation):
    def __init__(
        self,
        rotation: Optional[Union[ArrayLike, List[Any]]] = None,
        ndim: Optional[int] = None,
    ) -> None:
        """
        class for storing scale transformations.
        """
        if rotation is None:
            assert ndim is not None
            s = ndim
            rotation = np.eye(ndim, dtype=float)
        else:
            if isinstance(rotation, list):
                rotation = np.array(rotation)
            s = int(np.sqrt(len(rotation)))
            if len(rotation.shape) == 1:
                assert s * s == len(rotation)
                rotation = rotation.reshape((s, s))
            assert rotation.shape == (s, s)

        self.rotation = rotation
        self._ndim = s

    @property
    def src_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def des_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def ndim(self) -> Optional[int]:
        # TODO: support mixed ndim and remove this property
        return self._ndim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "rotation",
            "rotation": self.rotation.ravel().tolist(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        return (self.rotation @ points.T).T

    def inverse(self) -> BaseTransformation:
        return Rotation(self.rotation.T)

    def to_affine(self) -> Affine:
        m: ArrayLike = np.hstack((self.rotation, np.zeros((ndim, 1))))
        return Affine(affine=m)


class Sequence(BaseTransformation):
    def __init__(self, transformations: Union[List[Any], List[BaseTransformation]]) -> None:
        if isinstance(transformations[0], BaseTransformation):
            self.transformations = transformations
        else:
            self.transformations = [get_transformation_from_dict(t) for t in transformations]  # type: ignore[arg-type]
        self._src_dim = self.transformations[0].src_dim
        self._des_dim = self.transformations[-1].des_dim

    @property
    def src_dim(self) -> Optional[int]:
        return self._src_dim

    @property
    def des_dim(self) -> Optional[int]:
        return self._des_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "sequence",
            "transformations": [t.to_dict() for t in self.transformations],
        }

    def transform_points(self, points: DataArray) -> DataArray:
        for t in self.transformations:
            points = t.transform_points(points)
        return points

    def inverse(self) -> BaseTransformation:
        return Sequence([t.inverse() for t in reversed(self.transformations)])

    def to_affine(self) -> Affine:
        composed = np.eye(self.src_dim + 1)
        for t in self.transformations:
            a: Affine
            if isinstance(t, Affine):
                a = t
            elif isinstance(t, Translation) or isinstance(t, Scale) or isinstance(t, Rotation):
                a = t.to_affine()
            elif isinstance(t, Identity):
                # TODO: this is not the ndim case
                m: ArrayLike = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
                a = Affine(affine=m)
            elif isinstance(t, Sequence):
                a = t.to_affine()
            else:
                raise ValueError(f"Cannot convert {t} to affine")
            composed = a.affine @ composed
        return Affine(affine=composed[:-1, :])


class Displacements(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


# this class is not in the ngff transform specification and is a prototype
class VectorField(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


class Coordinates(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


class InverseOf(BaseTransformation):
    def __init__(self, transformation: Union[Dict[str, Any], BaseTransformation]) -> None:
        if isinstance(transformation, BaseTransformation):
            self.transformation = transformation
        else:
            self.transformation = get_transformation_from_dict(transformation)
        self._ndim = self.transformation.ndim

    @property
    def src_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def des_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def ndim(self) -> Optional[int]:
        # support mixed ndim and remove this property
        return self._ndim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "inverseOf",
            "transformation": self.transformation.to_dict(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        return self.transformation.inverse().transform_points(points)

    def inverse(self) -> BaseTransformation:
        return self.transformation


class Bijection(BaseTransformation):
    def __init__(
        self, forward: Union[Dict[str, Any], BaseTransformation], inverse: Union[Dict[str, Any], BaseTransformation]
    ) -> None:
        if isinstance(forward, BaseTransformation):
            self.forward = forward
        else:
            self.forward = get_transformation_from_dict(forward)

        if isinstance(inverse, BaseTransformation):
            self._inverse = inverse
        else:
            self._inverse = get_transformation_from_dict(inverse)
        assert self.forward.ndim == self._inverse.ndim
        self._ndim = self.forward.ndim

    @property
    def src_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def des_dim(self) -> Optional[int]:
        return self._ndim

    @property
    def ndim(self) -> Optional[int]:
        return self._ndim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "bijection",
            "forward": self.forward.to_dict(),
            "inverse": self._inverse.to_dict(),
        }

    def transform_points(self, points: DataArray) -> DataArray:
        return self.forward.transform_points(points)

    def inverse(self) -> BaseTransformation:
        return self._inverse


class ByDimension(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


#
# @singledispatch
# def get_transform(arg: Any) -> BaseTransformation:
#     raise ValueError(f"Unsupported type: {type(arg)}")
#
#
# @get_transform.register
# def _(arg: xr.DataArray) -> BaseTransformation:
#     if "transform" not in arg.attrs:
#         return BaseTransformation(ndim=arg.ndim)
#     else:
#         return BaseTransformation(
#             translation=arg.attrs["transform"].translation,
#             scale_factors=arg.attrs["transform"].scale_factors,
#             ndim=arg.ndim,
#         )
#
#
# @get_transform.register
# def _(arg: np.ndarray) -> BaseTransformation:  # type: ignore[type-arg]
#     return BaseTransformation(
#         ndim=arg.ndim,
#     )
#
#
# @get_transform.register
# def _(arg: ad.AnnData) -> BaseTransformation:
#     # check if the AnnData has transform information, otherwise fetches it from default scanpy storage
#     if "transform" in arg.uns:
#         return BaseTransformation(
#             translation=arg.uns["transform"]["translation"], scale_factors=arg.uns["transform"]["scale_factors"]
#         )
#     elif "spatial" in arg.uns and "spatial" in arg.obsm:
#         ndim = arg.obsm["spatial"].shape[1]
#         libraries = arg.uns["spatial"]
#         assert len(libraries) == 1
#         library = libraries.__iter__().__next__()
#         scale_factors = library["scalefactors"]["tissue_hires_scalef"]
#         return BaseTransformation(
#             translation=np.zeros(ndim, dtype=float), scale_factors=np.array([scale_factors] * ndim, dtype=float)
#         )
#     else:
#         return BaseTransformation(ndim=2)
