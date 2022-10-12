from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from spatialdata._types import ArrayLike

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
    "compose_transformations",
]


class BaseTransformation(ABC):
    @property
    @abstractmethod
    def src_dim(self) -> Optional[int]:
        pass

    @property
    @abstractmethod
    def des_dim(self) -> Optional[int]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def compose_with(self, transformation: BaseTransformation) -> BaseTransformation:
        return compose_transformations(self, transformation)

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseTransformation):
            return False
        return self.to_dict() == other.to_dict()

    @abstractmethod
    def to_affine(self, input_axes: Optional[Tuple[str]] = None, output_axes: Optional[Tuple[str]] = None) -> Affine:
        pass


def compose_transformations(transformation0: BaseTransformation, transformation1: BaseTransformation) -> Sequence:
    return Sequence([transformation0, transformation1])


def get_transformation_from_json(s: str) -> BaseTransformation:
    d = json.loads(s)
    return get_transformation_from_dict(d)


def get_transformation_from_dict(d: Dict[str, Any]) -> BaseTransformation:
    kw = d.copy()
    type = kw["type"]
    cls: Type[BaseTransformation]
    if type == "identity":
        cls = Identity
    elif type == "mapIndex":
        cls = MapIndex  # type: ignore
    elif type == "mapAxis":
        cls = MapAxis  # type: ignore
    elif type == "translation":
        cls = Translation
    elif type == "scale":
        cls = Scale
    elif type == "affine":
        cls = Affine
    elif type == "rotation":
        cls = Rotation
    elif type == "sequence":
        cls = Sequence
    elif type == "displacements":
        cls = Displacements  # type: ignore
    elif type == "coordinates":
        cls = Coordinates  # type: ignore
    elif type == "vectorField":
        cls = VectorField  # type: ignore
    elif type == "inverseOf":
        cls = InverseOf
    elif type == "bijection":
        cls = Bijection
    elif type == "byDimension":
        cls = ByDimension  # type: ignore
    else:
        raise ValueError(f"Unknown transformation type: {type}")
    del kw["type"]
    if "input" in kw:
        del kw["input"]
    if "output" in kw:
        del kw["output"]
    return cls(**kw)


class Identity(BaseTransformation):
    def __init__(self) -> None:
        self._src_dim = None
        self._des_dim = None

    @property
    def src_dim(self) -> Optional[int]:
        return self._src_dim

    @property
    def des_dim(self) -> Optional[int]:
        return self._des_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "identity",
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return points

    def inverse(self) -> BaseTransformation:
        return copy.deepcopy(self)

    def to_affine(self, input_axes: Optional[Tuple[str]] = None, output_axes: Optional[Tuple[str]] = None) -> Affine:
        raise NotImplementedError()

    # def to_affine(self, input_axes: Optional[Tuple[str]] = None, output_axes: Optional[Tuple[str]] = None) -> Affine:
    #     if input_axes is None and output_axes is None:
    #         raise ValueError("Either input_axes or output_axes must be specified")
    #     return Affine(np.eye(ndims_output, ndims_input))


class MapIndex(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


class MapAxis(BaseTransformation):
    def __init__(self) -> None:
        raise NotImplementedError()

    # @property
    # def ndim(self) -> Optional[int]:
    #     return self._ndim


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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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
            affine = self._get_affine_iniection_from_dims(src_dim, des_dim)
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

    @staticmethod
    def _get_affine_iniection_from_dims(src_dim: int, des_dim: int) -> np.ndarray:
        last_column = np.zeros(des_dim + 1, dtype=float).reshape(-1, 1)
        last_column[-1, -1] = 1
        affine = np.hstack(
            (
                np.vstack((np.eye(des_dim, src_dim, dtype=float), np.zeros(src_dim).reshape(1, -1))),
                last_column,
            )
        )
        return affine

    @staticmethod
    def _get_affine_iniection_from_axes(src_axes: Tuple[str, ...], des_axes: Tuple[str, ...]) -> np.ndarray:
        # this function could be implemented with a composition of map axis and _get_affine_iniection_from_dims
        affine = np.zeros((len(des_axes) + 1, len(src_axes) + 1), dtype=float)
        affine[-1, -1] = 1
        for i, des_axis in enumerate(des_axes):
            for j, src_axis in enumerate(src_axes):
                if des_axis == src_axis:
                    affine[i, j] = 1
        return affine

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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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

    def transform_points(self, points: ArrayLike) -> ArrayLike:
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
