from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np

from spatialdata._types import ArrayLike

__all__ = [
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


class BaseTransformation(ABC):
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        pass

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def compose_with(transformation) -> BaseTransformation:
        pass

    @staticmethod
    def compose(transformation0, transformation1):
        pass

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass


def get_transformation_from_json(s: str) -> BaseTransformation:
    d = json.loads(s)
    return get_transformation_from_dict(d)


def get_transformation_from_dict(d: Dict) -> BaseTransformation:
    kw = d["coordinateTransformations"]
    type = kw["type"]
    if type == "identity":
        cls = Identity
    elif type == "mapIndex":
        cls = MapIndex
    elif type == "mapAxis":
        cls = MapAxis
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
        cls = Displacements
    elif type == "coordinates":
        cls = Coordinates
    elif type == "vectorField":
        cls = VectorField
    elif type == "inverseOf":
        cls = InverseOf
    elif type == "bijection":
        cls = Bijection
    elif type == "byDimension":
        cls = ByDimension
    else:
        raise ValueError(f"Unknown transformation type: {type}")
    del kw["type"]
    del kw["input"]
    del kw["output"]
    return cls(**kw)


class Identity(BaseTransformation):
    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "identity",
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return points

    def inverse(self) -> BaseTransformation:
        return copy.deepcopy(self)


class MapIndex(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


class MapAxis(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


class Translation(BaseTransformation):
    def __init__(
        self,
        translation: Optional[ArrayLike] = None,
        ndim: Optional[int] = None,
    ):
        """
        class for storing translation transformations.
        """
        if translation is None:
            assert ndim is not None
            translation = np.ones(ndim, dtype=float)

        if ndim is None:
            assert translation is not None
            ndim = len(translation)

        assert len(translation) == ndim
        self.translation = translation
        self.ndim = ndim

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "translation",
                "translation": self.translation.tolist(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return points + self.translation

    def inverse(self) -> BaseTransformation:
        return Translation(translation=-self.translation)


class Scale(BaseTransformation):
    def __init__(
        self,
        scale: Optional[ArrayLike] = None,
        ndim: Optional[int] = None,
    ):
        """
        class for storing scale transformations.
        """
        if scale is None:
            assert ndim is not None
            scale = np.ones(ndim, dtype=float)

        if ndim is None:
            assert scale is not None
            ndim = len(scale)

        assert len(scale) == ndim
        self.scale = scale
        self.ndim = ndim

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "scale",
                "scale": self.scale.tolist(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return points * self.scale

    def inverse(self) -> BaseTransformation:
        new_scale = np.zeros_like(self.scale)
        new_scale[np.nonzero(self.scale)] = 1 / self.scale[np.nonzero(self.scale)]
        return Scale(scale=new_scale)


class Affine(BaseTransformation):
    def __init__(
        self,
        affine: Optional[ArrayLike] = None,
    ):
        """
        class for storing scale transformations.
        """
        if affine is None:
            affine = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        else:
            if len(affine.shape) == 1:
                assert len(affine) == 6
            else:
                assert affine.shape == (3, 3)

        self.affine = affine
        self.ndim = 2

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "scale",
                "scale": self.affine[:2, :].ravel().tolist(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        return q[:2, :].T

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


class Rotation(BaseTransformation):
    def __init__(
        self,
        rotation: Optional[ArrayLike] = None,
    ):
        """
        class for storing scale transformations.
        """
        if rotation is None:
            rotation = np.array([[1, 0], [0, 1]], dtype=float)
        else:
            if len(rotation.shape) == 1:
                assert len(rotation) == 4
            else:
                assert rotation.shape == (2, 2)

        self.rotation = rotation
        self.ndim = 2

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "scale",
                "scale": self.rotation.ravel().tolist(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return (self.rotation @ points.T).T

    def inverse(self) -> BaseTransformation:
        return Rotation(self.rotation.T)


class Sequence(BaseTransformation):
    def __init__(self, transformations: Union[List[Dict], List[BaseTransformation]]):
        if isinstance(transformations[0], BaseTransformation):
            self.transformations = transformations
        else:
            self.transformations = [get_transformation_from_dict(t) for t in transformations]

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "sequence",
                "transformations": [t.to_dict() for t in self.transformations],
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        for t in self.transformations:
            points = t.transform_points(points)
        return points

    def inverse(self) -> BaseTransformation:
        return Sequence([t.inverse() for t in reversed(self.transformations)])


class Displacements(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


# this class is not in the ngff transform specification and is a prototype
class VectorField(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


class Coordinates(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


class InverseOf(BaseTransformation):
    def __init__(self, transformation: Union[Dict, BaseTransformation]):
        if isinstance(transformation, BaseTransformation):
            self.transformation = transformation
        else:
            self.transformation = get_transformation_from_dict(transformation)

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "inverseOf",
                "transformation": self.transformation.to_dict(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return self.transformation.inverse().transform_points()

    def inverse(self) -> BaseTransformation:
        return self.transformation


class Bijection(BaseTransformation):
    def __init__(self, forward: Union[Dict, BaseTransformation], inverse: Union[Dict, BaseTransformation]):
        if isinstance(forward, BaseTransformation):
            self.forward = forward
        else:
            self.forward = get_transformation_from_dict(forward)

        if isinstance(inverse, BaseTransformation):
            self.inverse = inverse
        else:
            self.inverse = get_transformation_from_dict(inverse)

    def to_dict(self) -> Dict:
        return {
            "coordinateTransformations": {
                "type": "bijection",
                "forward": self.forward.to_dict(),
                "inverse": self.inverse.to_dict(),
            }
        }

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return self.forward.transform_points(points)

    def inverse(self) -> BaseTransformation:
        return self.inverse


class ByDimension(BaseTransformation):
    def __init__(self):
        raise NotImplementedError()


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
