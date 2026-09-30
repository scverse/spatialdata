from __future__ import annotations

import numpy as np

from spatialdata._types import ArrayLike


class IncompatibleCoordSystemsError(Exception):
    def __init__(
        self,
        transformation_name: str | None = None,
        axes_error_message: str | None = None,
    ) -> None:
        self.transformation_name = transformation_name
        axes_message = axes_error_message or "Output axes can't be mapped to input axes"
        axes_transformation_message = (
            transformation_name and f"In the context of a `{transformation_name}` transformation, {axes_message}"
        )
        super().__init__(axes_transformation_message or axes_message)


class MissingAxisError(Exception):
    def __init__(self, axis_name: str, cs_name: str) -> None:
        self.axis_name = axis_name
        self.cs_name = cs_name
        super().__init__(f"Axis {axis_name} is not in coordinate system {cs_name}")


class UnexpectedShapeError(Exception):
    def __init__(
        self,
        *,
        array_shape: tuple[int, ...],
        expected_shape: tuple[int, ...] | str | None = None,
        array_name: str | None = None,
    ) -> None:
        self.array_shape = array_shape
        self.expected_shape = expected_shape
        message = "Unexpected array shape"
        if array_name is not None:
            message += f"for '{array_name}'"
        message += f": {array_shape}"
        if expected_shape is not None:
            message += f" instead of {expected_shape}"
        super().__init__(message)


class DeterminantDifferentFromOne(Exception):
    def __init__(self, matrix: ArrayLike) -> None:
        self.matrix = matrix
        super().__init__("Matrix does not have det(M) == 1")


class NotOrthonormalError(Exception):
    def __init__(self, matrix: ArrayLike) -> None:
        self.matrix = matrix
        super().__init__("Matrix is not orthonormal")


class EmptyTransformSequenceError(Exception):
    def __init__(self) -> None:
        super().__init__("Empty sequence of transformations")


class AxisRedefinitionError(Exception):
    def __init__(self, axis_name: str) -> None:
        self.axis_name = axis_name
        super().__init__(f"Axis {axis_name} is defined multiple times")


class UnmappedAxisError(Exception):
    def __init__(self, axis_name: str, cs_name: str) -> None:
        self.axis_name = axis_name
        self.cs_name = cs_name
        super().__init__(f"Axis {axis_name} from coordinate system {cs_name} is not mapped to anything")


class NonZeroDroppedAxisError(ValueError):
    """
    Raised by :class:`DynamicDropAxisEdge` when a column that is supposed to be
    dropped does not contain only zeros.

    Parameters
    ----------
    axis_name : str
        Name of the axis whose column violated the zero‑only requirement.
    expected : float
        The constant value that was expected (always ``0`` in the current
        implementation).
    observed : numpy.ndarray
        The full column extracted from the input points that contains the
        offending values.
    """

    def __init__(self, axis_name: str, expected: float, observed: np.ndarray) -> None:
        self.axis_name = axis_name
        self.expected = expected
        self.observed = observed
        super().__init__(
            f"Axis '{axis_name}' contains non‑zero values; "
            f"expected all entries to be {expected}, but observed {observed}"
        )
