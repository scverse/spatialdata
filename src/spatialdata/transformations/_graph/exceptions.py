from __future__ import annotations

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from spatialdata.transformations import BaseTransformation


class CoordinateSystemNotFoundError(ValueError):
    """
    Exception raised when a coordinate system is not found in the transformation manager.

    Attributes
    ----------
    name : str
        The name of the coordinate system that was not found.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(f"Coordinate system '{name}' not found in the transformation manager.")


class CoordinateSystemAlreadyExistsError(ValueError):
    """
    Exception raised when a coordinate system already exists in the transformation manager.

    Attributes
    ----------
    name : str
        The name of the coordinate system to check.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(f"Coordinate system '{name}' already exists in the transformation manager.")


class ElementNotRegisteredToAnyCoordinateSystemError(KeyError):
    """
    Exception raised when an element is not found in the transformation manager.

    Attributes
    ----------
    element_name : str
        The name of the element that was not found.
    """

    def __init__(self, element_name: str) -> None:
        self.element_name: str = element_name
        super().__init__(f"Element '{element_name}' has not been registered to any coordinate system.")


class TransformationNotFoundError(KeyError):
    """
    Exception raised when a transformation is not found between coordinate systems.

    Attributes
    ----------
    input_cs_name : str
        The name of the input coordinate system.
    output_cs_name : str
        The name of the output coordinate system.
    edge_key: str or None
        key used when adding transformation
    """

    def __init__(self, source_cs_name: str, target_cs_name: str, edge_key: str | None = None) -> None:
        self.input_cs_name: str = source_cs_name
        self.output_cs_name: str = target_cs_name
        self.edge_key: str | None = edge_key
        msg = f"Transformation from '{source_cs_name}' to '{target_cs_name}' not found"
        if edge_key is not None:
            msg += f" with key '{edge_key}'"
        super().__init__(msg)


class ElementAlreadyExistsError(ValueError):
    """
    Exception raised when trying to add an Element that already exists.

    Attributes
    ----------
    name : str
        The name of the Element that already exists.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(f"Element '{name}' already exists in the transformation manager")


class InvalidPathError(ValueError):
    """
    Exception raised when a path is defined with less than 2 nodes.

    Attributes
    ----------
    invalid_path :
        list of coordinate system names
    """

    def __init__(self, invalid_path: list[str]) -> None:
        super().__init__(f"Found an invalid path with less than 2 nodes: {invalid_path}")


class TransformationPathNotFoundError(ValueError):
    """
    Exception raised when no transformation path exists between coordinate systems.

    Attributes
    ----------
    source_cs_name : str
        The name of the source coordinate system.
    target_cs_name : str
        The name of the target coordinate system.
    """

    def __init__(self, source_cs_name: str, target_cs_name: str) -> None:
        self.source_cs_name: str = source_cs_name
        self.target_cs_name: str = target_cs_name
        super().__init__(f"No transformation path found from {source_cs_name} to {target_cs_name}")


class TransformationPathAmbiguousError(ValueError):
    """
    Exception raised when multiple transformation path exists between coordinate systems.

    Attributes
    ----------
    source_cs_name : str
        The name of the source coordinate system.
    target_cs_name : str
        The name of the target coordinate system.
    """

    def __init__(self, source_cs_name: str, target_cs_name: str) -> None:
        self.source_cs_name: str = source_cs_name
        self.target_cs_name: str = target_cs_name
        base_msg = f"Transformation Path ambiguous from {source_cs_name} to {target_cs_name}."
        cause_of_confusion = self.cause_of_confusion()
        msg = f"{base_msg} {cause_of_confusion}" if cause_of_confusion else base_msg
        super().__init__(msg)

    def cause_of_confusion(self) -> str:
        return ""


class TransformationPathAmbiguousNoEdgeExpectedError(TransformationPathAmbiguousError):
    def __init__(self, source_cs_name: str, target_cs_name: str) -> None:
        super().__init__(source_cs_name, target_cs_name)

    @override
    def cause_of_confusion(self) -> str:

        return "Multiple edges found. None of them were specified to be expected"


class TransformationPathAmbiguousMultipleEdgeExpectedError(TransformationPathAmbiguousError):
    def __init__(self, source_cs_name: str, target_cs_name: str, number_of_edges_expected: int) -> None:
        self.number_of_edges_expected: int = number_of_edges_expected
        super().__init__(source_cs_name, target_cs_name)

    @override
    def cause_of_confusion(self) -> str:
        return f"Multiple ({self.number_of_edges_expected}) edges were specified to be expected"


class TransformationPathNotSimple(ValueError):
    """
    Exception raised when a path represented by a list of coordinate systems is not simple.

    A simple path is one in which each coordinate system appears only once

    Attributes
    ----------
    path:
        list of coordinate system names
    """

    def __init__(self, path: list[str]) -> None:
        self.path = path
        super().__init__(
            "Transformation Path not simple, i.e., some coordinate systems appear multiple times:\n" + f"{path}"
        )


class CannotRemoveCoordinateSystemError(ValueError):
    """Exception raised when trying to remove a coordinate system.

    Attributes
    ----------
    name: str
        name of the coordinate system
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(f"Cannot remove coordinate system with name {name}.")


class CoordinateSystemHasTransformationsError(ValueError):
    """
    Exception raised when trying to remove a coordinate system that has associated transformations.

    Attributes
    ----------
    name : str
        The name of the coordinate system that has transformations.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(f"Coordinate System ('{name}') has transformations.")


class CoordinateSystemHasElementsError(ValueError):
    """
    Exception raised when trying to remove a coordinate system that has associated elements.

    Attributes
    ----------
    name : str
        The name of the coordinate system that has elements.
    associated_elements : list[str]
        List of element names associated with the coordinate system.
    """

    def __init__(self, name: str, associated_elements: list[str]) -> None:
        self.name: str = name
        self.associated_elements: list[str] = associated_elements
        super().__init__(f"Coordinate system '{name}' has elements belonging to it: {associated_elements}")


class TransformationManagerWarning(UserWarning):
    """Base warning category for TransformationManager."""

    pass


class AxisParsingException(Exception):
    pass


class NGFFCompatibilityError(Exception):
    """
    Exception raised when a transformation cannot be mapped to an NGFF transformation edge.

    Attributes
    ----------
    msg : str
        The error message describing the compatibility issue.
    """

    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(msg)


class IncompatibleTransformationOutputCoordinateSystemError(Exception):
    """
    Exception raised the expected output coordinate system is incompatible with an input coordinate system and a transformation edge.
    """

    def __init__(
        self, input_coordinate_system_name: str, transformation: BaseTransformation, output_coordinate_system_name: str
    ) -> None:

        self.input_coordinate_system_name = input_coordinate_system_name
        self.output_coordinate_system_name = output_coordinate_system_name
        self.transformation = transformation
        super().__init__(
            f"Transformation incompatible with input/output coordinate systems: "
            f"\nInput coordinate system name: {input_coordinate_system_name}"
            f"\nOutput coordinate system name: {output_coordinate_system_name}"
            f"\nTransformation: {transformation}"
        )
