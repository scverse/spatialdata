from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager


class CoordinateSystemNotFoundError(ValueError):
    """
    Exception raised when a coordinate system is not found in the transformation manager.

    Attributes
    ----------
    name : str
        The name of the coordinate system that was not found.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Coordinate system '{name}' not found in the transformation manager.")


class ElementNotFoundError(KeyError):
    """
    Exception raised when an element is not found in the transformation manager.

    Attributes
    ----------
    element_name : str
        The name of the element that was not found.
    """

    def __init__(self, element_name: str) -> None:
        self.element_name = element_name
        super().__init__(f"Element '{element_name}' not found in the transformation manager.")


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
        self.input_cs_name = source_cs_name
        self.output_cs_name = target_cs_name
        msg = f"Transformation from '{source_cs_name}' to '{target_cs_name}' not found"
        if edge_key is not None:
            msg = f"{msg} with key '{edge_key}'"
        super().__init__(msg)


class CoordinateSystemAlreadyExistsError(ValueError):
    """
    Exception raised when coordinate system already exists.

    Attributes
    ----------
    name : str
        The name of the coordinate system that already exists.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Coordinate system '{name}' already exists")


class ElementAlreadyExistsError(ValueError):
    """
    Exception raised when trying to add an Element that already exists.

    Attributes
    ----------
    name : str
        The name of the Element that already exists.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Element '{name}' already exists in the transformation manager")


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
        self.source_cs_name = source_cs_name
        self.target_cs_name = target_cs_name
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
        self.source_cs_name = source_cs_name
        self.target_cs_name = target_cs_name
        super().__init__(f"Transformation Path ambiguous from {source_cs_name} to {target_cs_name}")


class CannotRemoveCoordinateSystemError(ValueError):
    """Exception raised when trying to remove a coordinate system.

    Attributes
    ----------
    name: str
        name of the coordinate system
    """

    def __init__(self, name: str) -> None:
        self.name = name
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
        self.name = name
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
        self.name = name
        self.associated_elements = associated_elements
        super().__init__(f"Coordinate system '{name}' has elements belonging to it: {associated_elements}")


class TransformationManagerWarning(UserWarning):
    """Base warning category for TransformationManager."""

    pass


class InternalAttributeAccessWarning(TransformationManagerWarning):
    """Warning for direct access to internal attributes."""

    pass


@contextmanager
def suppress_direct_internal_attribute_access_warning() -> Iterator[None]:
    """Context manager to suppress InternalAttributeAccessWarning when accessing internal attributes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InternalAttributeAccessWarning)
        yield
