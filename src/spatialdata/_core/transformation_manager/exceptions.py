from __future__ import annotations


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
    """

    def __init__(self, input_cs_name: str, output_cs_name: str) -> None:
        self.input_cs_name = input_cs_name
        self.output_cs_name = output_cs_name
        super().__init__(f"Transformation from '{input_cs_name}' to '{output_cs_name}' not found.")
