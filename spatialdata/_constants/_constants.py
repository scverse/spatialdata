from enum import unique

from spatialdata._constants._enum import ModeEnum


@unique
class RasterType(ModeEnum):
    IMAGE = "Image"
    LABEL = "Label"


@unique
class ShapeType(ModeEnum):
    CIRCLE = "Circle"
    SQUARE = "Square"
