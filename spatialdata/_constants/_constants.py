from enum import unique

from spatialdata._constants._enum import ModeEnum


@unique
class ShapesType(ModeEnum):
    CIRCLES = "circles"
    TRIANGLES = "triangles"
    SQUARES = "squares"
    HEXAGONS = "hexagons"
