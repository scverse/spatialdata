from enum import unique

from spatialdata._constants._enum import ModeEnum


@unique
class Geometry(ModeEnum):
    POLYGON = "POLYGON"
    MULTIPOLYGON = "MULTIPOLYGON"


@unique
class Raster(ModeEnum):
    IMAGE = "IMAGE"
    LABELS = "LABELS"
