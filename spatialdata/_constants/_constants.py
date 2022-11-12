from enum import unique

from spatialdata._constants._enum import ModeEnum


@unique
class Geometry(ModeEnum):
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"


@unique
class Raster(ModeEnum):
    IMAGE = "image"
    LABELS = "labels"
