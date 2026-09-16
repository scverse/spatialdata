from __future__ import annotations

from enum import StrEnum

TRANSFORM_KEY = "transform"
DEFAULT_COORDINATE_SYSTEM = "global"
C = "c"
Z = "z"
Y = "y"
X = "x"


class NgffAxisType(StrEnum):
    """Enum defining NGFF axis types"""

    SPACE = "space"
    CHANNEL = "channel"


axis_type_mapping_ngff = {
    C: NgffAxisType.CHANNEL,
    X: NgffAxisType.SPACE,
    Y: NgffAxisType.SPACE,
    Z: NgffAxisType.SPACE,
}
ATTRS_KEY = "spatialdata_attrs"
