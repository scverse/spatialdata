from __future__ import annotations

from enum import StrEnum

"""Key used to store transformation data in graph edge attributes."""
DEFAULT_COORDINATE_SYSTEM = "global"
C = "c"
Z = "z"
Y = "y"
X = "x"


class NgffAxisType(StrEnum):
    """Enum defining NGFF axis types"""

    CHANNEL = "channel"
    SPACE = "space"


axis_type_mapping_ngff = {
    C: NgffAxisType.CHANNEL,
    X: NgffAxisType.SPACE,
    Y: NgffAxisType.SPACE,
    Z: NgffAxisType.SPACE,
}
TRANSFORM_KEY = "transform"
