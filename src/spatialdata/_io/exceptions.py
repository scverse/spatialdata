from __future__ import annotations

from ome_zarr.format import Format


class FormatVersionUnknownError(ValueError):
    """Exception raised when an unknown element format is encountered."""

    def __init__(self, element_type: str, version_encountered: Format):
        self.element_type = element_type
        self.version_encountered = version_encountered
        self.message = (
            f"Encountered unknown element format version "
            f"`{self.version_encountered}` for element of type `{self.element_type}`"
        )
        super().__init__(self.message)
