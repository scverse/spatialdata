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


class WritingToZarrV2DeprecationWarning(DeprecationWarning):
    """Warning raised when writing to zarr v2 format."""

    message = (
        "Writing to zarr v2 format is currently deprecated in spatialdata "
        "and will be removed in a future version. "
        "Please consider writing to zarr v3."
    )
