from dataclasses import dataclass
from typing import Literal


@dataclass
class Settings:
    """Global settings for spatialdata.

    Attributes
    ----------
    shapes_geometry_encoding
        Default geometry encoding for GeoParquet files when writing shapes.
        Can be "WKB" (Well-Known Binary) or "geoarrow".
        See :meth:`geopandas.GeoDataFrame.to_parquet` for details.
    large_chunk_threshold_bytes
        Chunk sizes bigger than this value (bytes) can trigger a compression error.
        See https://github.com/scverse/spatialdata/issues/812#issuecomment-2559380276
        If detected during parsing/validation, a warning is raised.
    """

    shapes_geometry_encoding: Literal["WKB", "geoarrow"] = "WKB"
    large_chunk_threshold_bytes: int = 2147483647


settings = Settings()

# Backwards compatibility alias
LARGE_CHUNK_THRESHOLD_BYTES = settings.large_chunk_threshold_bytes
