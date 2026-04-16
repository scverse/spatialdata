from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from platformdirs import user_config_dir


def _config_path() -> Path:
    """Return the platform-appropriate path to the user config file."""
    return Path(user_config_dir(appname="spatialdata")) / "settings.json"


@dataclass
class Settings:
    """Global settings for spatialdata.

    Attributes
    ----------
    custom_config_path
        The path specified by the user of where to store the settings.
    shapes_geometry_encoding
        Default geometry encoding for GeoParquet files when writing shapes.
        Can be "WKB" (Well-Known Binary) or "geoarrow".
        See :meth:`geopandas.GeoDataFrame.to_parquet` for details.
    large_chunk_threshold_bytes
        Chunk sizes bigger than this value (bytes) can trigger a compression error.
        See https://github.com/scverse/spatialdata/issues/812#issuecomment-2559380276
        If detected during parsing/validation, a warning is raised.
    raster_chunks
        The chunksize to use for chunking an array. Length of the tuple must match
        the number of dimensions.
    raster_shards
        The default shard size (zarr v3) to use when storing arrays. Length of the tuple
        must match the number of dimensions.
    """

    custom_config_path: Path | None = None
    shapes_geometry_encoding: Literal["WKB", "geoarrow"] = "WKB"
    large_chunk_threshold_bytes: int = 2147483647

    raster_chunks: tuple[int, ...] | None = None
    raster_shards: tuple[int, ...] | None = None

    def save(self, path: Path | str | None = None) -> None:
        """Store current settings on disk.

        If Path is specified, it will store the config settings to this location. Otherwise, stores
        the config in the default config directory for the given operating system.

        Parameters
        ----------
        path
            The path to use for storing settings if different from default. Must be
            a json file. This will be stored in the global config as the custom_config_path.

        Returns
        -------
        Path
            The path the settings were written to.
        """
        target = Path(path) if path else _config_path()

        if not str(target).endswith(".json"):
            raise ValueError("Path must end with .json")

        if path is not None:
            data = asdict(self)
            data["custom_config_path"] = str(target)
            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            global_path = _config_path()
            global_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with global_path.open(encoding="utf-8") as f:
                    global_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                global_data = {}
            global_data["custom_config_path"] = str(target)
            with global_path.open("w", encoding="utf-8") as f:
                json.dump(global_data, f, indent=2)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            data = asdict(self)
            data["custom_config_path"] = str(data["custom_config_path"]) if data["custom_config_path"] else None
            with target.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Settings:
        """Load settings from disk.

        This method falls back to default settings if either there is no config at the
        given path or there is a decoding error. Unknown or renamed keys in the file
        are silently ignored, e.g. old config files will not cause errors.

        Parameters
        ----------
        path
            The path to the config file if different from default. If not specified,
            the default location is used.

        Returns
        -------
        Settings
            A populated Settings instance.
        """
        target = Path(path) if path else _config_path()

        if not target.exists():
            instance = cls()
            instance.apply_env()
            return instance

        try:
            with target.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            instance = cls()
            instance.apply_env()
            return instance

        # This prevents fields from old config files to be used.
        known_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        instance = cls(**known_fields)
        instance.apply_env()
        return instance

    def reset(self) -> None:
        """Inplace reset all settings to their built-in defaults (in memory only).

        Call 'save' method afterwards if you want the reset to be persisted.
        """
        defaults = Settings()
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(defaults, field_name))

    def apply_env(self) -> None:
        """Apply environment variable overrides on top of the current state.

        Env vars take precedence over both the config file and any
        in-session assignments. Useful in CI pipelines or HPC clusters
        where you cannot edit the config file.

        Supported variables
        -------------------
        SPATIALDATA_CUSTOM_CONFIG_PATH -> custom_config_path
        SPATIALDATA_SHAPES_GEOMETRY_ENCODING   → shapes_geometry_encoding
        SPATIALDATA_LARGE_CHUNK_THRESHOLD_BYTES → large_chunk_threshold_bytes
        SPATIALDATA_CHUNKS                     → chunks
        SPATIALDATA_SHARDS                     → shards (integer or "none")
        """
        _ENV: dict[str, tuple[str, type]] = {
            "SPATIALDATA_CUSTOM_CONFIG_PATH": ("custom_config_path", Path),
            "SPATIALDATA_SHAPES_GEOMETRY_ENCODING": ("shapes_geometry_encoding", str),
            "SPATIALDATA_LARGE_CHUNK_THRESHOLD_BYTES": ("large_chunk_threshold_bytes", int),
            "SPATIALDATA_CHUNKS": ("raster_chunks", str),
            "SPATIALDATA_SHARDS": ("raster_shards", str),  # handled specially below
        }
        for env_key, (field_name, cast) in _ENV.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if field_name in ("raster_chunks", "raster_shards"):
                setattr(
                    self,
                    field_name,
                    None if raw.lower() in ("none", "") else tuple(int(v) for v in raw.split(",")),
                )
            else:
                setattr(self, field_name, cast(raw))

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in asdict(self).items())
        return f"Settings({fields})"

    @staticmethod
    def config_path() -> Path:
        """Return platform-specific path where settings are stored."""
        return _config_path()


settings = Settings.load()

# Backwards compatibility alias
LARGE_CHUNK_THRESHOLD_BYTES = settings.large_chunk_threshold_bytes
