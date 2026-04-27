from __future__ import annotations

import os
from pathlib import Path


def _config_path_for(tmp_path: Path) -> Path:
    return tmp_path / "settings.json"


class TestDefaults:
    def test_default_settings(self, settings_cls):
        s = settings_cls()
        assert s.shapes_geometry_encoding == "WKB"
        assert s.large_chunk_threshold_bytes == 2_147_483_647
        assert s.raster_chunks is None
        assert s.raster_shards is None
        assert s.custom_config_path is None

    def test_change_settings_default_path(self, settings_cls):
        s = settings_cls()
        s.shapes_geometry_encoding = "geoarrow"
        s.large_chunk_threshold_bytes = 1_000_000_000
        s.raster_chunks = (512, 512)
        s.raster_shards = (1024, 1024)
        s.save()
        s = settings_cls().load()
        assert s.shapes_geometry_encoding == "geoarrow"
        assert s.large_chunk_threshold_bytes == 1_000_000_000
        assert s.raster_chunks == [512, 512]
        assert s.raster_shards == [1024, 1024]
        assert s.custom_config_path is None

    def test_change_settings_custom_path(self, settings_cls, tmp_path):
        os.environ["SPATIALDATA_SHAPES_GEOMETRY_ENCODING"] = "geoarrow"
        os.environ["SPATIALDATA_RASTER_CHUNKS"] = "40,40,40"

        target_path = tmp_path / "custom_settings.json"
        s = settings_cls().load()
        assert s.shapes_geometry_encoding == "geoarrow"
        assert s.raster_chunks == (40, 40, 40)

        # We set the value also using environment variables to test whether these properly overwrite
        s.large_chunk_threshold_bytes = 1_000_000_000
        os.environ["SPATIALDATA_LARGE_CHUNK_THRESHOLD_BYTES"] = "1_111_111_111"

        s.raster_chunks = (512, 512)
        s.raster_shards = (1024, 1024)
        s.save(path=target_path)
        s = settings_cls().load()
        assert s.shapes_geometry_encoding == "geoarrow"
        assert s.large_chunk_threshold_bytes == 1_111_111_111
        assert s.raster_chunks == (40, 40, 40)
        assert s.raster_shards is None
        assert s.custom_config_path == str(target_path)

        s.reset()
        s.save()
        assert s.custom_config_path is None  # This returns False
        s = settings_cls().load()
        assert s.custom_config_path is None
