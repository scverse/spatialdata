from typing import Any, Optional

import pytest

from spatialdata._core.models import PointsModel
from spatialdata._io.format import PointsFormat

Points_f = PointsFormat()


class TestFormat:
    """Test format."""

    @pytest.mark.parametrize("attrs_key", [PointsModel.ATTRS_KEY])
    @pytest.mark.parametrize("feature_key", [None, PointsModel.FEATURE_KEY])
    @pytest.mark.parametrize("instance_key", [None, PointsModel.INSTANCE_KEY])
    def test_format_points(
        self,
        attrs_key: Optional[str],
        feature_key: Optional[str],
        instance_key: Optional[str],
    ) -> None:
        metadata: dict[str, Any] = {attrs_key: {"version": Points_f.spatialdata_version}}
        format_metadata: dict[str, Any] = {attrs_key: {}}
        if feature_key is not None:
            metadata[attrs_key][feature_key] = "target"
        if instance_key is not None:
            metadata[attrs_key][instance_key] = "cell_id"
        format_metadata[attrs_key] = Points_f.attrs_from_dict(metadata)
        metadata[attrs_key].pop("version")
        assert metadata[attrs_key] == Points_f.attrs_to_dict(format_metadata)
        if feature_key is None and instance_key is None:
            assert len(format_metadata[attrs_key]) == len(metadata[attrs_key]) == 0
