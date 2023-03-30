from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData


class SpatialDataToDataDict:
    def __init__(self, data_mapping: dict[str, str]):
        self.data_mapping = data_mapping

    @staticmethod
    def _parse_spatial_data_path(sdata_path: str) -> tuple[str, str]:
        """Convert a path in a SpatialData object to the element type and element name."""
        path_components = sdata_path.split("/")
        assert len(path_components) == 2

        element_type = path_components[0]
        element_name = path_components[1]

        return element_type, element_name

    def __call__(self, sdata: SpatialData) -> dict[str, Any]:
        data_dict = {}
        for sdata_path, data_dict_key in self.data_mapping.items():
            # get data item from the SpatialData object and add it to the data dictig
            element_type, element_name = self._parse_spatial_data_path(sdata_path)

            if element_type == "table":
                element = getattr(sdata, element_type)
            else:
                element_dict = getattr(sdata, element_type)
                element = element_dict[element_name]

            data_dict[data_dict_key] = element

        return data_dict
