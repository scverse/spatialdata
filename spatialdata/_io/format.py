from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from anndata import AnnData
from ome_zarr.format import CurrentFormat
from pandas.api.types import is_categorical_dtype

CoordinateTransform_t = List[Dict[str, Any]]


class SpatialDataFormat(CurrentFormat):
    """
    SpatialDataFormat defines the format of the spatialdata
    package.
    """

    def validate_shapes_parameters(
        self,
        shapes_parameters: Dict[str, Union[str, float, Dict[str, CoordinateTransform_t]]],
    ) -> None:
        """
        Validate the shape parameters.

        Parameters
        ----------
        shapes_parameters
            Shape parameters.

        Returns
        ------
            Nothing.
        """
        if shapes_parameters is None:
            raise ValueError("`shapes_parameters` must be provided.")

        if "type" not in shapes_parameters:
            raise ValueError("`shapes_parameters` must contain a `type`.")
        # loosely follows coordinateTransform specs from ngff
        if "coordinateTransformations" in shapes_parameters:
            coordinate_transformations = shapes_parameters["coordinateTransformations"]
            if TYPE_CHECKING:
                assert isinstance(coordinate_transformations, list)
            if len(coordinate_transformations) != 1:
                raise ValueError("`coordinate_transformations` must be a list of one list.")
            transforms = coordinate_transformations[0]
            if "type" not in coordinate_transformations:
                raise ValueError("`coordinate_transformations` must contain a `type`.")

            if transforms["type"] != "scale":
                raise ValueError("`coordinate_transformations` must contain a `type` of `scale`.")
            if "scale" not in transforms:
                raise ValueError("`coordinate_transformations` must contain a `scale`.")
            if len(transforms["scale"]):
                raise ValueError("`coordinate_transformations` must contain a `scale` of length 0.")

    def validate_tables(
        self,
        tables: AnnData,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> None:
        if not isinstance(tables, AnnData):
            raise ValueError("`tables` must be an `anndata.AnnData`.")
        if region_key is not None:
            if not is_categorical_dtype(tables.obs[region_key]):
                raise ValueError(
                    f"`tables.obs[region_key]` must be of type `categorical`, not `{type(tables.obs[region_key])}`."
                )
        if instance_key is not None:
            if tables.obs[instance_key].isnull().values.any():
                raise ValueError("`tables.obs[instance_key]` must not contain null values, but it does.")
