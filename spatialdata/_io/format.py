from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData
from ome_zarr.format import CurrentFormat
from pandas.api.types import is_categorical_dtype

CoordinateTransform_t = List[Dict[str, Any]]


class SpatialDataFormat(CurrentFormat):
    """
    SpatialDataFormat defines the format of the spatialdata
    package.
    """

    # TODO: old code, to be adjusted (since coordinate transformations are validated separately now)
    # def validate_shapes_parameters(
    #     self,
    #     shapes_parameters: Dict[str, Union[str, float, Dict[str, CoordinateTransform_t]]],
    # ) -> None:
    #     """
    #     Validate the shape parameters.
    #
    #     Parameters
    #     ----------
    #     shapes_parameters
    #         Shape parameters.
    #
    #     Returns
    #     ------
    #         Nothing.
    #     """
    #     if shapes_parameters is None:
    #         raise ValueError("`shapes_parameters` must be provided.")
    #
    #     if "type" not in shapes_parameters:
    #         raise ValueError("`shapes_parameters` must contain a `type`.")
    #     # loosely follows coordinateTransform specs from ngff
    #     if "coordinateTransformations" in shapes_parameters:
    #         coordinate_transformations = shapes_parameters["coordinateTransformations"]
    #         if TYPE_CHECKING:
    #             assert isinstance(coordinate_transformations, list)
    #         if len(coordinate_transformations) != 1:
    #             raise ValueError("`coordinate_transformations` must be a list of one list.")
    #         transforms = coordinate_transformations[0]
    #         if "type" not in coordinate_transformations:
    #             raise ValueError("`coordinate_transformations` must contain a `type`.")
    #
    #         if transforms["type"] != "scale":
    #             raise ValueError("`coordinate_transformations` must contain a `type` of `scale`.")
    #         if "scale" not in transforms:
    #             raise ValueError("`coordinate_transformations` must contain a `scale`.")
    #         if len(transforms["scale"]):
    #             raise ValueError("`coordinate_transformations` must contain a `scale` of length 0.")

    def validate_tables(
        self,
        tables: AnnData,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> None:
        # TODO: this code is not used atm, but should be used in the future for validation
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

    def generate_coordinate_transformations(self, shapes: List[Tuple[Any]]) -> Optional[List[List[Dict[str, Any]]]]:

        data_shape = shapes[0]
        coordinate_transformations: List[List[Dict[str, Any]]] = []
        # calculate minimal 'scale' transform based on pyramid dims
        for shape in shapes:
            assert len(shape) == len(data_shape)
            scale = [full / level for full, level in zip(data_shape, shape)]
            from spatialdata._core.transform import Scale

            coordinate_transformations.append([Scale(scale=scale).to_dict()])
        return coordinate_transformations

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Validates that a list of dicts contains a 'scale' transformation

        Raises ValueError if no 'scale' found or doesn't match ndim
        :param ndim:       Number of image dimensions
        """

        if coordinate_transformations is None:
            raise ValueError("coordinate_transformations must be provided")
        ct_count = len(coordinate_transformations)
        if ct_count != nlevels:
            raise ValueError(f"coordinate_transformations count: {ct_count} must match datasets {nlevels}")
        for transformations in coordinate_transformations:
            assert isinstance(transformations, list)
            types = [t.get("type", None) for t in transformations]
            if any([t is None for t in types]):
                raise ValueError("Missing type in: %s" % transformations)

            # new validation
            import json

            json0 = [json.dumps(t) for t in transformations]
            from spatialdata._core.transform import get_transformation_from_dict

            parsed = [get_transformation_from_dict(t) for t in transformations]
            json1 = [p.to_json() for p in parsed]
            import numpy as np

            assert np.all([j0 == j1 for j0, j1 in zip(json0, json1)])

            # old validation
            # validate scales...
            # if sum(t == "scale" for t in types) != 1:
            #     raise ValueError("Must supply 1 'scale' item in coordinate_transformations")
            # # first transformation must be scale
            # if types[0] != "scale":
            #     raise ValueError("First coordinate_transformations must be 'scale'")
            # first = transformations[0]
            # if "scale" not in transformations[0]:
            #     raise ValueError("Missing scale argument in: %s" % first)
            # scale = first["scale"]
            # if len(scale) != ndim:
            #     raise ValueError(f"'scale' list {scale} must match number of image dimensions: {ndim}")
            # for value in scale:
            #     if not isinstance(value, (float, int)):
            #         raise ValueError(f"'scale' values must all be numbers: {scale}")
            #
            # # validate translations...
            # translation_types = [t == "translation" for t in types]
            # if sum(translation_types) > 1:
            #     raise ValueError("Must supply 0 or 1 'translation' item in" "coordinate_transformations")
            # elif sum(translation_types) == 1:
            #     transformation = transformations[types.index("translation")]
            #     if "translation" not in transformation:
            #         raise ValueError("Missing scale argument in: %s" % first)
            #     translation = transformation["translation"]
            #     if len(translation) != ndim:
            #         raise ValueError(f"'translation' list {translation} must match image dimensions count: {ndim}")
            #     for value in translation:
            #         if not isinstance(value, (float, int)):
            #             raise ValueError(f"'translation' values must all be numbers: {translation}")
