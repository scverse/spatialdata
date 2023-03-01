import numpy as np
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import Labels2DModel
from spatialdata._core._spatial_query import bounding_box_query
from spatialdata._core._spatialdata_ops import (
    get_transformation,
    remove_transformation,
    set_transformation,
)
from spatialdata._core.transformations import Affine


def _visualize_crop_affine_labels_2d() -> None:
    """
    This examples show how the bounding box spatial query works for data that has been rotated.

    Notes
    -----
    The bounding box query gives the data, from the intrinsic coordinate system, that is inside the bounding box of
    the inverse-transformed query bounding box.
    In this example I show this data, and I also show how to obtain the data back inside the original bounding box.

    To undertand the example I suggest to run it and then:
    1) select the "rotated" coordinate system from napari
    2) disable all the layers but "0 original"
    3) then enable "1 cropped global", this shows the data in the extrinsic coordinate system we care ("rotated"),
    and the bounding box we want to query
    4) then enable "2 cropped rotated", this show the data that has been queries (this is a bounding box of the
    requested crop, as exaplained above)
    5) then enable "3 cropped rotated processed", this shows the data that we wanted to query in the first place,
    in the target coordinate system ("rotated"). This is probaly the data you care about if for instance you want to
    use tiles for deep learning. Note that for obtaning this answer there is also a better function (not available at
    the time of this writing): rasterize(), which is faster and more accurate, so it should be used instead. The
    function rasterize() transforms all the coordinates of the data into the target coordinate system, and it returns
    only SpatialImage objects. So it has different use cases than the bounding box query.
    6) finally switch to the "global" coordinate_system. This is, for how we constructed the example, showing the
    original image as it would appear its intrinsic coordinate system (since the transformation that maps the
    original image to "global" is an identity. It then shows how the data showed at the point 5), localizes in the
    original image.
    """
    ##
    # in this test let's try some affine transformations, we could do that also for the other tests
    image = np.random.randint(low=10, high=100, size=(100, 100))
    # y: [5, 9], x: [0, 4] has value 1
    image[50:, :50] = 2
    labels_element = Labels2DModel.parse(image)
    set_transformation(
        labels_element,
        Affine(
            np.array(
                [
                    [np.cos(np.pi / 6), np.sin(-np.pi / 6), 20],
                    [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                    [0, 0, 1],
                ]
            ),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        "rotated",
    )

    # bounding box: y: [5, 9], x: [0, 4]
    labels_result_rotated = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="rotated",
    )
    labels_result_global = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="global",
    )
    from napari_spatialdata import Interactive

    from spatialdata import SpatialData

    old_transformation = get_transformation(labels_result_global, "global")
    remove_transformation(labels_result_global, "global")
    set_transformation(labels_result_global, old_transformation, "rotated")
    d = {
        "1 cropped_global": labels_result_global,
        "0 original": labels_element,
    }
    if labels_result_rotated is not None:
        d["2 cropped_rotated"] = labels_result_rotated

        assert isinstance(labels_result_rotated, SpatialImage) or isinstance(
            labels_result_rotated, MultiscaleSpatialImage
        )
        transform = labels_result_rotated.attrs["transform"]["rotated"]
        transform_rotated_processed = transform.transform(labels_result_rotated, maintain_positioning=True)
        transform_rotated_processed_recropped = bounding_box_query(
            transform_rotated_processed,
            axes=("y", "x"),
            min_coordinate=np.array([25, 25]),
            max_coordinate=np.array([75, 100]),
            target_coordinate_system="rotated",
        )
        d["3 cropped_rotated_processed_recropped"] = transform_rotated_processed_recropped
        remove_transformation(labels_result_rotated, "global")

    sdata = SpatialData(labels=d)
    Interactive(sdata)
    ##


if __name__ == "__main__":
    _visualize_crop_affine_labels_2d()
