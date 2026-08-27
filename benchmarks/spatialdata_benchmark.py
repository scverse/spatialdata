# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from spatialdata import bounding_box_query
from spatialdata.transformations import Scale, set_transformation
import spatialdata as sd
from spatialdata.utils.points import _make_points

from .utils import cluster_blobs  # type: ignore[attr-defined] # utils is a type checker minefield
import numpy as np


class MemorySpatialData:
    # TODO: see what the memory overhead is e.g. Python interpreter...
    """Calculate the peak memory usage is for artificial datasets with increasing channels."""

    def peakmem_list(self) -> sd.SpatialData:
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=1)
        return sdata

    def peakmem_list2(self) -> sd.SpatialData:
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=2)
        return sdata


class TimeMapRaster:
    """Time the."""

    params = [100, 1000, 10_000]
    param_names = ["length"]

    def setup(self, length: int) -> None:
        self.sdata = cluster_blobs(length=length)

    def teardown(self, _length: int) -> None:
        del self.sdata

    def time_map_blocks(self, _length: int) -> None:
        sd.map_raster(self.sdata["blobs_image"], lambda x: x + 1)


class TimeQueries:
    params = ([100, 1_000, 10_000], [True, False], [100, 1_000])
    param_names = ["length", "filter_table", "n_transcripts_per_cell"]

    def setup(self, length: int, _filter_table: bool, n_transcripts_per_cell: bool) -> None:
        import shapely

        self.sdata = cluster_blobs(length=length, n_transcripts_per_cell=n_transcripts_per_cell)
        self.polygon = shapely.box(0, 0, length // 2, length // 2)

    def teardown(self, _length: int, _filter_table: bool, _n_transcripts_per_cell: bool) -> None:
        del self.sdata

    def time_query_bounding_box(self, length: int, filter_table: bool, _n_transcripts_per_cell: bool) -> None:
        self.sdata.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[0, 0],
            max_coordinate=[length // 2, length // 2],
            target_coordinate_system="global",
            filter_table=filter_table,
        )

    def time_query_polygon_box(self, _length: int, filter_table: bool, _n_transcripts_per_cell: bool) -> None:
        sd.polygon_query(
            self.sdata,
            self.polygon,
            target_coordinate_system="global",
            filter_table=filter_table,
        )


class TimeQueriesWithScaleTransformations:
    params = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    param_names = ["n_repeats"]

    def setup(self, n_repeats: int) -> None:
        coordinates = np.array(
            [
                [10.0, 10.0, 1.0],
                [70.0, 30.0, 2.0],
                [100.0, 50.0, 3.0],
                [150.0, 70.0, 4.0],
                [220.0, 90.0, 5.0],
                [10.0, -10.0, 1.0],
                [70.0, -30.0, 2.0],
                [100.0, -50.0, 3.0],
                [150.0, -70.0, 4.0],
                [220.0, -90.0, 5.0],
            ]
            * n_repeats
        )

        self.points_element = _make_points(coordinates)
        scale_x, scale_y = (1.1, 1)
        scale = Scale([scale_x, scale_y], axes=("x", "y"))
        set_transformation(self.points_element, transformation=scale, to_coordinate_system="global")

    def time_bbquery_scale_transform(self, n_repeats: int) -> None:

        x_min, x_max = 60.0, 240.0
        y_min, y_max = 20.0, 160.0

        _result_xy = bounding_box_query(
            self.points_element,
            axes=("x", "y"),
            min_coordinate=[x_min, y_min],
            max_coordinate=[x_max, y_max],
            target_coordinate_system="global",
        )
