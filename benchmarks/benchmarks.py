# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import spatialdata as sd

try:
    from .utils import cluster_blobs
except ImportError:
    # TODO: remove ugly hack used for local testing
    from utils import cluster_blobs


class MemorySpatialData:
    # TODO: see what the memory overhead is e.g. Python interpreter...
    """Calculate the peak memory usage is for artificial datasets with increasing channels."""

    def peakmem_list(self):
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=1)
        return sdata

    def peakmem_list2(self):
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=2)
        return sdata


def timeraw_import_inspect():
    """Time the import of the spatialdata module."""
    return """
    import spatialdata
    """

class TimeMapRaster:
    """Time the."""

    params = [100, 1000, 10_000]
    param_names = ["length"]

    def setup(self, length):
        self.sdata = cluster_blobs(length=length)

    def teardown(self, _):
        del self.sdata

    def time_map_blocks(self, _):
        sd.map_raster(self.sdata["blobs_image"], lambda x: x+1)

class TimeQueries:

    params = ([100, 1000, 10_000], [True, False])
    param_names = ["length", "filter_table"]

    def setup(self, length, filter_table):
        import shapely

        self.sdata = cluster_blobs(length=length)
        self.polygon = shapely.box(0, 0, length//2, length//2)


    def teardown(self, length, filter_table):
        del self.sdata

    def time_query_bounding_box(self, length, filter_table):
        self.sdata.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[0, 0],
            max_coordinate=[length//2, length//2],
            target_coordinate_system="global",
            filter_table=filter_table,
        )

    def time_query_polygon_box(self, length, filter_table):
        sd.polygon_query(self.sdata, self.polygon, target_coordinate_system="global",
                         filter_table=filter_table,
        )


if __name__ == "__main__":
    length = 10_000
    sdata = cluster_blobs(length)
    # sdata.write("tmp_test")
    sdata.query.bounding_box(
        axes=["x", "y"],
        min_coordinate=[0, 0],
        max_coordinate=[length//2, length//2],
        target_coordinate_system="global",
        filter_table=True,
    )
    print(sdata)
