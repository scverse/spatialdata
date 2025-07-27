# type: ignore

# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import spatialdata as sd

from .utils import cluster_blobs


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
        sd.map_raster(self.sdata["blobs_image"], lambda x: x + 1)


class TimeQueries:
    params = ([100, 1_000, 10_000], [True, False], [100, 1_000])
    param_names = ["length", "filter_table", "n_transcripts_per_cell"]

    def setup(self, length, filter_table, n_transcripts_per_cell):
        import shapely

        self.sdata = cluster_blobs(length=length, n_transcripts_per_cell=n_transcripts_per_cell)
        self.polygon = shapely.box(0, 0, length // 2, length // 2)

    def teardown(self, length, filter_table, n_transcripts_per_cell):
        del self.sdata

    def time_query_bounding_box(self, length, filter_table, n_transcripts_per_cell):
        self.sdata.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[0, 0],
            max_coordinate=[length // 2, length // 2],
            target_coordinate_system="global",
            filter_table=filter_table,
        )

    def time_query_polygon_box(self, length, filter_table, n_transcripts_per_cell):
        sd.polygon_query(
            self.sdata,
            self.polygon,
            target_coordinate_system="global",
            filter_table=filter_table,
        )


class TimeGeopandasQuery:
    params = (
        [100, 1000, 10000],  # TODO: test for larger number of points
        ["geopandas", "dask_geopandas"],
    )
    param_names = ["num_objects", "lib"]
    query_size = 100
    partition_size = 100  # TODO: expose npartitions as benchmark parameter

    def setup(self, num_objects, lib):
        # The point / points to query
        self.query_points = self._create_random_points(self.query_size)
        # Geometry
        # TODO: Test clustered points (not grid), and polygons
        geometry = self._create_regular_grid(num_objects=num_objects)
        if lib == "geopandas":
            import geopandas as gpd
            from geopandas.sindex import SpatialIndex

            self.df = gpd.GeoDataFrame(geometry=geometry)
            sindex: SpatialIndex = self.df.sindex
            self.nearest = sindex.nearest
            self.query = sindex.query
        elif lib == "dask_geopandas":
            import geopandas as gpd
            import dask_geopandas

            gdf = gpd.GeoDataFrame(geometry=geometry)
            npartitions = max(1, int(len(gdf) / self.partition_size))
            self.df = dask_geopandas.from_geopandas(gdf, npartitions=npartitions)
            # TODO: Instead, save gdf to tempfile and read with dask_geopandas.read_parquet
            #  to test larger-than-memory datasets.

            self.nearest = self.df.sindex.nearest
            self.query = self.df.sindex.query

    def _create_regular_grid(self, num_objects):
        import numpy as np
        from shapely.geometry import Point

        n_x = int(np.ceil(np.sqrt(num_objects)))
        coordinates_x = np.linspace(0.0, 1.0, n_x)
        coordinates = np.asarray(np.meshgrid(coordinates_x, coordinates_x)).T.reshape((-1, 2))
        return [Point(x, y) for y, x in coordinates[:num_objects]]

    def _create_random_points(self, num_points):
        import numpy as np
        from shapely.geometry import Point

        return [Point(x, y) for y, x in np.random.rand(num_points, 2)]

    def time_geopandas_nearest_point_point(self, num_objects, lib):
        self.nearest(self.query_points, return_distance=True)

    def time_geopandas_query_point_point(self, num_objects, lib):
        self.query(self.query_points)
