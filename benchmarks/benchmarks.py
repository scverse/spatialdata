# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import spatialdata as sd

# class TimeSuite:
#     """
#     An example benchmark that times the performance of various kinds
#     of iterating over dictionaries in Python.
#     """
#     def setup(self):
#         self.d = {}
#         for x in range(500):
#             self.d[x] = None

#     def time_keys(self):
#         for key in self.d.keys():
#             pass

#     def time_values(self):
#         for value in self.d.values():
#             pass

#     def time_range(self):
#         d = self.d
#         for key in range(500):
#             d[key]

# class MemSuite:
#     def mem_list(self):
#         sdata: sd.SpatialData = sd.datasets.blobs()
#         return sdata

class SpatialBlobsSuite:
    def peakmem_list(self):
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=1)
        return sdata

    def peakmem_list2(self):
        sdata: sd.SpatialData = sd.datasets.blobs(n_channels=2)
        return sdata


def timeraw_import_inspect():
    return """
    import spatialdata
    """

class SpatialDataLoading:

    params = [100, 200, 300]
    param_names = ["length"]

    def setup(self, length):
        self.sdata = sd.datasets.blobs(length=length)

    def teardown(self, _):
        del self.sdata

    def time_map_blocks(self, _):
        sd.map_raster(self.sdata["blobs_image"], lambda x: x+1)

