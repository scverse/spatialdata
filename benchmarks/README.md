# Benchmarking

setup

```
pip install -e '.[docs,benchmark]'
```

In PyCharm, configure your Configuration to include the benchmark module. In Python, you can run using

```
python -m benchmarks.spatialdata_benchmark
```

See napari [docs](https://napari.org/stable/developers/contributing/performance/benchmarks.html) on profiling and benchmarking for more information.

run a specific benchmark

```
PYTHONWARNINGS="ignore" asv run --python=same --show-stderr -b time_query_bounding_box
```

output:

```
[100.00%] ··· ======== ============ ============
              --              filter_table
              -------- -------------------------
               length      True        False
              ======== ============ ============
                100      191±2ms      185±2ms
                1000     399±4ms      382±7ms
               10000    2.67±0.02s   2.18±0.01s
              ======== ============ ============
```

run everything in new env

```
asv run
```
