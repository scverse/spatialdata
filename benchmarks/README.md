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
[100.00%] ··· ======== ============ ============= ============= ==============
              --                filter_table / n_transcripts_per_cell
              -------- -------------------------------------------------------
               length   True / 100   True / 1000   False / 100   False / 1000
              ======== ============ ============= ============= ==============
                100      177±5ms       195±4ms      168±0.5ms      186±2ms
                1000     195±3ms       402±2ms       187±3ms       374±4ms
               10000     722±3ms      2.65±0.01s     389±3ms      2.22±0.02s
              ======== ============ ============= ============= ==============
```

run everything in new env

```
asv run
```
