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
[50.00%] ··· ======== ============ ============== ============= ===============
         	--             	filter_table / n_transcripts_per_cell
         	-------- ---------------------------------------------------------
          	length   True / 100   True / 10000   False / 100   False / 10000
         	======== ============ ============== ============= ===============
           	100  	813±0ms   	1.09±0s    	803±0ms    	980±0ms
           	1000 	799±0ms   	2.96±0s    	789±0ms    	2.81±0s
          	10000 	1.32±0s   	24.4±0s    	962±0ms    	21.5±0s
         	======== ============ ============== ============= ===============
```

run everything in new env

```
asv run
```
