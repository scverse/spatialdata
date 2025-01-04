# Benchmarking SpatialData code

This `benchmarks/` folder contains code to benchmark the performance of the SpatialData code. You can use it to see how code behaves for different options or data sizes. For more information, check the [SpatialData Contributing Guide](https://spatialdata.scverse.org/en/stable/contributing.html).

Note that to run code, your current working directory should be the SpatialData repo, not this `benchmarks/` folder.

## Installation

The benchmarks use the [airspeed velocity](https://asv.readthedocs.io/en/stable/) (asv) framework. Install it with the `benchmark` option:

```
pip install -e '.[docs,test,benchmark]'
```

## Usage

Running all the benchmarks is usually not needed. You run the benchmark using `asv run`. See the [asv documentation](https://asv.readthedocs.io/en/stable/commands.html#asv-run) for interesting arguments, like selecting the benchmarks you're interested in by providing a regex pattern `-b` or `--bench` that links to a function or class method e.g. the option `-b timeraw_import_inspect` selects the function `timeraw_import_inspect` in `benchmarks/spatialdata_benchmark.py`. You can run the benchmark in your current environment with `--python=same`. Some example benchmarks:

Importing the SpatialData library can take around 4 seconds:

```
PYTHONWARNINGS="ignore" asv run --python=same --show-stderr -b timeraw_import_inspect
Couldn't load asv.plugins._mamba_helpers because
No module named 'conda'
· Discovering benchmarks
· Running 1 total benchmarks (1 commits * 1 environments * 1 benchmarks)
[ 0.00%] ·· Benchmarking existing-py_opt_homebrew_Caskroom_mambaforge_base_envs_spatialdata2_bin_python3.12
[50.00%] ··· Running (spatialdata_benchmark.timeraw_import_inspect--).
[100.00%] ··· spatialdata_benchmark.timeraw_import_inspect                                                                            3.65±0.2s
```

Querying using a bounding box without a spatial index is highly impacted by large amounts of points (transcripts), more than table rows (cells).

```
$ PYTHONWARNINGS="ignore" asv run --python=same --show-stderr -b time_query_bounding_box

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

You can use `asv` to run all the benchmarks in their own environment. This can take a long time, so it is not recommended for regular use:

```
$ asv run
Couldn't load asv.plugins._mamba_helpers because
No module named 'conda'
· Creating environments....
· Discovering benchmarks..
·· Uninstalling from virtualenv-py3.12
·· Building a89d16d8 <v0.2.6-pre0~7> for virtualenv-py3.12
·· Installing a89d16d8 <v0.2.6-pre0~7> into virtualenv-py3.12.............
· Running 6 total benchmarks (1 commits * 1 environments * 6 benchmarks)
[ 0.00%] · For spatialdata commit a89d16d8 <v0.2.6-pre0~7>:
[ 0.00%] ·· Benchmarking virtualenv-py3.12
[25.00%] ··· Running (spatialdata_benchmark.TimeMapRaster.time_map_blocks--)...
...
[100.00%] ··· spatialdata_benchmark.timeraw_import_inspect                                                                                                                                    3.33±0.06s
```

## Notes

When using PyCharm, remember to set [Configuration](https://www.jetbrains.com/help/pycharm/run-debug-configuration.html) to include the benchmark module, as this is separate from the main code module.

In Python, you can run a module using the following command:

```
python -m benchmarks.spatialdata_benchmark
```
