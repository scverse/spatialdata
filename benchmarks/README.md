# Benchmarking

setup
```
pip install -e '.[docs,benchmark]'
```

See napari [docs](https://napari.org/stable/developers/contributing/performance/benchmarks.html) on profiling and benchmarking for more information.

run a specific benchmark
```
PYTHONWARNINGS="ignore" asv run --python=same --show-stderr --quick -b time_query_bounding_box 
```
output:
```
[100.00%] ··· ======== ========== ============
              --             filter_table     
              -------- -----------------------
               length     True       False    
              ======== ========== ============
                100     89.1±5ms   85.6±0.8ms 
                1000    99.0±8ms    87.7±1ms  
               10000    427±10ms    92.4±2ms  
              ======== ========== ============
```

run everything in new env
```
asv run
```
