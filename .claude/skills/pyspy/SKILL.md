---
name: pyspy
description: Profile the execution time of a Python script using py-spy and visualize the result with speedscope. Use when the user wants to benchmark performance, find slow code paths, or profile CPU time.
compatibility: Requires the pixi profiling environment (pixi run -e profiling). Speedscope must be installed separately (npm install -g speedscope). sudo is required on macOS.
allowed-tools: Bash(pixi run -e profiling pyspy:*) Bash(pixi run -e profiling speedscope:*) Bash(sudo pixi run -e profiling pyspy:*)
---

## Steps

1. Ask the user which script to profile (full or relative path).

2. Run py-spy to record the profile. The output is always written to `profile.speedscope.json` in the current directory.

    **Linux** (no sudo needed):

    ```bash
    pixi run -e profiling pyspy script.py
    ```

    **macOS** (sudo required — py-spy needs to attach to the process):

    ```bash
    sudo pixi run -e profiling pyspy script.py
    ```

    If sudo fails to find the pixi environment, use absolute paths:

    ```bash
    sudo /path/to/.pixi/envs/profiling/bin/py-spy record --gil \
      -o profile.speedscope.json --format speedscope \
      -- /path/to/.pixi/envs/profiling/bin/python script.py
    ```

3. Open the result in speedscope:
    ```bash
    pixi run -e profiling speedscope
    ```
    This opens `profile.speedscope.json` in the browser via the local speedscope CLI.

## Notes

- If the speedscope view is blank, switch threads using the thread selector in the top-right corner.
- To save a profile before overwriting: `cp profile.speedscope.json profile-before.speedscope.json`
- `--gil` records only time when the GIL is held (Python-level CPU time). Drop it to include C extension time.
- speedscope must be installed globally: `npm install -g speedscope`
