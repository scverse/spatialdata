---
name: memray
description: Profile the memory usage of a Python script using memray and visualize a temporal flamegraph in the browser. Use when the user wants to investigate memory consumption, find leaks, or understand allocation patterns.
compatibility: Requires the pixi profiling environment (pixi run -e profiling). Supports Linux and macOS.
allowed-tools: Bash(pixi run -e profiling memray-run:*) Bash(pixi run -e profiling memray-flame:*) Bash(open:*) Bash(xdg-open:*) Bash(python -m webbrowser:*)
---

## Steps

1. Ask the user which script to profile (full or relative path).

2. Run the script under memray:

    ```bash
    pixi run -e profiling memray-run script.py
    ```

    This produces a binary file named `memray-script.py.<pid>.bin` in the current directory.

3. Generate the flamegraph HTML report from the `.bin` file:

    ```bash
    pixi run -e profiling memray-flame memray-script.py.<pid>.bin
    ```

    Replace `<pid>` with the actual PID shown in the filename. This writes `memray-flamegraph-script.py.<pid>.html`.

4. Open the report in the browser:
    - macOS: `open memray-flamegraph-script.py.<pid>.html`
    - Linux: `xdg-open memray-flamegraph-script.py.<pid>.html`
    - Either: `python -m webbrowser memray-flamegraph-script.py.<pid>.html`

## Notes

- The `--temporal` flag (included in `memray-flame`) shows memory over time, not just peak — use this to spot leaks and allocation bursts.
- To find the `.bin` file if unsure of the name: `ls memray-*.bin`
- To compare runs, save the previous report: `cp memray-flamegraph-script.py.<pid>.html memray-flamegraph-before.html`
