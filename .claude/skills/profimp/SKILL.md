---
name: profimp
description: Profile Python import time using profimp and open a waterfall HTML report. Use when investigating slow startup or wanting to identify which imports are most expensive.
compatibility: Requires profimp (available as a pixi dependency). macOS or Linux.
allowed-tools: Bash(profimp:*) Bash(open:*) Bash(xdg-open:*) Bash(python -m webbrowser:*)
---

## Steps

1. Ask what to profile. Suggest common patterns for this repo:
    - `import spatialdata`
    - `from spatialdata import SpatialData`
    - `from spatialdata_io import xenium`

2. Run:

```bash
profimp --html "<import_stmt>" > /tmp/profimp.html
```

3. Open the report:
    - macOS: `open /tmp/profimp.html`
    - Linux: `xdg-open /tmp/profimp.html`
    - Either: `python -m webbrowser /tmp/profimp.html`

## Notes

- The report is a waterfall chart showing every sub-import and its timing.
- To compare before/after: `cp /tmp/profimp.html /tmp/profimp-before.html` before re-running.
- `pixi run python -m profimp` also works if `profimp` is not on PATH.
