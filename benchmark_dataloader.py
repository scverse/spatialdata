"""
Benchmark for ImageTilesDataset: init time + iteration time.

Usage:
    python benchmark_dataloader.py

Measures two phases:
  1. Init  — constructing ImageTilesDataset (includes bounding-box pre-computation).
  2. Fetch — iterating over every tile once (pure __getitem__ calls, no DataLoader overhead).

Designed to run identically on `main` and the `giovp/dataloader3` branch so the two
numbers can be compared directly.
"""

# ruff: noqa: T201

from __future__ import annotations

import time

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import spatialdata as sd
from spatialdata.dataloader import ImageTilesDataset
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

IMAGE_SIZE = 2048  # pixels (square)
N_CIRCLES = 500  # number of region instances / tiles
N_CHANNELS = 3


def make_sdata(n: int = N_CIRCLES, img_size: int = IMAGE_SIZE) -> sd.SpatialData:
    """Build an in-memory SpatialData with a large image and N circle regions."""
    # Image: random uint8, shape (C, H, W)
    img_data = RNG.integers(0, 256, size=(N_CHANNELS, img_size, img_size), dtype=np.uint8).astype(np.float32)
    image = Image2DModel.parse(
        img_data,
        dims=["c", "y", "x"],
        transformations={"global": Identity()},
    )

    # Circles: random centres, fixed radius so each tile is ~64 pixels wide
    radius = 32.0
    cx = RNG.uniform(radius, img_size - radius, size=n)
    cy = RNG.uniform(radius, img_size - radius, size=n)
    geom = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in zip(cx, cy, strict=True)]})
    geom["radius"] = radius
    circles = ShapesModel.parse(geom, transformations={"global": Identity()})

    # Table: one row per circle
    table = ad.AnnData(
        RNG.random((n, 10)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "region": pd.Categorical(["circles"] * n),
                "instance_id": np.arange(n, dtype=np.int64),
            },
            index=[str(i) for i in range(n)],
        ),
    )
    table = TableModel.parse(table, region="circles", region_key="region", instance_key="instance_id")

    return sd.SpatialData(
        images={"image": image},
        shapes={"circles": circles},
        tables={"table": table},
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def bench_init(sdata: sd.SpatialData, n_reps: int = 5) -> float:
    """Time ImageTilesDataset construction."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _ = ImageTilesDataset(
            sdata=sdata,
            regions_to_images={"circles": "image"},
            regions_to_coordinate_systems={"circles": "global"},
            table_name="table",
            return_annotations="instance_id",
        )
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_fetch(ds: ImageTilesDataset, n_reps: int = 3) -> float:
    """Time iterating over every item in the dataset."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        for i in range(len(ds)):
            _ = ds[i]
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all benchmark phases and print a timing summary."""
    import spatialdata

    print(f"spatialdata version : {spatialdata.__version__}")
    print(f"Image size          : {IMAGE_SIZE}×{IMAGE_SIZE}  ({N_CHANNELS} channels)")
    print(f"Circles (tiles)     : {N_CIRCLES}")
    print()

    print("Building synthetic SpatialData …", flush=True)
    t0 = time.perf_counter()
    sdata = make_sdata()
    print(f"  done in {time.perf_counter() - t0:.2f} s\n")

    print("Benchmarking init (5 reps) …", flush=True)
    t_init = bench_init(sdata, n_reps=5)
    print(f"  median init time : {t_init * 1000:.1f} ms\n")

    # Build one dataset for the fetch benchmark
    ds = ImageTilesDataset(
        sdata=sdata,
        regions_to_images={"circles": "image"},
        regions_to_coordinate_systems={"circles": "global"},
        table_name="table",
        return_annotations="instance_id",
    )
    print(f"Benchmarking fetch of {len(ds)} tiles (3 reps) …", flush=True)
    t_fetch = bench_fetch(ds, n_reps=3)
    per_tile_us = t_fetch / len(ds) * 1e6
    print(f"  median fetch time : {t_fetch * 1000:.1f} ms total  ({per_tile_us:.0f} µs/tile)")


if __name__ == "__main__":
    main()
