"""Point-in-polygon join on a uniform grid, compiled with numba, with no shapely point geometries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numba as nb
import numpy as np
import shapely

BIG = 500  # vertices; larger geometries are tested with prepared GEOS instead of the edge loop


def _jit(f: Callable[..., Any]) -> Any:
    try:
        return nb.njit(nogil=True, cache=True)(f)
    except RuntimeError:  # no writable cache directory
        return nb.njit(nogil=True)(f)


@_jit
def _covers(px: float, py: float, coords: np.ndarray, ring_off: np.ndarray, r0: int, r1: int) -> bool:
    """Point inside (even-odd over rings r0..r1, so holes work) or on an edge."""
    inside = False
    for r in range(r0, r1):
        for k in range(ring_off[r], ring_off[r + 1] - 1):  # rings are closed
            x1, y1, x2, y2 = coords[k, 0], coords[k, 1], coords[k + 1, 0], coords[k + 1, 1]
            if (
                (x2 - x1) * (py - y1) == (y2 - y1) * (px - x1)
                and min(x1, x2) <= px <= max(x1, x2)
                and min(y1, y2) <= py <= max(y1, y2)
            ):
                return True
            if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


@_jit
def _query(x: np.ndarray, y: np.ndarray, grid: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray]:
    bounds, coords, ring_off, geom_ring, big, x0, y0, cs, nx, ny, g_off, g_items = grid
    pts = np.empty(len(x) + 16, np.int64)
    geoms = np.empty(len(x) + 16, np.int64)
    m = 0
    for i in range(len(x)):
        px, py = x[i], y[i]
        fx, fy = (px - x0) / cs, (py - y0) / cs
        if not (0 <= fx < nx and 0 <= fy < ny):  # also skips NaN and inf
            continue
        c = int(fy) * nx + int(fx)
        for t in range(g_off[c], g_off[c + 1]):
            g = g_items[t]
            if not (bounds[g, 0] <= px <= bounds[g, 2] and bounds[g, 1] <= py <= bounds[g, 3]):
                continue
            # big geometries only become candidates here, PolygonIndex.query() tests them
            if big[g] or _covers(px, py, coords, ring_off, geom_ring[g], geom_ring[g + 1]):
                if m == len(pts):
                    pts, geoms = np.concatenate((pts, pts)), np.concatenate((geoms, geoms))
                pts[m], geoms[m] = i, g
                m += 1
    return pts[:m], geoms[:m]


class PolygonIndex:
    """
    Join points to (multi)polygons like `STRtree.query(points, predicate="intersects")`, from coordinate arrays.

    The kernel releases the GIL, so threads can share one index. Points on an edge count as inside; non-polygonal and
    empty geometries match nothing.
    """

    def __init__(self, geoms: np.ndarray) -> None:
        self.n_regions = len(geoms)
        parts, owner = shapely.get_parts(geoms, return_index=True)
        ok = (shapely.get_type_id(parts) == shapely.GeometryType.POLYGON) & ~shapely.is_empty(parts)
        parts, owner = parts[ok], owner[ok]
        self.geom = np.unique(owner)  # geometries with at least one polygon part
        if len(parts):
            _, coords, (ring_off, part_ring) = shapely.to_ragged_array(parts)
        else:
            coords, ring_off, part_ring = np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros(1, np.int64)
        # the parts of a geometry are contiguous, so its rings are too: one even-odd test over all of them
        first_part = np.searchsorted(owner, self.geom)
        geom_ring = np.append(part_ring[first_part], part_ring[-1]).astype(np.int64)
        n_vertices = np.diff(ring_off[geom_ring])
        self.big = np.flatnonzero(n_vertices > BIG)
        self.prepared = geoms[self.geom[self.big]]
        shapely.prepare(self.prepared)

        b = shapely.bounds(geoms[self.geom]).reshape(-1, 4)
        x0, y0 = (b[:, 0].min(), b[:, 1].min()) if len(b) else (0.0, 0.0)
        w, h = (b[:, 2].max() - x0, b[:, 3].max() - y0) if len(b) else (0.0, 0.0)
        n = max(len(b), 1)
        # about one geometry per cell, but never more cells than ~4x the geometries
        cs = max(float(np.median(b[:, 2:] - b[:, :2])) if len(b) else 0.0, np.sqrt(w * h / n), (w + h) / n, 1e-9)
        nx, ny = int(w / cs) + 1, int(h / cs) + 1
        i0, j0, i1, j1 = (((b - [x0, y0, x0, y0]) / cs).astype(np.int64)).T
        k = (i1 - i0 + 1) * (j1 - j0 + 1)  # grid cells covered by each geometry's bounding box
        g = np.repeat(np.arange(len(b)), k)
        local = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        cell = (j0[g] + local // (i1 - i0 + 1)[g]) * nx + i0[g] + local % (i1 - i0 + 1)[g]
        g_items = g[np.argsort(cell, kind="stable")]
        g_off = np.append(0, np.cumsum(np.bincount(cell, minlength=nx * ny)))
        big = np.zeros(len(b), np.bool_)
        big[self.big] = True
        coords = np.ascontiguousarray(coords[:, :2], np.float64)
        self.grid = (b, coords, ring_off.astype(np.int64), geom_ring, big, x0, y0, cs, nx, ny, g_off, g_items)

    def query(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (point index, geometry index) pairs."""
        x, y = np.ascontiguousarray(x, np.float64), np.ascontiguousarray(y, np.float64)
        pts, geoms = _query(x, y, self.grid)
        if len(self.big):
            keep = np.ones(len(pts), bool)
            for b, poly in zip(self.big, self.prepared, strict=True):
                cand = np.flatnonzero(geoms == b)
                keep[cand] = shapely.intersects_xy(poly, x[pts[cand]], y[pts[cand]])
            pts, geoms = pts[keep], geoms[keep]
        return pts, self.geom[geoms]
