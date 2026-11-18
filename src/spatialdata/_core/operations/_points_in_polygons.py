"""Point-in-polygon join with a uniform grid and the crossing-number test, compiled with numba.

Semantics match shapely's `intersects`: a point on a polygon's boundary (outer ring or hole) counts as inside, a point
inside a hole does not, and a point in several overlapping polygons is returned once for each polygon.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import shapely

from spatialdata._types import ArrayLike


@nb.njit(cache=True)
def _build_grid(bounds: np.ndarray, x0: float, y0: float, cs: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(nx * ny + 1, np.int64)
    for p in range(bounds.shape[0]):
        i0 = max(int((bounds[p, 0] - x0) / cs), 0)
        i1 = min(int((bounds[p, 2] - x0) / cs), nx - 1)
        j0 = max(int((bounds[p, 1] - y0) / cs), 0)
        j1 = min(int((bounds[p, 3] - y0) / cs), ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                counts[j * nx + i + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    items = np.empty(offsets[-1], np.int64)
    for p in range(bounds.shape[0]):
        i0 = max(int((bounds[p, 0] - x0) / cs), 0)
        i1 = min(int((bounds[p, 2] - x0) / cs), nx - 1)
        j0 = max(int((bounds[p, 1] - y0) / cs), 0)
        j1 = min(int((bounds[p, 3] - y0) / cs), ny - 1)
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                items[fill[j * nx + i]] = p
                fill[j * nx + i] += 1
    return offsets, items


@nb.njit(inline="always")
def _covers(px: float, py: float, coords: np.ndarray, ring_off: np.ndarray, poly_off: np.ndarray, p: int) -> bool:
    """Point in polygon p or on its boundary (even-odd over all rings, so holes work)."""
    inside = False
    for r in range(poly_off[p], poly_off[p + 1]):
        a, b = ring_off[r], ring_off[r + 1]
        for k in range(a, b - 1):  # rings are closed: last coord == first
            x1, y1 = coords[k, 0], coords[k, 1]
            x2, y2 = coords[k + 1, 0], coords[k + 1, 1]
            # on the segment: collinear and inside its bounding box
            if (
                (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1) == 0.0
                and min(x1, x2) <= px <= max(x1, x2)
                and min(y1, y2) <= py <= max(y1, y2)
            ):
                return True
            if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


@nb.njit(nogil=True, cache=True)
def _join(
    x: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    coords: np.ndarray,
    ring_off: np.ndarray,
    poly_off: np.ndarray,
    x0: float,
    y0: float,
    cs: float,
    nx: int,
    ny: int,
    g_off: np.ndarray,
    g_items: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First matching polygon per point (-1 if none) and number of matches."""
    n = x.shape[0]
    first = np.full(n, -1, np.int64)
    nhit = np.zeros(n, np.int32)
    for i in range(n):
        px, py = x[i], y[i]
        gi, gj = int((px - x0) / cs), int((py - y0) / cs)
        if px < x0 or py < y0 or gi >= nx or gj >= ny:
            continue
        c = gj * nx + gi
        for t in range(g_off[c], g_off[c + 1]):
            p = g_items[t]
            if px < bounds[p, 0] or px > bounds[p, 2] or py < bounds[p, 1] or py > bounds[p, 3]:
                continue
            if _covers(px, py, coords, ring_off, poly_off, p):
                if nhit[i] == 0:
                    first[i] = p
                nhit[i] += 1
    return first, nhit


@nb.njit(nogil=True, cache=True)
def _all_hits(
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounds: np.ndarray,
    coords: np.ndarray,
    ring_off: np.ndarray,
    poly_off: np.ndarray,
    x0: float,
    y0: float,
    cs: float,
    nx: int,
    ny: int,
    g_off: np.ndarray,
    g_items: np.ndarray,
    total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """All (point, polygon) pairs for the few points that hit more than one polygon."""
    pts = np.empty(total, np.int64)
    polys = np.empty(total, np.int64)
    m = 0
    for i in idx:
        px, py = x[i], y[i]
        c = int((py - y0) / cs) * nx + int((px - x0) / cs)
        for t in range(g_off[c], g_off[c + 1]):
            p = g_items[t]
            if (
                bounds[p, 0] <= px <= bounds[p, 2]
                and bounds[p, 1] <= py <= bounds[p, 3]
                and _covers(px, py, coords, ring_off, poly_off, p)
            ):
                pts[m], polys[m] = i, p
                m += 1
    return pts[:m], polys[:m]


class PolygonIndex:
    """
    Spatial index of polygons for joining points without creating shapely point geometries.

    The kernels are compiled with numba and release the GIL, so several threads can query the same index at once.
    """

    def __init__(self, geoms: ArrayLike) -> None:
        parts, self.region = shapely.get_parts(np.asarray(geoms), return_index=True)  # multipolygons -> polygons
        _, coords, (self.ring_off, self.poly_off) = shapely.to_ragged_array(parts)
        self.coords = np.ascontiguousarray(coords[:, :2], np.float64)
        self.bounds = shapely.bounds(parts)
        self.x0, self.y0 = self.bounds[:, 0].min(), self.bounds[:, 1].min()
        w = self.bounds[:, 2:] - self.bounds[:, :2]
        self.cs = float(max(np.median(w), 1e-9))  # about one polygon per grid cell
        self.nx = int((self.bounds[:, 2].max() - self.x0) / self.cs) + 1
        self.ny = int((self.bounds[:, 3].max() - self.y0) / self.cs) + 1
        self.g_off, self.g_items = _build_grid(self.bounds, self.x0, self.y0, self.cs, self.nx, self.ny)

    def query(self, x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Return (point index, geometry index) pairs, like `STRtree.query(points, predicate="intersects")`."""
        x, y = np.ascontiguousarray(x, np.float64), np.ascontiguousarray(y, np.float64)
        a = (
            self.bounds,
            self.coords,
            self.ring_off,
            self.poly_off,
            self.x0,
            self.y0,
            self.cs,
            self.nx,
            self.ny,
            self.g_off,
            self.g_items,
        )
        first, nhit = _join(x, y, *a)
        single = np.flatnonzero(nhit == 1)
        multi = np.flatnonzero(nhit > 1)
        mp, mq = _all_hits(multi, x, y, *a, int(nhit[multi].sum()))
        pts = np.concatenate([single, mp])
        return pts, self.region[np.concatenate([first[single], mq])]
