# SPDX-License-Identifier: GPL-3.0-or-later
"""Painted features, cut off along the sharp edge they stand on.

A painted region becomes its own part when it is a *feature*: where its paint border meets the
rest of the object, the surface turns sharply (a chimney standing on a roof). The feature is
cut off by a cap that continues the surrounding base surface under it, so the two parts meet
on one shared surface and no depth or clearance is involved.

Paint on a smooth surface (a stripe, a ring, a logo) has no edge to cut along. It stays on the
base and is reported; the slicer prints it in color as painted.

Rules for real, imperfect paint:

* Sharpness is measured over a ``WINDOW`` around each border point, not per triangle, so a
  rounded (filleted) foot still counts as a sharp edge.
* Paint that spills onto the neighboring face, or stops short of the edge, by up to ``SPILL``
  is snapped to the edge.
* The cap follows the base surface around the feature (a quadric fit), so a feature on a
  curved base is cut along the curve.
* Unpainted surfaces enclosed by a feature (a chimney's bore) belong to the feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import manifold3d
import numpy as np
from scipy.spatial import cKDTree

SHARP_DEG = 45.0  # a border turning at least this much is an edge to cut along
WINDOW = 1.0  # mm around a border point over which its angle is measured
SPILL = 0.5  # mm; paint this close to a sharp edge is snapped to it
SHARP_SHARE = 0.9  # share of a cut border (by length) that must be sharp
LIFT = 0.02  # mm the cap sits above the base surface, so no faces coincide
GROW = 0.5  # mm the cutter reaches past the border, through the air beside the feature
TOP_MARGIN = 0.05  # mm the cutter reaches above the feature's top (a bridge just above stays)
BAND = 3.0  # mm of base surface around the border that shapes the cap
MIN_FEATURE_VOLUME = 0.5  # mm^3 a feature must enclose with its cap
_WELD = 1e-6
_CELL = 0.1  # mm grid of the cap surface


# --- mesh helpers -----------------------------------------------------------------------

def conform(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weld leaf triangles (T,3,3) into (V, F, parent): T-junctions from paint subdivision are
    fanned around the triangle's centroid so every edge is shared by two triangles."""
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3, 3)

    def keys(p):
        return np.round(p / _WELD).astype(np.int64)

    def hashed(p):  # one 64-bit number per welded point: arrays, not a set of Python tuples
        k = keys(p).astype(np.uint64)
        h = k[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        h = (h ^ (h >> np.uint64(29))) + k[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
        h = (h ^ (h >> np.uint64(31))) + k[:, 2] * np.uint64(0x165667B19E3779F9)
        return h ^ (h >> np.uint64(32))
    known = np.unique(hashed(corners.reshape(-1, 3)))
    batch, bp = corners, np.arange(len(corners))
    out, out_parent = [], []
    while len(batch):
        mids = 0.5 * (batch + batch[:, [1, 2, 0]])
        h = hashed(mids.reshape(-1, 3))
        hit = (known[np.minimum(np.searchsorted(known, h), len(known) - 1)] == h).reshape(-1, 3)
        plain = ~hit.any(axis=1)
        out.append(batch[plain])
        out_parent.append(bp[plain])
        tri, mid, flags, p = batch[~plain], mids[~plain], hit[~plain], bp[~plain]
        if not len(tri):
            break
        # each triangle's outline: corner 0, its edge's midpoint if any, corner 1, ...
        ring = np.stack([tri[:, 0], mid[:, 0], tri[:, 1], mid[:, 1], tri[:, 2], mid[:, 2]], axis=1)
        always = np.ones(len(tri), dtype=bool)
        present = np.c_[always, flags[:, 0], always, flags[:, 1], always, flags[:, 2]]
        count = present.sum(axis=1)
        first = np.cumsum(count) - count
        owner = np.repeat(np.arange(len(tri)), count)
        following = np.arange(len(owner)) + 1
        following[first + count - 1] = first
        points = ring[present]
        centre = tri.mean(axis=1)
        known = np.union1d(known, hashed(centre))
        batch = np.stack([centre[owner], points, points[following]], axis=1)
        bp = p[owner]
    soup = np.concatenate(out) if out else np.zeros((0, 3, 3))
    points = soup.reshape(-1, 3)
    uniq, inverse = np.unique(hashed(points), return_inverse=True)
    inverse = inverse.reshape(-1)
    V = np.zeros((len(uniq), 3))
    V[inverse] = points
    if not np.array_equal(keys(V)[inverse], keys(points)):  # two points hashed alike
        uniq, inverse = np.unique(keys(points), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        V = np.zeros((len(uniq), 3))
        V[inverse] = points
    return V, inverse.reshape(-1, 3), np.concatenate(out_parent) if out_parent else np.zeros(0, int)


def components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    labels = np.arange(n)
    while len(a):
        low = np.minimum(labels[a], labels[b])
        before = labels.copy()
        np.minimum.at(labels, a, low)
        np.minimum.at(labels, b, low)
        labels = labels[labels]
        if np.array_equal(before, labels):
            break
    return labels


def edge_pairs(F: np.ndarray):
    """(tri a, tri b, edge vertices) for every interior edge."""
    e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=2).reshape(-1, 2)
    tri = np.repeat(np.arange(len(F)), 3)
    key = e[:, 0] * (int(F.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind="stable")
    same = key[order][1:] == key[order][:-1]
    return tri[order][:-1][same], tri[order][1:][same], e[order][:-1][same]


def normals_and_areas(V: np.ndarray, F: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Unit face normals (slivers take their neighbors' average) and areas."""
    c = V[F]
    n = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
    area2 = np.linalg.norm(n, axis=1)
    normals = n / np.maximum(area2, 1e-300)[:, None]
    longest = np.max(np.linalg.norm(c - c[:, [1, 2, 0]], axis=2), axis=1)
    reliable = area2 > 1e-3 * np.maximum(longest, 1e-12) ** 2
    for _ in range(5):
        if reliable.all():
            break
        acc = np.zeros_like(normals)
        np.add.at(acc, a, np.where(reliable[b, None], normals[b], 0.0))
        np.add.at(acc, b, np.where(reliable[a, None], normals[a], 0.0))
        length = np.linalg.norm(acc, axis=1)
        fix = ~reliable & (length > 1e-12)
        normals[fix] = acc[fix] / length[fix, None]
        reliable = reliable | fix
    return normals, area2 / 2


def _segment_points(P: np.ndarray, Q: np.ndarray, spacing: float) -> np.ndarray:
    n = np.maximum(np.ceil(np.linalg.norm(Q - P, axis=1) / spacing), 1).astype(np.int64)
    owner = np.repeat(np.arange(len(P)), n + 1)
    t = (np.arange(owner.size) - np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)) / n[owner]
    return P[owner] + t[:, None] * (Q - P)[owner]


def samples(tris: np.ndarray, spacing: float, with_owner: bool = False):
    """Points at most ``spacing`` apart on every triangle (long slivers cost a row, not a grid)."""
    e = np.linalg.norm(tris - tris[:, [1, 2, 0]], axis=2)
    k = e.argmax(axis=1)
    t = np.take_along_axis(tris, np.stack([k, (k + 1) % 3, (k + 2) % 3], axis=1)[:, :, None], axis=1)
    p0, p1, p2 = t[:, 0], t[:, 1], t[:, 2]
    length = e.max(axis=1)
    height = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) / np.maximum(length, 1e-12)
    nl = np.maximum(np.ceil(length / spacing), 1).astype(np.int64)
    nh = np.maximum(np.ceil(height / spacing), 1).astype(np.int64)
    count = (nl + 1) * (nh + 1)
    owner = np.repeat(np.arange(len(tris)), count)
    j = np.arange(owner.size) - np.repeat(np.cumsum(count) - count, count)
    u = (j % (nl[owner] + 1)) / nl[owner]
    w = (j // (nl[owner] + 1)) / nh[owner]
    pts = (1 - w)[:, None] * (p0[owner] + u[:, None] * (p1 - p0)[owner]) + w[:, None] * p2[owner]
    return (pts, owner) if with_owner else pts


# --- paint snapping ---------------------------------------------------------------------

def snap_to_edges(V, F, S, normals, area, a, b, edges) -> tuple[np.ndarray, int]:
    """Paint strips within ``SPILL`` of a sharp edge take the state of the face they lie on.

    The surface is split into faces along sharp edges; on each face, a connected piece of a
    minority state that lies entirely within ``SPILL`` of the face's edges is brush spill (or
    a strip the brush missed) and takes the face's main state."""
    crease = np.einsum("ij,ij->i", normals[a], normals[b]) < np.cos(np.radians(SHARP_DEG))
    if not crease.any():
        return S, 0
    face = components(len(F), a[~crease], b[~crease])
    pts = _segment_points(V[edges[crease, 0]], V[edges[crease, 1]], 0.1)
    dist, _ = cKDTree(pts).query(V, distance_upper_bound=SPILL + 0.05)
    near = (dist[F] <= SPILL + 0.05).all(axis=1)
    states = int(S.max()) + 1
    _, fid = np.unique(face, return_inverse=True)
    share = np.bincount(fid * states + S, area, minlength=(fid.max() + 1) * states)
    main = share.reshape(-1, states).argmax(axis=1)[fid]
    odd = S != main
    link = odd[a] & odd[b] & (face[a] == face[b]) & (S[a] == S[b])
    piece = components(len(F), a[link], b[link])
    all_near = np.ones(len(F), dtype=bool)
    np.logical_and.at(all_near, piece[odd], near[odd])
    flip = odd & all_near[piece]
    S = S.copy()
    S[flip] = main[flip]
    return S, int(len(np.unique(piece[flip])))


# --- features -----------------------------------------------------------------------------

@dataclass
class Cutter:
    solid: manifold3d.Manifold
    caps: list  # per cut surface: on_cap(points) -> bool mask (points lying on that surface)
    where: list  # region center, for messages
    fallback: object = None  # () -> Cutter | None: the film cutter, tried if this one fails
    mix: dict | None = None  # several colors touching: color -> triangle mask
    film: bool = False  # cut along a stretched film instead of the base surface


@dataclass
class Feature:
    colour: int
    cutters: list[Cutter] = field(default_factory=list)


@dataclass
class Findings:
    features: list[Feature] = field(default_factory=list)
    surface: dict[int, int] = field(default_factory=dict)  # color -> regions left as paint
    unsafe: dict[int, int] = field(default_factory=dict)  # color -> features not cut safely
    snapped: int = 0
    films: int = 0  # features cut along a stretched film (their border winds around the part)
    islands: list = field(default_factory=list)  # (color, center) of patches taken into a part
    surface_regions: list = field(default_factory=list)  # (color, triangle indices, center)
    recesses: dict = field(default_factory=dict)  # center (str) -> (owned mask, opening loop)
    normals: np.ndarray | None = None  # face normals and areas of the analysed mesh
    areas: np.ndarray | None = None
    failed: list = field(default_factory=list)  # centers of regions whose cut failed
    sharp: dict = field(default_factory=dict)  # region center (str) -> share of sharp border
    states: np.ndarray | None = None  # per-triangle states after snapping
    notes: list[str] = field(default_factory=list)  # why features were not cut


def _window_sharp(V, F, normals, area, centroid_tree, region, base, mid, rt, ot) -> np.ndarray:
    """Angle between the region and the base side, each averaged over ``WINDOW`` of ``mid``."""
    sharp = np.zeros(len(mid), dtype=bool)
    cos_limit = np.cos(np.radians(SHARP_DEG))
    for i, near in enumerate(centroid_tree.query_ball_point(mid, WINDOW)):
        near = np.asarray(near, dtype=np.int64)
        sides = []
        for mask, own in ((region, rt[i]), (base, ot[i])):
            idx = np.append(near[mask[near]], own)
            n = (normals[idx] * area[idx, None]).sum(axis=0)
            sides.append(n / max(np.linalg.norm(n), 1e-300))
        sharp[i] = float(np.dot(sides[0], sides[1])) < cos_limit
    return sharp


def _loops(edges: np.ndarray) -> list[np.ndarray]:
    """Directed border edges (i -> j) chained into closed loops of vertex indices."""
    nxt: dict[int, list[int]] = {}
    for i, j in edges:
        nxt.setdefault(int(i), []).append(int(j))
    loops = []
    while nxt:
        start = next(iter(nxt))
        loop, v = [start], start
        while True:
            options = nxt.get(v)
            if not options:
                break
            w = options.pop()
            if not options:
                del nxt[v]
            if w == start:
                break
            loop.append(w)
            v = w
        if len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def _in_polygons(pts: np.ndarray, polygons) -> np.ndarray:
    """Even-odd point-in-polygon test for 2D points against several outlines."""
    inside = np.zeros(len(pts), dtype=bool)
    px, py = pts[:, 0:1], pts[:, 1:2]
    for poly in polygons:
        poly = np.asarray(poly, dtype=np.float64)
        x0, y0 = poly[:, 0], poly[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        dy = np.where(y1 == y0, 1e-300, y1 - y0)
        cross = ((y0 > py) != (y1 > py)) & (px < (x1 - x0) * (py - y0) / dy + x0)
        inside ^= (cross.sum(axis=1) % 2).astype(bool)
    return inside


def _raster_heights(tris: np.ndarray, u0: float, v0: float, cell: float, nu: int, nv: int):
    """Height of the highest triangle above each grid node (-inf where none covers it).

    ``tris`` (T,3,3) are in the cap frame (u, v, w). The value is the triangle's own plane at
    the node, so the cap meets the base exactly, not within a sampling step."""
    out = np.full(nu * nv, -np.inf)
    uv = tris[:, :, :2]
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    tris, uv, e1, e2, det = (x[np.abs(det) > 1e-12] for x in (tris, uv, e1, e2, det))
    lo = np.clip(np.ceil((uv.min(axis=1) - [u0, v0]) / cell).astype(np.int64), 0, [nu, nv])
    hi = np.clip(np.floor((uv.max(axis=1) - [u0, v0]) / cell).astype(np.int64), -1, [nu - 1, nv - 1])
    cu, cv = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0), np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    count = cu * cv
    if not count.sum():
        return out
    for chunk in np.array_split(np.arange(len(tris)), max(1, int(count.sum() // 2_000_000) + 1)):
        c = count[chunk]
        owner = np.repeat(chunk, c)
        j = np.arange(owner.size) - np.repeat(np.cumsum(c) - c, c)
        i_u = lo[owner, 0] + j // cv[owner]
        i_v = lo[owner, 1] + j % cv[owner]
        p = np.stack([u0 + i_u * cell, v0 + i_v * cell], axis=1) - uv[owner, 0]
        a = (p[:, 0] * e2[owner, 1] - p[:, 1] * e2[owner, 0]) / det[owner]
        b = (e1[owner, 0] * p[:, 1] - e1[owner, 1] * p[:, 0]) / det[owner]
        hit = (a >= -1e-9) & (b >= -1e-9) & (a + b <= 1 + 1e-9)
        t = tris[owner[hit]]
        w = t[:, 0, 2] + a[hit] * (t[:, 1, 2] - t[:, 0, 2]) + b[hit] * (t[:, 2, 2] - t[:, 0, 2])
        np.maximum.at(out, i_u[hit] * nv + i_v[hit], w)
    return out


def _membrane(known: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fill the unknown cells of a grid by the Laplace equation (a stretched membrane) from
    the known ones: the smoothest surface that meets the base all along the border."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    h = np.where(known, values, 0.0)
    unknown = np.flatnonzero(~known.ravel())
    if not len(unknown):
        return h
    nu, nv = known.shape
    index = np.full(nu * nv, -1)
    index[unknown] = np.arange(len(unknown))
    i, j = np.unravel_index(unknown, known.shape)
    rows, cols, data = [], [], []
    rhs = np.zeros(len(unknown))
    count = np.zeros(len(unknown))
    flat_h = h.ravel()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = i + di, j + dj
        ok = (ni >= 0) & (ni < nu) & (nj >= 0) & (nj < nv)
        count += ok
        flat = np.where(ok, ni * nv + nj, 0)
        neighbour = index[flat]
        free = ok & (neighbour >= 0)
        rows.append(np.flatnonzero(free))
        cols.append(neighbour[free])
        data.append(-np.ones(int(free.sum())))
        fixed = ok & (neighbour < 0)
        rhs[fixed] += flat_h[flat[fixed]]
    rows.append(np.arange(len(unknown)))
    cols.append(np.arange(len(unknown)))
    data.append(count)
    A = coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(unknown), len(unknown))).tocsr()
    flat_h[unknown] = spsolve(A, rhs)
    return flat_h.reshape(known.shape)


def _cutter(V, loops, base_tris, top_pts, grow, structure=False, groups=None):
    """Closed solid above the cap surface of one feature, or a reason it cannot be cut.

    The cap is the base surface itself outside the border, and a membrane stretched across
    the border inside it, lifted ``LIFT``: it meets the base exactly along the whole border,
    however curved the base is. Returns (solid, frame, cap function, polygons, top).

    ``structure``: a shell standing on the base with openings in its paint (a cabin with bare
    door frames). Its axis is the way most of the base around it faces, and the membrane spans
    its whole shadow, so faces inside its walls (a frame around a door) do not lift the cap;
    the cap follows only the smooth faces (``groups``) that face along the axis, a deck, not a
    wall or a frame's rim."""
    from scipy.interpolate import NearestNDInterpolator
    border = np.concatenate(loops)
    # axis: the normal of the plane that best fits the base around the border, pointing
    # from the base towards the feature
    base_pts = base_tris.reshape(-1, 3)
    centred = base_pts - base_pts.mean(axis=0)
    n = np.linalg.svd(centred, full_matrices=False)[2][-1]
    if structure:
        crossed = np.cross(base_tris[:, 1] - base_tris[:, 0], base_tris[:, 2] - base_tris[:, 0])
        weighted = crossed.sum(axis=0)
        if np.linalg.norm(weighted) > 0.5 * np.linalg.norm(crossed, axis=1).sum():
            n = weighted / np.linalg.norm(weighted)
    if np.dot(top_pts.reshape(-1, 3).mean(axis=0) - base_pts.mean(axis=0), n) < 0:
        n = -n
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    frame = np.stack([u, v, n])  # rows: world -> local

    def local(p):
        return p @ frame.T
    raster = base_tris
    owned_pts = top_pts.reshape(-1, 3)
    proud = np.zeros((0, 3, 3))
    if structure and groups is not None:
        crossed = np.cross(base_tris[:, 1] - base_tris[:, 0], base_tris[:, 2] - base_tris[:, 0])
        facing = np.zeros(len(base_tris), dtype=bool)
        for g in np.unique(groups):
            m = groups == g
            total = np.linalg.norm(crossed[m], axis=1).sum()
            facing |= m & (total > 0 and crossed[m].sum(axis=0) @ n > LEVEL_ALIGN * total)
        raster = base_tris[facing]
        # the rest close to the paint (a frame standing proud of a wall) is part of the
        # structure: inside its footprint, not sliced by the cutter's side. Farther (a step
        # in the deck the structure leans on) it is base.
        proud = base_tris[~facing]
        d, _ = cKDTree(samples(top_pts.reshape(-1, 3, 3), 0.25)).query(
            proud.reshape(-1, 3), distance_upper_bound=FRAME_REACH)
        proud = proud[np.isfinite(d.reshape(-1, 3)).all(axis=1)]
        owned_pts = np.vstack([owned_pts, proud.reshape(-1, 3)])
    # footprint: the border loops, plus the shadow of everything the feature owns (a collar
    # wider than the chimney's foot must not be left behind)
    outlines = [local(V[loop])[:, :2] for loop in loops]
    section = manifold3d.CrossSection([o.tolist() for o in outlines], manifold3d.FillRule.EvenOdd)
    shadow = local(owned_pts).reshape(-1, 3, 3)[:, :, :2]
    e1, e2 = shadow[:, 1] - shadow[:, 0], shadow[:, 2] - shadow[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    shadow[signed < 0] = shadow[signed < 0][:, ::-1]  # all counter-clockwise, or they cancel
    big = np.abs(signed) > 1e-6
    if big.any():
        section = section + manifold3d.CrossSection(shadow[big].tolist(), manifold3d.FillRule.NonZero)
    if len(proud):
        # a frame's faces stand mostly along the axis, their shadows lines: each one's
        # longest edge as a thin strip too
        flat = local(proud.reshape(-1, 3)).reshape(-1, 3, 3)[:, :, :2]
        ends = np.stack([flat, np.roll(flat, -1, axis=1)], axis=2)  # (T, 3 edges, 2 ends, 2)
        longest = np.linalg.norm(ends[:, :, 1] - ends[:, :, 0], axis=2).argmax(axis=1)
        p0, p1 = ends[np.arange(len(flat)), longest, 0], ends[np.arange(len(flat)), longest, 1]
        along = p1 - p0
        length = np.linalg.norm(along, axis=1, keepdims=True)
        ok = length[:, 0] > 1e-6
        side = np.c_[-along[:, 1], along[:, 0]] / np.maximum(length, 1e-12) * 0.05
        strips = np.stack([p0 - side, p0 + side, p1 + side, p1 - side], axis=1)[ok]
        e1, e2 = strips[:, 1] - strips[:, 0], strips[:, 2] - strips[:, 0]
        cw = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0] < 0
        strips[cw] = strips[cw][:, ::-1]
        section = section + manifold3d.CrossSection(strips.tolist(), manifold3d.FillRule.NonZero)
    section = section.offset(grow, manifold3d.JoinType.Round, circular_segments=16)
    if section.is_empty():
        return None
    polygons = section.to_polygons()
    (umin, vmin), (umax, vmax) = np.asarray(section.bounds()).reshape(2, 2)
    top = float(local(top_pts.reshape(-1, 3))[:, 2].max()) + TOP_MARGIN

    # the base surface as a height field over the footprint: the highest base triangle over
    # each node. A wall running along the axis (the side of a body cut straight across) covers
    # no node; nodes nothing covers take the nearest covered one (or the border itself).
    nu = max(int(np.ceil((umax - umin) / _CELL)), 1) + 3
    nv = max(int(np.ceil((vmax - vmin) / _CELL)), 1) + 3
    us = umin - _CELL + _CELL * np.arange(nu)
    vs = vmin - _CELL + _CELL * np.arange(nv)
    U, W = np.meshgrid(us, vs, indexing="ij")
    q = np.stack([U.ravel(), W.ravel()], axis=1)
    heights = np.full(nu * nv, np.nan)
    if len(raster):
        heights = _raster_heights(local(raster.reshape(-1, 3)).reshape(-1, 3, 3), us[0], vs[0],
                                  _CELL, nu, nv)
    filled = np.isfinite(heights)
    ref = local(V[border])
    if filled.any():
        known = np.c_[q[filled], heights[filled]]
        ref = known if structure else np.vstack([ref, known])
    if not filled.all():
        heights[~filled] = NearestNDInterpolator(ref[:, :2], ref[:, 2])(q[~filled])
    inside = np.zeros(len(q), dtype=bool)
    if structure:
        inside = _in_polygons(q, [np.asarray(p) for p in polygons]) & ~filled
    else:
        for o in outlines:
            inside ^= _in_polygons(q, [o])
    h = _membrane(~inside.reshape(nu, nv), heights.reshape(nu, nv)) + LIFT

    def cap(points):
        # the cap exactly as it is triangulated below (each cell split along a-c), so a
        # point on the cut face is on the cap however steep the cap is there
        fu = np.clip((points[:, 0] - us[0]) / _CELL, 0, nu - 1 - 1e-9)
        fv = np.clip((points[:, 1] - vs[0]) / _CELL, 0, nv - 1 - 1e-9)
        i, j = fu.astype(int), fv.astype(int)
        s, t = fu - i, fv - j
        ha, hb, hc, hd = h[i, j], h[i + 1, j], h[i + 1, j + 1], h[i, j + 1]
        return np.where(s >= t, ha + s * (hb - ha) + t * (hc - hb),
                        ha + t * (hd - ha) + s * (hc - hd))
    low = h.ravel()
    if low.max() >= top:
        return "feature too flat to cut"
    bottom = np.c_[q, low]
    upper = np.c_[q, np.full(len(q), top)]
    verts = np.vstack([bottom, upper])
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    m = nu * nv
    faces = [np.stack([a, d, c], 1), np.stack([a, c, b], 1),  # bottom, facing -w
             np.stack([a, b, c], 1) + m, np.stack([a, c, d], 1) + m]  # top, facing +w
    rim = np.r_[idx[:, 0], idx[-1, 1:], idx[-2::-1, -1], idx[0, -2:0:-1]]
    r0, r1 = rim, np.roll(rim, -1)
    faces += [np.stack([r0, r1, r1 + m], 1), np.stack([r0, r1 + m, r0 + m], 1)]
    faces = np.vstack(faces)
    slab = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts.astype(np.float32),
                                               tri_verts=faces.astype(np.uint32)))
    if slab.status() != manifold3d.Error.NoError:
        return "cap failed"
    if slab.volume() < 0 or slab.is_empty():
        faces = faces[:, ::-1]
        slab = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts.astype(np.float32),
                                                   tri_verts=faces.astype(np.uint32)))
    zlo = float(low.min()) - 1.0
    prism = manifold3d.Manifold.extrude(section, top - zlo).translate((0.0, 0.0, zlo))
    solid = prism ^ slab
    to_world = np.c_[frame.T, np.zeros(3)]  # local -> world
    return solid.transform(to_world.tolist()), frame, cap, polygons, top


FRAME_REACH = 1.5  # mm a frame may stand proud of a structure's painted wall
LEVEL_ALIGN = 0.9  # a structure's cap follows smooth faces facing this close to its axis
FILM_SHARP_SHARE = 0.15  # with Raise or Inlay chosen, a smoother border stays surface paint
STRUCTURE_SHARP_SHARE = 0.5  # with no film, a border this sharp is still cut at its foot
FILM_OFFSET = 0.02  # mm the film cutter's painted side lies outside the model surface
FILM_CELL = 0.5  # mm between the rings of a film
FILM_SMOOTHING = 400  # relaxation sweeps that stretch a film like a soap film


def _height_cap(frame, cap):
    def on_cap(points):
        local = points @ frame.T
        return np.abs(local[:, 2] - cap(local[:, :2])) < 5e-3  # float32 mesh: ~3e-3
    return on_cap


def _film(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A disc of triangles spanning the closed polyline ``ring`` (N,3), relaxed like a soap
    film: every inner vertex moves to the mean of its neighbors. Returns (points, faces);
    the first N points are ``ring``. Faces run against the ring's direction."""
    n = len(ring)
    centre = ring.mean(axis=0)
    radius = float(np.linalg.norm(ring - centre, axis=1).mean())
    rings = max(2, int(np.ceil(radius / FILM_CELL)))
    t = (np.arange(rings) / rings)[:, None, None]
    grid = ring[None] + (centre - ring)[None] * t  # (rings, n, 3); grid[0] is the ring
    points = np.vstack([grid.reshape(-1, 3), centre[None]])
    middle = len(points) - 1
    idx = np.arange(rings * n).reshape(rings, n)
    faces = []
    for j in range(rings - 1):
        a, b = idx[j], np.roll(idx[j], -1)
        c, d = idx[j + 1], np.roll(idx[j + 1], -1)
        faces += [np.stack([b, a, c], 1), np.stack([b, c, d], 1)]
    last = idx[rings - 1]
    faces.append(np.stack([np.roll(last, -1), last, np.full(n, middle)], 1))
    faces = np.vstack(faces)
    # relax the inner vertices (ring 0 stays fixed) towards a minimal surface
    free = np.arange(n, len(points))
    nb = [[] for _ in range(len(points))]
    for f in faces:
        for i in range(3):
            nb[f[i]].append(f[(i + 1) % 3])
            nb[f[i]].append(f[(i + 2) % 3])
    lists = [np.unique(nb[v]) for v in free]
    owner = np.repeat(np.arange(len(free)), [len(l) for l in lists])
    flat = np.concatenate(lists)
    count = np.bincount(owner, minlength=len(free))
    for _ in range(FILM_SMOOTHING):
        acc = np.zeros((len(free), 3))
        np.add.at(acc, owner, points[flat])
        points[free] = acc / count[:, None]
    return points, faces


def _film_cutter(V, F, normals, area, owned, side_loops, where, offset=FILM_OFFSET):
    """Closed solid of the feature: its painted surface (pushed FILM_OFFSET outward) closed by
    one stretched film per border loop. Works however the border winds around the part (a
    ball wrapped in paint), where no single direction sees the whole cap. The part touches the
    base only through the films, so it can always be taken off."""
    tris = F[owned]
    used = np.unique(tris)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    vn = np.zeros((len(used), 3))
    np.add.at(vn, remap[tris].reshape(-1), np.repeat(normals[owned] * area[owned, None], 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    P = V[used] + offset * vn
    verts, faces, films = [P], [remap[tris]], []
    offset = len(P)
    for loop in side_loops:
        L = remap[loop]
        if (L < 0).any():
            return None
        pts, ff = _film(P[L])
        inner = np.arange(len(pts) - len(L)) + offset
        index = np.r_[L, inner]
        verts.append(pts[len(L):])
        faces.append(index[ff])
        films.append(pts[ff])
        offset += len(pts) - len(L)
    verts, faces = np.vstack(verts), np.vstack(faces)
    for attempt in (faces, faces[:, ::-1]):
        solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts.astype(np.float32),
                                                    tri_verts=attempt.astype(np.uint32)))
        if solid.status() == manifold3d.Error.NoError and not solid.is_empty() and solid.volume() > 0:
            break
    else:
        return None
    film_pts = samples(np.concatenate(films), 0.05)
    tree = cKDTree(film_pts)

    def on_film(points):
        d, _ = tree.query(points, distance_upper_bound=0.06)
        return np.isfinite(d)
    return Cutter(solid, [on_film], where, film=True)


def _surface_paint(findings: Findings, colour: int, mine: np.ndarray, centroids,
                   recess=None) -> None:
    """Count a region as surface paint and keep its triangles (it may be raised or inlaid).
    ``recess`` (owned mask, border loop) marks paint lining a recess with one opening."""
    findings.surface[colour] = findings.surface.get(colour, 0) + 1
    where = np.round(centroids[mine].mean(axis=0), 1).tolist()
    findings.surface_regions.append((colour, np.flatnonzero(mine), where))
    if recess is not None:
        findings.recesses[str(where)] = recess


def recess_fill(V, F, normals, area, owned, loop) -> manifold3d.Manifold | None:
    """The air inside a painted recess, closed flush by a film across its opening."""
    cutter = _film_cutter(V, F, normals, area, owned, [loop], None, offset=-FILM_OFFSET)
    return None if cutter is None else cutter.solid


def find_features(V, F, S, base_state: int, cut_smooth: bool = True) -> Findings:
    """Feature cutters for every color (states other than 0 and ``base_state``)."""
    a, b, edges = edge_pairs(F)
    normals, area = normals_and_areas(V, F, a, b)
    findings = Findings()
    S, findings.snapped = snap_to_edges(V, F, S, normals, area, a, b, edges)
    findings.states = S
    findings.normals, findings.areas = normals, area
    crease = np.einsum("ij,ij->i", normals[a], normals[b]) < np.cos(np.radians(SHARP_DEG))
    face = components(len(F), a[~crease], b[~crease])  # smooth faces between sharp edges
    centroids = V[F].mean(axis=1)
    tree = cKDTree(centroids)
    colours = sorted(int(s) for s in np.unique(S) if s not in (0, base_state))
    # Colors that touch each other (a tower painted half red, half blue) are one feature: it
    # is cut from the base as a whole, then split between its colors (solids.split).
    any_paint = np.isin(S, colours)
    group = components(len(F), a[any_paint[a] & any_paint[b]], b[any_paint[a] & any_paint[b]])
    mixed = np.zeros(len(F), dtype=bool)
    jobs = []  # (color, painted mask, [(region mask, {color: mask} or None)])
    for label in np.unique(group[any_paint]):
        members = any_paint & (group == label)
        here = np.unique(S[members])
        if len(here) > 1:
            mixed |= members
            share = [area[members & (S == c)].sum() for c in here]
            jobs.append((int(here[int(np.argmax(share))]), any_paint,
                         [(members, {int(c): members & (S == c) for c in here})]))
    for colour in colours:
        painted = (S == colour) & ~mixed
        region = components(len(F), a[painted[a] & painted[b]], b[painted[a] & painted[b]])
        jobs.append((colour, painted, [(painted & (region == l), None)
                                       for l in np.unique(region[painted])]))
    by_colour: dict[int, Feature] = {}
    for colour, painted, masks in jobs:
        rest = components(len(F), a[~painted[a] & ~painted[b]], b[~painted[a] & ~painted[b]])
        rest_area = np.bincount(rest, np.where(painted, 0.0, area), minlength=len(F))
        main = int(rest_area.argmax())
        feature = by_colour.setdefault(colour, Feature(colour))
        for mine, mix in masks:
            if not mine.any():
                continue
            cross_ab = mine[a] & ~painted[b]
            cross_ba = mine[b] & ~painted[a]
            rt = np.r_[a[cross_ab], b[cross_ba]]
            ot = np.r_[b[cross_ab], a[cross_ba]]
            ev = np.r_[edges[cross_ab], edges[cross_ba]]
            # The unpainted surfaces next to the region are either base *sides* the feature is
            # cut from (the body; or a shin below a knee ball, attached only through the ball),
            # or *cavities* enclosed by it (a chimney's bore), which go with the feature. Closed
            # by its border, a side encloses material (positive volume), a cavity air.
            sides, cavities = [], []
            for c in np.unique(rest[ot]):
                if c == main:
                    sides.append(c)
                    continue
                comp = ~painted & (rest == c)
                o = V[ev[rest[ot] == c]].reshape(-1, 3).mean(axis=0)
                vol = float(np.sum(area[comp] * np.einsum("ij,ij->i", centroids[comp] - o,
                                                          normals[comp])) / 3)
                if abs(vol) < MIN_FEATURE_VOLUME:
                    # a flat patch of another color inside the region (a dot on a ball): it
                    # goes with the feature, which is printed in one color
                    cavities.append(c)
                    findings.islands.append((colour, np.round(centroids[comp].mean(axis=0), 1).tolist()))
                    continue
                (sides if vol > 0 else cavities).append(c)
            cut = np.isin(rest[ot], sides)
            if not cut.any():
                _surface_paint(findings, colour, mine, centroids)
                continue
            rt, ot, ev = rt[cut], ot[cut], ev[cut]
            base_side = ~painted & np.isin(rest, sides)
            mid = V[ev].mean(axis=1)
            sharp = _window_sharp(V, F, normals, area, tree, mine, base_side, mid, rt, ot)
            length = np.linalg.norm(V[ev[:, 0]] - V[ev[:, 1]], axis=1)
            findings.sharp[str(np.round(centroids[mine].mean(axis=0), 1).tolist())] =                 float(length[sharp].sum() / max(length.sum(), 1e-12))
            share = length[sharp].sum() / max(length.sum(), 1e-12)
            # partly sharp (a ball half sunk into a part, its paint winding around it): no
            # single cap fits, but a film across the border can still cut it off
            film_only = share < SHARP_SHARE
            if film_only and share < FILM_SHARP_SHARE and not cut_smooth:
                _surface_paint(findings, colour, mine, centroids)
                continue
            owned = mine | (~painted & np.isin(rest, cavities))
            top_pts = V[F[owned]]
            # a feature encloses volume together with its caps (a chimney); paint lining a
            # hole through a wall encloses none (outward normals point into the hole)
            o = V[ev].reshape(-1, 3).mean(axis=0)
            enclosed_volume = float(np.sum(area[owned] * np.einsum(
                "ij,ij->i", centroids[owned] - o, normals[owned])) / 3)
            if enclosed_volume < MIN_FEATURE_VOLUME:
                recess = None
                if enclosed_volume < -MIN_FEATURE_VOLUME and len(sides) == 1:
                    # paint lining a recess (engraved text): with one opening, it can be
                    # filled flush on request; a hole through a wall has two and is not
                    directed = []
                    for t, (i, j) in zip(rt, ev):
                        f = list(F[t])
                        k = f.index(i)
                        directed.append((i, j) if f[(k + 1) % 3] == j else (j, i))
                    loops = _loops(np.array(directed))
                    if len(loops) == 1:
                        recess = (owned, loops[0])
                _surface_paint(findings, colour, mine, centroids, recess)
                continue
            where = np.round(centroids[mine].mean(axis=0), 1).tolist()
            # one cutter per base side, all its borders sharing one cap (a frame around a
            # window is a ring, not a disc over the window); the feature is what lies above
            # every cap (a knee ball between a thigh and a shin)
            solid, caps, reason, side_loops = None, [], None, []
            for side in sorted(sides, key=lambda c: c != main):  # the main side first
                on_side = rest[ot] == side
                directed = []  # each border edge as it runs in its painted triangle
                for t, (i, j) in zip(rt[on_side], ev[on_side]):
                    f = list(F[t])
                    k = f.index(i)
                    directed.append((i, j) if f[(k + 1) % 3] == j else (j, i))
                loops = _loops(np.array(directed))
                if not loops:
                    reason = "border does not close"
                    break
                side_loops.append(loops)
                if film_only and share < STRUCTURE_SHARP_SHARE:
                    continue  # only a film can cut this one
                # the cap continues the faces the feature stands on (not faces across an
                # edge: the walls of a window inside a frame, of a washer around a chimney's
                # foot), within BAND of the outer border
                outer = max(loops, key=lambda l: np.linalg.norm(V[l] - V[np.roll(l, 1)], axis=1).sum())
                if film_only and side != main:
                    # another side of a structure (a chimney on a cabin's roof) is a feature
                    # of its own standing on it: cut off at its foot and taken out of it
                    near = np.flatnonzero(mine)
                    d, _ = cKDTree(V[outer]).query(V[F[near]].reshape(-1, 3), distance_upper_bound=BAND)
                    near = near[np.isfinite(d.reshape(-1, 3)).any(axis=1)]
                    built = _cutter(V, loops, V[F[near]], V[F[~painted & (rest == side)]], GROW)
                    if isinstance(built, str) or built is None or solid is None:
                        reason = (built if isinstance(built, str) else None) or "cap failed"
                        break
                    piece, frame, cap, _, _ = built
                    solid = solid - piece
                    caps.append(_height_cap(frame, cap))
                    continue
                on_outer = np.isin(ev, outer).all(axis=1) & on_side
                # ... and the feature stands on a base face across a concave edge (a deck under a
                # cabin wall); across a convex one the face is the reveal of an opening in the
                # feature (a door arch left unpainted), which the cut must not follow up
                concave = on_outer & (np.einsum("ij,ij->i", normals[ot],
                                                centroids[rt] - centroids[ot]) > 0)
                feet = ot[concave] if concave.any() else ot[on_outer]
                standing = base_side & (rest == side) & np.isin(face, np.unique(face[feet]))
                cand = np.flatnonzero(standing)
                d, _ = cKDTree(V[outer]).query(V[F[cand]].reshape(-1, 3), distance_upper_bound=BAND)
                cand = cand[np.isfinite(d.reshape(-1, 3)).any(axis=1)]
                if not len(cand):
                    reason = "no base surface around the border"
                    break
                built = _cutter(V, loops, V[F[cand]], top_pts, GROW, structure=film_only,
                                groups=face[cand])
                if isinstance(built, str) or built is None:
                    reason = built or "cap failed"
                    break
                piece, frame, cap, _, _ = built
                solid = piece if solid is None else solid ^ piece
                caps.append(_height_cap(frame, cap))

            # the film cutter: every side has one closed border (a frame around a window has
            # two, and a film would fill the window)
            def film(owned=owned, side_loops=list(side_loops), where=where):
                if not side_loops or any(len(l) != 1 for l in side_loops):
                    return None
                return _film_cutter(V, F, normals, area, owned, [l[0] for l in side_loops], where)
            primary_ok = reason is None and solid is not None and not solid.is_empty()
            if film_only:
                # partly sharp. Mostly sharp (a cabin painted outside, its door and window
                # reveals left bare): cut off at its foot like any feature, walls and all, with
                # the film as the fallback. Less (a ball sunk into a part): only a film fits.
                if primary_ok and share >= STRUCTURE_SHARP_SHARE:
                    feature.cutters.append(Cutter(solid, caps, where, fallback=film, mix=mix))
                    continue
                cutter = film() if reason != "border does not close" else None
                if cutter is None:
                    _surface_paint(findings, colour, mine, centroids)
                    continue
                cutter.mix = mix
                feature.cutters.append(cutter)
                continue
            if reason is not None or solid is None or solid.is_empty():
                cutter = film() if reason != "border does not close" else None
                if cutter is None:
                    findings.notes.append(f"color {colour} at {where}: {reason or 'caps do not meet'}")
                    findings.unsafe[colour] = findings.unsafe.get(colour, 0) + 1
                    findings.failed.append(where)
                    continue
                cutter.mix = mix
                feature.cutters.append(cutter)
                continue
            # whether the cut is safe is checked on the result (solids.split): base caught by
            # the cutter that comes off as its own piece simply goes back to the base; if it
            # slices through base joined to the feature, the film cutter is tried instead
            feature.cutters.append(Cutter(solid, caps, where, fallback=film, mix=mix))
    findings.features = [f for f in by_colour.values() if f.cutters]
    return findings


def colour_film(V, F, normals, area, mask) -> manifold3d.Manifold | None:
    """The solid a color patch claims inside a feature painted in several colors: its
    surface closed by a film across its border (with the other colors and the base)."""
    a, b, edges = edge_pairs(F)
    cross_ab = mask[a] & ~mask[b]
    cross_ba = mask[b] & ~mask[a]
    rt = np.r_[a[cross_ab], b[cross_ba]]
    ev = np.r_[edges[cross_ab], edges[cross_ba]]
    directed = []
    for t, (i, j) in zip(rt, ev):
        f = list(F[t])
        k = f.index(i)
        directed.append((i, j) if f[(k + 1) % 3] == j else (j, i))
    loops = _loops(np.array(directed)) if directed else []
    if len(loops) != 1:
        return None
    cutter = _film_cutter(V, F, normals, area, mask, loops, None)
    return None if cutter is None else cutter.solid
