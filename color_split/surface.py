# SPDX-License-Identifier: GPL-3.0-or-later
"""Surface paint made into parts, like OrcaSlicer's text tool: raised or inlaid.

Paint on a smooth surface (a stripe, a logo, a ring) has no edge to cut along. On request it
becomes a part anyway:

* **raised**: a layer ``height`` thick on top of the surface. The base is unchanged.
* **inlaid**: a pocket ``depth`` deep in the base, and a part that fills it flush.

Both are a *thickened* copy of the painted surface: the surface moved along its vertex
normals, closed by walls along the paint border. Two things keep it clean on real meshes:

* vertex normals come from repaired face normals (sliver triangles have meaningless ones)
  and are smoothed over the region, so a staircase border or a sliver cannot tilt them;
* the offset is clipped by the model with a boolean, so the visible surface stays exact.
"""
from __future__ import annotations

import manifold3d
import numpy as np

from . import features

SMOOTHING = 8  # passes that average each vertex normal with its neighbors
MAX_MITER = 2.0  # an offset is never stretched more than this at a sharp bend
OVERLAP = 0.05  # mm the thickened layer reaches past the surface, so faces never coincide
DEFAULT_THICKNESS = 0.6  # mm
THICKNESS_RANGE = (0.2, 5.0)


def thicken(V: np.ndarray, F: np.ndarray, outward: float, inward: float) -> manifold3d.Manifold | None:
    """Closed solid between the surface (V, F) moved ``outward`` and moved ``inward`` along its
    smoothed vertex normals; None if it is not a valid solid."""
    used, inverse = np.unique(F.reshape(-1), return_inverse=True)
    P = V[used]
    T = inverse.reshape(-1, 3)
    a, b, _ = features.edge_pairs(T)
    normals, area = features.normals_and_areas(P, T, a, b)
    vn = np.zeros_like(P)
    np.add.at(vn, T.reshape(-1), np.repeat(normals * area[:, None], 3, axis=0))
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    # neighbors along edges (each vertex pair once)
    e = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    for _ in range(SMOOTHING):
        acc = vn.copy()
        np.add.at(acc, e[:, 0], vn[e[:, 1]])
        np.add.at(acc, e[:, 1], vn[e[:, 0]])
        vn = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    # keep the layer's thickness where faces meet at an angle
    dots = np.einsum("ij,ij->i", vn[T.reshape(-1)], np.repeat(normals, 3, axis=0))
    worst = np.ones(len(P))
    np.minimum.at(worst, T.reshape(-1), dots)
    scale = 1.0 / np.clip(worst, 1.0 / MAX_MITER, 1.0)
    top = P + vn * (outward * scale)[:, None]
    bottom = P - vn * (inward * scale)[:, None]
    n = len(P)
    half = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    key = np.sort(half, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = half[counts[inv.reshape(-1)] == 1]
    i, j = border[:, 0], border[:, 1]
    faces = np.vstack([T, T[:, ::-1] + n,
                       np.stack([j, i, i + n], axis=1), np.stack([j, i + n, j + n], axis=1)])
    verts = np.vstack([top, bottom])
    solid = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts.astype(np.float32),
                                                tri_verts=faces.astype(np.uint32)))
    if solid.status() != manifold3d.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        return None
    return solid


def raised(obj: manifold3d.Manifold, V, F, height: float) -> manifold3d.Manifold | None:
    """A layer ``height`` thick on the painted surface; it never overlaps the model."""
    layer = thicken(V, F, height, OVERLAP)
    if layer is None:
        return None
    part = layer - obj
    return None if part.is_empty() else part


def inlaid(obj: manifold3d.Manifold, V, F, depth: float):
    """(part, cutter): the model ``depth`` deep under the painted surface, and the cutter that
    makes its pocket in the base."""
    cutter = thicken(V, F, OVERLAP, depth)
    if cutter is None:
        return None, None
    part = obj ^ cutter
    return (None, None) if part.is_empty() else (part, cutter)
