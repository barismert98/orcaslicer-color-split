# SPDX-License-Identifier: GPL-3.0-or-later
"""Carry the paint that stays on the base over to the base mesh written in the project.

Paint that could not be cut off (surface paint) stays on the base and is printed as painted,
so the project must say so. The base is a boolean result, but almost all of its triangles
are the model's own: those keep their ``paint_color`` string exactly, written from the same
corner, since the string depends on it. Triangles the cuts made new take the paint under them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import features, paint

_MATCH = 1e-3  # mm a base vertex may be from the model's: the base is float32
_ON_SURFACE = 5e-3  # mm; farther from the model's surface is a new face (a cap is 0.02 above it)
_SAMPLING = 0.25  # mm between the surface points paint is looked up on
_MAX_DEPTH = 5  # subdivisions of one new triangle along a paint border
_MIN_EDGE = 0.4  # mm; a triangle this small takes one color


@dataclass
class Surface:
    """The painted model: triangles, their final states, and their unit normals."""
    V: np.ndarray
    F: np.ndarray
    states: np.ndarray
    normals: np.ndarray


def leaf_code(state: int) -> str:
    """``paint_color`` of a whole triangle in ``state`` (the string is read from its end)."""
    if state < 3:
        return format(state << 2, "X")
    if state < 18:
        return f"{state - 3:X}C"
    return f"{state - 18:X}FC"


def _welded(V: np.ndarray) -> np.ndarray:
    _, vid = np.unique(np.round(V / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    return vid.reshape(-1)


def _lookup(surface: Surface, tris: np.ndarray, base_state: int):
    """A function giving the state (0: unpainted) on the model's surface at given points."""
    state = np.where(np.isin(surface.states, (0, base_state)), 0, surface.states)
    c = surface.V[surface.F]
    near = (c.max(axis=1) >= tris.min(axis=(0, 1)) - 1.0).all(axis=1) & (
        c.min(axis=1) <= tris.max(axis=(0, 1)) + 1.0).all(axis=1)
    index = np.flatnonzero(near)
    if not len(index):
        return lambda points: np.zeros(len(points), dtype=np.int64)
    pts, owner = features.samples(c[index], _SAMPLING, with_owner=True)
    tree = cKDTree(pts)

    def at(points):
        _, i = tree.query(points)
        t = index[owner[i]]
        off = np.abs(np.einsum("ij,ij->i", points - c[t, 0], surface.normals[t]))
        return np.where(off <= _ON_SURFACE, state[t], 0)
    return at


def _subdivided(tri: np.ndarray, at, depth: int) -> list[int]:
    """Nibbles (in stream order) of ``tri`` painted as the surface under it."""
    centre = tri.mean(axis=0)
    probes = at(np.vstack([centre[None], 0.5 * (tri + centre)]))
    longest = np.linalg.norm(tri - tri[[1, 2, 0]], axis=1).max()
    if len(np.unique(probes)) == 1 or depth >= _MAX_DEPTH or longest < _MIN_EDGE:
        return [int(ch, 16) for ch in reversed(leaf_code(int(probes[0])))]
    children = paint._children(tri, 3, 0)
    return [3] + [n for i in range(3, -1, -1) for n in _subdivided(children[i], at, depth + 1)]


def transfer(V, F, src_V, src_F, src_codes, src_defaults, surface: Surface, base_state: int,
             verbatim: bool = True) -> tuple[np.ndarray, list]:
    """(F with matched triangles rotated to the model's first corner, ``paint_color`` per triangle).

    ``verbatim`` False looks every triangle up on ``surface`` (some paint became parts)."""
    V, F = np.asarray(V, dtype=np.float64), np.array(F, dtype=np.int64).reshape(-1, 3)
    src_V, src_F = np.asarray(src_V, dtype=np.float64), np.asarray(src_F, dtype=np.int64).reshape(-1, 3)
    codes: list = [None] * len(F)
    if not len(F) or not np.isin(surface.states, (0, base_state), invert=True).any():
        return F, codes
    todo = np.ones(len(F), dtype=bool)
    if verbatim:
        vid = _welded(src_V)
        dist, near = cKDTree(src_V).query(V, distance_upper_bound=_MATCH)
        found = np.isfinite(dist)
        out_id = np.full(len(V), -1, dtype=np.int64)
        out_id[found] = vid[near[found]]
        G = out_id[F]
        H = vid[src_F]
        own = {tuple(k): i for i, k in enumerate(np.sort(H, axis=1).tolist())}
        for t in np.flatnonzero((G >= 0).all(axis=1)):
            i = own.get(tuple(sorted(G[t].tolist())))
            if i is None or src_defaults[i] != base_state:
                continue
            first = np.flatnonzero(G[t] == H[i, 0])
            if len(first) != 1:
                continue
            F[t] = np.roll(F[t], -int(first[0]))
            if not np.array_equal(out_id[F[t]], H[i]):
                continue  # the opposite winding: not the same triangle
            codes[t] = src_codes[i] or None
            todo[t] = False
    rest = np.flatnonzero(todo)
    if not len(rest):
        return F, codes
    tris = V[F[rest]]
    at = _lookup(surface, tris, base_state)
    centre = tris.mean(axis=1)
    inner = 0.5 * (tris + centre[:, None])
    probes = at(np.vstack([centre, inner.reshape(-1, 3)]))
    probes = np.c_[probes[:len(rest)], probes[len(rest):].reshape(-1, 3)]
    for k in np.flatnonzero((probes != probes[:, [0]]).any(axis=1) | (probes[:, 0] != 0)):
        nibbles = _subdivided(tris[k], at, 0)
        if nibbles != [0]:
            codes[rest[k]] = "".join(format(n, "X") for n in reversed(nibbles))
    return F, codes
