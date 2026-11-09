# SPDX-License-Identifier: GPL-3.0-or-later
"""Clean up careless painting before splitting.

A brush rarely covers every triangle: a few slivers inside a painted area stay unpainted, and a
stray click leaves a speck of colour elsewhere. Each such island would become a needle of base
material left in a pocket, or a crumb of an inlay. Islands smaller than ``MIN_AREA`` mm², or
thinner than ``MIN_WIDTH`` mm (2 · area / border length), are given the state they share the
longest border with.
"""
from __future__ import annotations

import numpy as np

MIN_AREA = 1.0  # mm²
MIN_WIDTH = 0.3  # mm


def _edges(leaves: np.ndarray):
    keys = np.round(leaves.reshape(-1, 3) / 1e-6).astype(np.int64)
    _, vid = np.unique(keys, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    edges = np.sort(np.stack([vid[:, [0, 1]], vid[:, [1, 2]], vid[:, [2, 0]]], axis=1), axis=2)
    _, edge_id = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    edge_id = edge_id.reshape(-1)
    tri = np.repeat(np.arange(len(leaves)), 3)
    order = np.argsort(edge_id, kind="stable")
    same = edge_id[order][1:] == edge_id[order][:-1]
    a, b = tri[order][:-1][same], tri[order][1:][same]
    first = order[:-1][same]  # index into the 3·T edge list of the shared edge
    corner = np.stack([leaves[:, [0, 1]], leaves[:, [1, 2]], leaves[:, [2, 0]]], axis=1).reshape(-1, 2, 3)
    length = np.linalg.norm(corner[first, 1] - corner[first, 0], axis=1)
    return a, b, length


def _labels(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    label = np.arange(n)
    while len(a):
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        label = label[label]
        if np.array_equal(before, label):
            break
    return label


def clean_states(leaves, states, min_area: float = MIN_AREA, min_width: float = MIN_WIDTH,
                 rounds: int = 6) -> tuple[np.ndarray, int]:
    """Return (cleaned states, number of islands reassigned)."""
    leaves = np.asarray(leaves, dtype=np.float64).reshape(-1, 3, 3)
    states = np.asarray(states, dtype=np.int64).copy()
    if len(leaves) == 0:
        return states, 0
    area = 0.5 * np.linalg.norm(np.cross(leaves[:, 1] - leaves[:, 0], leaves[:, 2] - leaves[:, 0]), axis=1)
    a, b, length = _edges(leaves)
    changed_total = 0
    for _ in range(rounds):
        same = states[a] == states[b]
        label = _labels(len(leaves), a[same], b[same])
        ids, inv = np.unique(label, return_inverse=True)
        region_area = np.bincount(inv, area)
        cross = ~same
        ra, rb = inv[a[cross]], inv[b[cross]]
        border = np.bincount(np.r_[ra, rb], np.r_[length[cross], length[cross]], minlength=len(ids))
        width = np.divide(2 * region_area, border, out=np.full(len(ids), np.inf), where=border > 0)
        small = (region_area < min_area) | (width < min_width)
        small &= border > 0  # an island must touch something to be merged into it
        if not small.any():
            break
        # longest shared border per (small region, neighbour state)
        target = np.full(len(ids), -1)
        best = np.zeros(len(ids))
        for src, dst_tri, w in ((ra, b[cross], length[cross]), (rb, a[cross], length[cross])):
            pick = small[src]
            src, dst_state, w = src[pick], states[dst_tri[pick]], w[pick]
            if not len(src):
                continue
            pairs, pinv = np.unique(np.stack([src, dst_state], axis=1), axis=0, return_inverse=True)
            total = np.bincount(pinv.reshape(-1), w)
            for (r, s), t in zip(pairs, total):
                if t > best[r]:
                    best[r], target[r] = t, s
        # merge only the smallest island of each touching pair per round, largest first stays
        movable = small & (target >= 0)
        if not movable.any():
            break
        states = np.where(movable[inv], target[inv], states)
        changed_total += int(movable.sum())
    return states, changed_total
