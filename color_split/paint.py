# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode OrcaSlicer multi-material paint (3MF ``paint_color``) into leaf sub-triangles.

Mirrors ``TriangleSelector::deserialize`` / ``perform_split`` and
``FacetsAnnotation::set_triangle_from_string`` (OrcaSlicer v2.5.0-rc):

* the nibble stream is the hex string read from its last character to its first;
* a code nibble's low two bits are the number of split sides; a leaf (0 split sides)
  stores its state in the upper two bits, or, when they are ``0b11``, in the next nibble
  (+3), where a nibble of 15 escapes to one more nibble (+18);
* a split node's special side is ``code >> 2`` (always 0 for three split sides); its
  ``split + 1`` children follow depth-first in child-index order ``split .. 0``;
* children are formed from the parent's corners rotated to start at the special side,
  with exact edge midpoints.

State 0 is unpainted (the object's own extruder); state k >= 1 is extruder k.
"""
from __future__ import annotations

import numpy as np


class PaintFormatError(ValueError):
    """A ``paint_color`` string that OrcaSlicer itself would reject as malformed."""


def _nibbles(code: str) -> list[int]:
    try:
        return [int(ch, 16) for ch in reversed(code)]
    except ValueError as exc:
        raise PaintFormatError(f"not a hex paint string: {code!r}") from exc


def _children(corners: np.ndarray, split: int, special: int) -> list[np.ndarray]:
    v0, v1, v2 = corners[special % 3], corners[(special + 1) % 3], corners[(special + 2) % 3]
    if split == 1:
        m = 0.5 * (v1 + v2)
        return [np.array([v0, v1, m]), np.array([m, v2, v0])]
    if split == 2:
        a, b = 0.5 * (v0 + v1), 0.5 * (v0 + v2)
        return [np.array([v0, a, b]), np.array([a, v1, b]), np.array([v1, v2, b])]
    a, b, c = 0.5 * (v0 + v1), 0.5 * (v1 + v2), 0.5 * (v2 + v0)
    return [np.array([v0, a, c]), np.array([a, v1, b]), np.array([b, v2, c]), np.array([a, b, c])]


def leaf_triangles(corners, code: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the leaf sub-triangles (L,3,3) and their states (L,) of one painted triangle.

    Leaves are listed depth-first in ascending child-index order.
    """
    nibbles = _nibbles(code)
    pos = 0

    def take() -> int:
        nonlocal pos
        if pos >= len(nibbles):
            raise PaintFormatError(f"truncated paint string: {code!r}")
        pos += 1
        return nibbles[pos - 1]

    def node(tri: np.ndarray) -> list[tuple[np.ndarray, int]]:
        code_nibble = take()
        split = code_nibble & 0b11
        if split == 0:
            if code_nibble & 0b1100 == 0b1100:
                n = take()
                return [(tri, take() + 18 if n == 0b1111 else n + 3)]
            return [(tri, code_nibble >> 2)]
        if split != 3 and (code_nibble >> 2) == 3:
            raise PaintFormatError(f"invalid split code in {code!r}")
        children = _children(tri, split, 0 if split == 3 else code_nibble >> 2)
        decoded = {idx: node(children[idx]) for idx in range(split, -1, -1)}  # serialized order
        return [leaf for idx in range(split + 1) for leaf in decoded[idx]]

    leaves = node(np.asarray(corners, dtype=np.float64))
    return (np.array([t for t, _ in leaves]).reshape(-1, 3, 3),
            np.array([s for _, s in leaves], dtype=np.int64))


def split_mesh(V, F, codes, defaults=None) -> tuple[np.ndarray, np.ndarray]:
    """Expand every triangle into its painted leaves; unpainted triangles stay whole with state 0.

    ``defaults`` (per triangle) replaces state 0, unpainted, by the filament of the
    triangle's part, so parts set to other filaments keep their color."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    out_corners = []
    out_states = []
    defaults = np.zeros(len(F), dtype=np.int64) if defaults is None else np.asarray(defaults)
    for tri, code, default in zip(V[F], codes, defaults):
        if not code:
            out_corners.append(tri[None])
            out_states.append(np.full(1, default, dtype=np.int64))
            continue
        corners, states = leaf_triangles(tri, code)
        out_corners.append(corners)
        out_states.append(np.where(states == 0, default, states))
    if not out_corners:
        return np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64)
    return np.concatenate(out_corners), np.concatenate(out_states)
