# SPDX-License-Identifier: GPL-3.0-or-later
"""Binary STL writing and reading."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_RECORD = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])


def write_stl(path, V, F) -> None:
    """Write a binary STL from vertices (N,3) and triangles (M,3)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    corners = V[F]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(F), dtype=_RECORD)
    records["normal"] = normals
    records["corners"] = corners
    with Path(path).open("wb") as out:
        out.write(b"color_split".ljust(80, b"\0"))
        out.write(np.uint32(len(F)).tobytes())
        out.write(records.tobytes())


def read_stl(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary STL; vertices are welded by exact coordinates."""
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    if len(data) < 84 + count * _RECORD.itemsize:
        raise ValueError(f"{path}: truncated binary STL")
    records = np.frombuffer(data, dtype=_RECORD, count=count, offset=84)
    corners = records["corners"].astype(np.float64).reshape(-1, 3)
    V, inverse = np.unique(corners, axis=0, return_inverse=True)
    return V, inverse.reshape(-1, 3)
