#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build release/Color_Split-<version>_any.whl with LF line endings in its METADATA.

OrcaSlicer's METADATA parser keeps a trailing CR, which would end up in the installed plugin's
name and version; on Windows the build writes CRLF.
"""
from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


def _hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def normalize_wheel(source: Path, target: Path) -> None:
    """Copy wheel ``source`` to ``target``: METADATA and RECORD with LF, RECORD's hashes updated."""
    with zipfile.ZipFile(source) as z:
        files = {i.filename: z.read(i) for i in z.infolist()}
    meta = next(n for n in files if n.endswith(".dist-info/METADATA"))
    record = next(n for n in files if n.endswith(".dist-info/RECORD"))
    files[meta] = files[meta].replace(b"\r\n", b"\n")
    lines = []
    for line in files[record].decode("utf-8").splitlines():
        name = line.split(",", 1)[0]
        lines.append(f"{name},{_hash(files[name])},{len(files[name])}" if name != record else f"{name},,")
    files[record] = ("\n".join(lines) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", tmp], cwd=root, check=True)
        target = root / "release" / f"Color_Split-{version}_any.whl"
        normalize_wheel(next(Path(tmp).glob("*.whl")), target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
