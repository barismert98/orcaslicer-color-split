# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line: python -m color_split FILE.3mf [--list] [--object N] [--out DIR]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline, threemf
from .solids import NotManifoldError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="color_split", description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--list", action="store_true", help="list objects and exit")
    parser.add_argument("--object", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--surface", choices=("paint", "raise", "inlay"), default="paint",
                        help="surface paint (stripes, logos): leave it painted, raise it, or "
                             "inlay it into the base")
    parser.add_argument("--thickness", type=float, default=0.6,
                        help="height of raised / depth of inlaid surface paint, mm")
    parser.add_argument("--split-joints", action="store_true",
                        help="take buried round parts (ball joints) out whole, cutting the base "
                             "through them")
    args = parser.parse_args(argv)
    try:
        if args.list:
            for index, name in threemf.list_objects(args.file):
                print(f"{index}: {name}")
            return 0
        out = args.out or args.file.with_suffix("").parent / f"{args.file.stem}_split"
        summary = pipeline.run(args.file, args.object, out, split_joints=args.split_joints,
                               surface_mode=args.surface, surface_thickness=args.thickness)
    except (threemf.ThreeMFError, NotManifoldError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in summary.files:
        print(f"{path}  {summary.volumes[path.name]:.1f} mm^3")
    if summary.project is not None:
        print(f"{summary.project}  (all parts as one object: File > Import in OrcaSlicer)")
    for warning in summary.warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
