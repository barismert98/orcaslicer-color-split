# SPDX-License-Identifier: GPL-3.0-or-later
"""Split one painted object of a 3MF into inlay STLs and a pocketed base STL."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import basepaint, paint, paintfix, project3mf, solids, surface, threemf
from .solids import Cancelled, ExportError
from .stl import write_stl

__all__ = ["Cancelled", "ExportError", "Scan", "Summary", "run", "scan"]


@dataclass
class Summary:
    object_name: str
    files: list[Path] = field(default_factory=list)
    volumes: dict[str, float] = field(default_factory=dict)
    skipped: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    project: Path | None = None  # the parts as one colored object, for File > Import
    pieces: dict[str, int] = field(default_factory=dict)  # file name -> separate pieces
    base: int = 1  # the object's own filament
    shares: dict[int, float] = field(default_factory=dict)  # filament -> percent of the surface


@dataclass
class Scan:
    name: str
    base: int
    shares: dict[int, float]  # filament -> percent of the surface


def paint_area(corners) -> float:
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 3, 3)
    return float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                               corners[:, 2] - corners[:, 0]), axis=1).sum())


def surface_shares(leaves, states) -> dict[int, float]:
    """Percent of the surface on each filament, from the decoded paint leaves."""
    leaves, states = np.asarray(leaves, dtype=np.float64).reshape(-1, 3, 3), np.asarray(states)
    areas = 0.5 * np.linalg.norm(np.cross(leaves[:, 1] - leaves[:, 0],
                                          leaves[:, 2] - leaves[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0:
        return {}
    return {int(s): 100.0 * float(areas[states == s].sum()) / total for s in np.unique(states)}


def _decoded(obj: threemf.ObjectMesh):
    """The paint leaves as the split sees them: brush specks already cleaned away."""
    leaves, states = paint.split_mesh(obj.V, obj.F, obj.codes, obj.part_extruders)
    return (leaves, *paintfix.clean_states(leaves, states))


def scan(threemf_path, index: int) -> Scan:
    """Which filaments object ``index`` is painted with, and how much of its surface each has."""
    obj = threemf.read_object(threemf_path, index)
    leaves, states, _ = _decoded(obj)
    return Scan(obj.name, obj.extruder, surface_shares(leaves, states))


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\|?*\x00-\x1f]+', "_", name).strip(" .") or "object"


def run(threemf_path, index: int, out_dir,
        progress: Callable[[int, int, str], bool] = lambda step, total, msg: True,
        split_joints: bool = False, surface_mode: str = "paint",
        surface_thickness: float = surface.DEFAULT_THICKNESS) -> Summary:
    """Read object ``index``, split it, and write the STLs into ``out_dir``.

    ``progress(step, total, message)`` returning False cancels (raises ``Canceled``);
    nothing is written unless every part was built.
    """
    if surface_mode not in ("paint", "raise", "inlay"):
        raise ValueError(f"surface paint mode must be paint, raise or inlay, not {surface_mode!r}")
    lo, hi = surface.THICKNESS_RANGE
    if not lo <= surface_thickness <= hi:
        raise ValueError(f"surface paint thickness must be between {lo} and {hi} mm")
    total = 4

    def step(n: int, message: str) -> None:
        if not progress(n, total, message):
            raise Cancelled()

    step(0, "reading project")
    obj = threemf.read_object(threemf_path, index)
    summary = Summary(object_name=obj.name)
    step(1, "decoding paint")
    leaves, states, islands = _decoded(obj)
    summary.base, summary.shares = obj.extruder, surface_shares(leaves, states)
    repairs: list[str] = []
    solid = solids.object_manifold(obj.V, obj.F, repairs)
    step(2, "building parts")
    result = solids.split(solid, leaves, states, obj.extruder,
                          cancelled=lambda: not progress(2, total, "building parts"),
                          split_joints=split_joints, surface_mode=surface_mode,
                          surface_thickness=surface_thickness)
    found = result.findings
    w = summary.warnings
    w.extend(repairs)
    if islands:
        w.append(f"cleaned up {islands} tiny paint island(s) left by the brush")
    for colour, where in found.islands:
        w.append(f"color {colour}: an unpainted patch at {where} lies inside the painted area "
                 "and is printed in that color with it")
    if found.snapped:
        w.append(f"snapped {found.snapped} paint strip(s) to the edge next to them")
    for colour, n in sorted(found.surface.items()):
        w.append(f"color {colour}: {n} painted area(s) are surface paint (a stripe, a ring, "
                 "a hole lining) and stay painted on the base in the split project (the STL "
                 "has no paint); set Surface paint to Raise or Inlay to make them parts")
    for colour, n in sorted(found.unsafe.items()):
        w.append(f"color {colour}: {n} feature(s) could not be cut safely and stay on the base")
    w.extend(found.notes)
    if found.films:
        w.append(f"cut {found.films} painted part(s) along a surface stretched across their "
                 "border, closing the opening")
    if result.surface_parts:
        kind = "raised" if surface_mode == "raise" else "inlaid"
        made = result.surface_parts - result.recess_fills
        if made:
            w.append(f"{made} surface-painted area(s) {kind} {surface_thickness:g} mm")
        if result.recess_fills:
            w.append(f"filled {result.recess_fills} painted recess(es) (engraved text, a "
                     "pocket) flush with the surface")
    if result.bodies:
        w.append(f"took {result.bodies} separately modeled body(ies) painted all over as color "
                 "parts as they are")
    if result.balls:
        w.append(f"took {result.balls} buried round part(s) out whole")
    if result.held_balls:
        w.append(f"{result.held_balls} buried painted part(s) could not be cut along their "
                 "border and stay painted; 'Take buried parts out whole' can take round ones out "
                 "(it cuts the base through them)")
    if result.split_joints:
        w.append(f"cut the base through {result.split_joints} buried part(s), so each fits "
                 "between the two halves of its socket")
    if result.returned:
        w.append(f"gave {result.returned} unpainted piece(s) caught by a cut back to the base")
    if result.crumbs:
        w.append(f"removed {result.crumbs} crumb(s) smaller than {solids.MIN_PIECE_VOLUME} mm^3")
    if not result.inlays and not result.held_balls:
        w.append("nothing to split off: no painted feature stands on a sharp edge")
    step(3, "writing files")
    out_dir = Path(out_dir)
    stem = _safe(obj.name)
    parts = [(f"{stem}_base.stl", result.base, obj.extruder)] + [
        (f"{stem}_color{colour}.stl", inlay, colour) for colour, inlay in sorted(result.inlays.items())]
    meshes = []
    for name, m, extruder in parts:
        if m.is_empty():
            continue
        V, F = solids.export_mesh(m)
        codes = None
        if m is result.base and result.painted is not None:
            F, codes = basepaint.transfer(V, F, obj.V, obj.F, obj.codes, obj.part_extruders,
                                          result.painted, obj.extruder,
                                          verbatim=not result.surface_parts)
        elif result.painted is not None and extruder in result.part_paint:
            # colors too thin to be parts stay painted on the part around them
            only = np.isin(result.painted.states, list(result.part_paint[extruder]))
            on_part = replace(result.painted, states=np.where(only, result.painted.states, extruder))
            F, codes = basepaint.transfer(V, F, obj.V, obj.F, obj.codes, obj.part_extruders,
                                          on_part, extruder, verbatim=False)
        meshes.append((name, (V, F), m.volume(), extruder, codes))
    out_dir.mkdir(parents=True, exist_ok=True)
    project = out_dir / f"{stem}_split.3mf"
    for old in [out_dir / f"{stem}_base.stl", project, *out_dir.glob(f"{stem}_color*.stl")]:
        old.unlink(missing_ok=True)  # files from an earlier run would otherwise be loaded too
    for name, (V, F), volume, _, _ in meshes:
        write_stl(out_dir / name, V, F)
        summary.files.append(out_dir / name)
        summary.volumes[name] = volume
    # one object with every part on its extruder: File > Import loads it aligned and colored
    # every separate piece its own part, so each can be printed (or moved) on its own
    parts = []
    for name, (V, F), _, extruder, codes in meshes:
        label = solids.edge_components(F)
        pieces = np.unique(label)
        summary.pieces[name] = len(pieces)
        for n, piece in enumerate(pieces, start=1):
            Vp, Fp = solids._compact(V, F[label == piece])
            suffix = f"_{n}" if len(pieces) > 1 else ""
            part_codes = None if codes is None else [codes[t] for t in np.flatnonzero(label == piece)]
            parts.append(project3mf.Part(Path(name).stem + suffix, Vp, Fp, extruder, part_codes))
    # baked world coordinates, set beside the original (to its +x side) so both stay visible
    allV = np.vstack([p.V for p in parts]) if parts else np.zeros((1, 3))
    beside = (float(np.ptp(allV[:, 0])) + 10.0, 0.0, 0.0)
    summary.project = project3mf.write_project(project, f"{obj.name} (split)", parts,
                                               baked=True, offset=beside)
    return summary
