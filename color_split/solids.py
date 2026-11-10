# SPDX-License-Identifier: GPL-3.0-or-later
"""Color parts and base from a painted object (manifold3d booleans).

``features`` finds the painted features and one cutter per feature; a color part is the
object inside its cutters, the base is the object outside all of them. Both share the cut
surface exactly, so the parts fit back together without any clearance.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import manifold3d
import numpy as np
from scipy.spatial import cKDTree

from . import basepaint, features, surface


class NotManifoldError(ValueError):
    """The object mesh is not a closed manifold, so booleans are impossible."""


class Cancelled(Exception):
    """The caller asked to stop between colors."""


@dataclass
class SplitResult:
    base: manifold3d.Manifold
    inlays: dict[int, manifold3d.Manifold] = field(default_factory=dict)  # color -> part
    findings: features.Findings | None = None
    crumbs: int = 0  # small pieces with real volume dropped
    returned: int = 0  # unpainted pieces caught by a cutter and given back to the base
    bodies: int = 0  # separate bodies painted all over, taken as color parts as they are
    balls: int = 0  # painted parts of a sphere taken as whole balls (joints)
    split_joints: int = 0  # sockets split through a ball buried past its middle
    surface_parts: int = 0  # surface paint made into raised or inlaid parts
    recess_fills: int = 0  # of which painted recesses (engraved text) filled flush
    held_balls: int = 0  # buried balls left painted because joint splitting is off
    painted: basepaint.Surface | None = None  # the model and the paint left on the base
    part_paint: dict[int, set] = field(default_factory=dict)  # part color -> colors painted on it


MIN_PIECE_VOLUME = 0.5  # mm^3; smaller pieces cannot be printed and are boolean debris


def drop_crumbs(solid: manifold3d.Manifold, min_volume: float = MIN_PIECE_VOLUME):
    """Remove disconnected pieces below ``min_volume``; return (kept, real pieces dropped).

    Booleans over many coplanar prism faces leave zero-volume shells and slivers; OrcaSlicer's
    "split to objects" would show each of them as a separate object.
    """
    pieces = solid.decompose()
    kept = [piece for piece in pieces if piece.volume() >= min_volume]
    if len(kept) == len(pieces):
        return solid, 0
    # zero-volume debris is not worth reporting; real small pieces are
    real_dropped = sum(1 for piece in pieces if 1e-6 < piece.volume() < min_volume)
    if not kept:
        return manifold3d.Manifold(), real_dropped
    return manifold3d.Manifold.batch_boolean(kept, manifold3d.OpType.Add), real_dropped


MAX_HOLE_EDGES = 64  # holes up to this many edges are filled before giving up on a mesh


def _as_manifold(V, F) -> manifold3d.Manifold:
    return manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(V, dtype=np.float32),
                                               tri_verts=np.asarray(F, dtype=np.uint32).reshape(-1, 3)))


def repair_mesh(V, F) -> tuple[np.ndarray, np.ndarray, int]:
    """Close the small defects OrcaSlicer repairs silently on load: duplicate vertices,
    degenerate and doubled triangles, and small holes (filled with a fan). Returns
    (V, F, holes filled)."""
    V = np.asarray(V, dtype=np.float64)
    _, vid = np.unique(np.round(V / 1e-6).astype(np.int64), axis=0, return_inverse=True)
    G = vid.reshape(-1)[np.asarray(F, dtype=np.int64).reshape(-1, 3)]
    W = np.zeros((G.max() + 1, 3))
    W[vid.reshape(-1)] = V
    G = G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])]
    _, first = np.unique(np.sort(G, axis=1), axis=0, return_index=True)
    G = G[np.sort(first)]
    half = np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]])
    key = np.sort(half, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    open_half = half[counts[inverse.reshape(-1)] == 1]
    nxt = {}
    for a, b in open_half:
        nxt.setdefault(int(b), []).append(int(a))  # walk the hole against the existing edges
    added, holes = [], 0
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
        if 3 <= len(loop) <= MAX_HOLE_EDGES:
            added += [[loop[0], loop[i], loop[i + 1]] for i in range(1, len(loop) - 1)]
            holes += 1
    if added:
        G = np.vstack([G, np.array(added)])
    return W, G, holes


def _shell_manifold(V, F, report: list | None) -> manifold3d.Manifold:
    solid = _as_manifold(V, F)
    if solid.status() != manifold3d.Error.NoError or solid.is_empty():
        V2, F2, holes = repair_mesh(V, F)
        solid = _as_manifold(V2, F2)
        if solid.status() != manifold3d.Error.NoError or solid.is_empty():
            raise NotManifoldError(f"mesh is not a closed manifold ({solid.status()})")
        if report is not None:
            report.append(f"repaired the mesh (duplicate vertices or faces, {holes} small hole(s))")
    return solid


def _compact(V, F):
    used, inverse = np.unique(np.asarray(F).reshape(-1), return_inverse=True)
    return np.asarray(V)[used], inverse.reshape(-1, 3)


def object_manifold(V, F, report: list | None = None) -> manifold3d.Manifold:
    """The object as one solid. Each separate shell (a part, a body modeled on its own) is
    made solid by itself and the shells are united, so bodies pushed into each other (a stripe
    modeled as a slab through a bar) give their union, not an invalid self-intersecting mesh."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    label = edge_components(F)
    shells = [F[label == l] for l in np.unique(label)]
    if len(shells) == 1:
        return _shell_manifold(V, F, report)
    solids = [_shell_manifold(*_compact(V, f), report) for f in shells]
    return manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add)


def _painted_share(piece: manifold3d.Manifold, painted: cKDTree) -> float:
    """Share of a piece's surface that lies on the painted surface."""
    V, F = to_mesh(piece)
    c = V[F]
    area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
    d, _ = painted.query(c.mean(axis=1), distance_upper_bound=0.05)
    return float(area[np.isfinite(d)].sum() / max(area.sum(), 1e-12))


SPHERE_TOLERANCE = 0.1  # mm a painted point may lie off the fitted ball
SPHERE_SHARE = 0.6  # share of a region's paint that must lie on the ball
BALL_TOLERANCE_SHARE = 0.02  # of the radius: modeled balls are not perfect spheres
MAX_BALL_TOLERANCE = 0.5  # mm
BALL_SHARP_SHARE = 0.15  # share of the border that must be a sharp edge (a patch on a head: ~0)
MIN_BALL_AREA = 20.0  # mm^2 of paint on the ball before it counts as one
BALL_SEGMENTS = 256
JOINT_GAP = 0.02  # mm between the two halves of a split socket


@dataclass
class Ball:
    colour: int
    centre: np.ndarray
    radius: float
    exposed: float  # share of the ball's surface that is painted (visible)
    where: list  # the painted region's center, as in the notes
    cut_radius: float = 0.0  # radius of the cut: the ball plus the model's own deviation


def find_balls(V, F, S, colours, cut, sharp) -> list[Ball]:
    """Painted regions that are part of a sphere (a ball joint half buried in a knee).

    Regions already cut off along an edge (centers in ``cut``) are skipped. A region is a ball
    only if most of its border is a sharp edge (``sharp``: center -> share): a patch painted
    on a round head has a smooth border and is surface paint, not a ball. Real balls are not
    perfect spheres, so the tolerance grows with the radius."""
    a, b, _ = features.edge_pairs(F)
    c = V[F]
    area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
    centroids = c.mean(axis=1)
    balls = []
    for colour in colours:
        painted = S == colour
        region = features.components(len(F), a[painted[a] & painted[b]], b[painted[a] & painted[b]])
        for label in np.unique(region[painted]):
            mine = painted & (region == label)
            if area[mine].sum() < MIN_BALL_AREA:
                continue
            centre_here = np.round(centroids[mine].mean(axis=0), 1).tolist()
            if centre_here in cut or sharp.get(str(centre_here), 0.0) < BALL_SHARP_SHARE:
                continue
            P = V[np.unique(F[mine])]
            A = np.c_[2 * P, np.ones(len(P))]
            sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
            centre = sol[:3]
            radius = float(np.sqrt(max(sol[3] + centre @ centre, 0.0)))
            # refit on the points that are on the ball (a stem or a rim painted too is not)
            for _ in range(2):
                on = np.abs(np.linalg.norm(P - centre, axis=1) - radius) < 5 * SPHERE_TOLERANCE
                if on.sum() < 10:
                    break
                A = np.c_[2 * P[on], np.ones(on.sum())]
                sol, *_ = np.linalg.lstsq(A, (P[on] ** 2).sum(axis=1), rcond=None)
                centre = sol[:3]
                radius = float(np.sqrt(max(sol[3] + centre @ centre, 0.0)))
            # the paint must cover a real part of the ball: a nearly flat patch fits a huge
            # sphere just as well
            extent = float(np.linalg.norm(np.ptp(P, axis=0)))
            if radius < 1.0 or extent < radius:
                continue
            tolerance = min(max(SPHERE_TOLERANCE, BALL_TOLERANCE_SHARE * radius), MAX_BALL_TOLERANCE)
            off = np.abs(np.linalg.norm(centroids[mine] - centre, axis=1) - radius)
            on_ball = area[mine][off < tolerance].sum()
            if on_ball < SPHERE_SHARE * area[mine].sum() or on_ball < MIN_BALL_AREA:
                continue
            # cut just outside the model's own ball surface, so none of it is left as slivers
            on_tris = np.flatnonzero(mine)[off < tolerance]
            deviation = np.linalg.norm(V[np.unique(F[on_tris])] - centre, axis=1) - radius
            spread = min(float(np.percentile(deviation, 99)), tolerance)
            balls.append(Ball(colour, centre, radius, float(on_ball / (4 * np.pi * radius ** 2)),
                              centre_here, radius + max(spread, 0.0) + 0.02))
    return balls


def _split_at(base: manifold3d.Manifold, ball: Ball) -> tuple[manifold3d.Manifold, list]:
    """Cut ``base`` through the ball's center across the limb it sits in, so a ball buried
    more than halfway can be put in: the two halves of its socket come apart.

    The plane only separates the joint itself: where it crosses other geometry (the other
    leg, the body), the pieces are joined back. Returns (base, loose slivers next to the
    ball, which go with the ball)."""
    from .features import _in_polygons
    V, _ = to_mesh(base)
    near = np.linalg.norm(V - ball.centre, axis=1) < 3 * ball.radius
    P = V[near] if near.sum() >= 3 else V
    axis = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)[2][0]  # along the limb
    u = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    frame = np.stack([u, np.cross(axis, u), axis])  # world -> local, the plane is w = 0
    to_local = np.c_[frame, -frame @ ball.centre]
    local = base.transform(to_local.tolist())
    # cross-section regions: the joint is the one around the ball's center
    regions = [r.to_polygons() for r in local.slice(0.0).decompose()]
    joint = [i for i, polys in enumerate(regions)
             if _in_polygons(np.zeros((1, 2)), polys)[0]
             or min((np.linalg.norm(np.asarray(p), axis=1).min() for p in polys if len(p)),
                    default=np.inf) < ball.cut_radius + 1.0]
    halves = [local.trim_by_plane((0.0, 0.0, 1.0), JOINT_GAP / 2),
              local.trim_by_plane((0.0, 0.0, -1.0), JOINT_GAP / 2)]
    slit = local - halves[0] - halves[1]
    pieces = [(side, piece) for side, half in enumerate(halves) for piece in half.decompose()
              if piece.volume() > 1e-9]
    contacts = []
    for _, piece in pieces:
        Vp, _ = to_mesh(piece)
        contacts.append(Vp[np.abs(np.abs(Vp[:, 2]) - JOINT_GAP / 2) < 1e-4][:, :2])

    def region_of(points):
        return {i for i, polys in enumerate(regions) if len(points) and _in_polygons(points, polys).any()}
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    fill = []  # slit pieces that join two pieces back together
    slit_pieces = [p for p in slit.decompose() if p.volume() > 1e-12]
    slit_regions = [region_of(to_mesh(p)[0][:, :2]) for p in slit_pieces]
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if pieces[i][0] == pieces[j][0] or not len(contacts[i]) or not len(contacts[j]):
                continue
            d, _ = cKDTree(contacts[j]).query(contacts[i], distance_upper_bound=JOINT_GAP + 1e-3)
            shared = contacts[i][np.isfinite(d)]
            if len(shared) and not (region_of(shared) & set(joint)):
                parent[find(i)] = find(j)
    for k, sp in enumerate(slit_pieces):
        if not (slit_regions[k] & set(joint)):
            fill.append(sp)
    groups: dict[int, list] = {}
    for i, (_, piece) in enumerate(pieces):
        groups.setdefault(find(i), []).append(piece)
    joined, at_joint = [], []
    for root, g in groups.items():
        merged = manifold3d.Manifold.batch_boolean(g, manifold3d.OpType.Add)
        touching = [sp for sp in fill if merged.min_gap(sp, JOINT_GAP) < 1e-6]
        if len(g) > 1 and touching:
            merged = manifold3d.Manifold.batch_boolean([merged, *touching], manifold3d.OpType.Add)
        joined.append(merged)
        members = [i for i in range(len(pieces)) if find(i) == root]
        at_joint.append(any(region_of(contacts[i]) & set(joint) for i in members))
    # pieces the plane cut loose at the joint (a sliver of the socket rim) go with the ball;
    # other pieces of the base are not touched
    biggest = max([p.volume() for p, j in zip(joined, at_joint) if j], default=0.0)
    loose = [p for p, j in zip(joined, at_joint) if j and p.volume() < 0.02 * biggest]
    kept = [p for p, j in zip(joined, at_joint) if not (j and p.volume() < 0.02 * biggest)]
    to_world = np.c_[frame.T, ball.centre]
    result = manifold3d.Manifold.compose(kept) if len(kept) > 1 else kept[0]
    return result.transform(to_world.tolist()), [p.transform(to_world.tolist()) for p in loose]


CUT_SIDE_AREA = 0.5  # mm^2 of a part's surface that may lie off the model and off the cap


def _cut_through_base(piece: manifold3d.Manifold, surface: cKDTree, cutter) -> float:
    """Area of the piece's surface that is neither the model's surface nor the cap: faces the
    cutter's sides made by slicing through base material still joined to the feature."""
    V, F = to_mesh(piece)
    c = V[F]
    area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
    centre = c.mean(axis=1)
    on_model, _ = surface.query(centre, distance_upper_bound=0.06)
    on_cap = np.zeros(len(centre), dtype=bool)
    for on_surface in cutter.caps:
        on_cap |= on_surface(centre)
    return float(area[~np.isfinite(on_model) & ~on_cap].sum())


CARRIED_SHARE = 0.9  # of a surface-painted area on a cut-off part of its color: it went with it


def _carried_paint(found, V, F, taken) -> None:
    """Surface paint lying on a part already cut off in its own color (the inside walls of a
    cabin painted inside and out) went with that part: it is neither surface paint left on the
    base nor to be raised or inlaid."""
    keep = []
    for colour, tris, where in found.surface_regions:
        mine = [part for c, part in taken if c == colour]
        carried = False
        if mine:
            pts = features.samples(V[F[tris]], 0.3)
            for part in mine:
                PV, PF = to_mesh(part)
                d, _ = cKDTree(features.samples(PV[PF], 0.1)).query(pts, distance_upper_bound=0.1)
                if np.isfinite(d).mean() >= CARRIED_SHARE:
                    carried = True
                    break
        if not carried:
            keep.append((colour, tris, where))
            continue
        found.surface[colour] = found.surface.get(colour, 0) - 1
        if found.surface[colour] <= 0:
            del found.surface[colour]
    found.surface_regions = keep


def split(obj: manifold3d.Manifold, leaves, states, base_state: int,
          cancelled: Callable[[], bool] = lambda: False, split_joints: bool = False,
          surface_mode: str = "paint", surface_thickness: float = 0.6) -> SplitResult:
    """Cut every painted feature off ``obj``; paint on smooth surfaces stays on the base."""
    V, F, parent = features.conform(leaves)
    S = np.asarray(states, dtype=np.int64)[parent]
    add = manifold3d.OpType.Add
    # A separate body painted all over in one color (a stripe modeled as a slab through a
    # bar) is that color's part as it is; the rest of the object loses its volume.
    a, b, _ = features.edge_pairs(F)
    shell = features.components(len(F), a, b)
    whole = np.zeros(len(F), dtype=bool)
    bodies: dict[int, list] = {}
    for label in np.unique(shell):
        mine = shell == label
        states_here = np.unique(S[mine])
        if len(states_here) == 1 and int(states_here[0]) not in (0, base_state):
            whole |= mine
            body = _as_manifold(*_compact(V, F[mine]))
            if body.status() == manifold3d.Error.NoError and not body.is_empty():
                bodies.setdefault(int(states_here[0]), []).append(body)
    F, S = F[~whole], S[~whole]
    # Raise and Inlay are asked for smooth paint by name: only the default cuts it off
    found = (features.find_features(V, F, S, base_state, cut_smooth=surface_mode == "paint")
             if len(F) else features.Findings(states=S))
    result = SplitResult(base=obj, findings=found)
    left_paint = np.array(found.states)  # the paint left on the base at the end
    everything, returned = [], []
    colours = sorted(int(c) for c in np.unique(found.states) if c not in (0, base_state))
    # with joints on, a ball buried past its middle comes out whole (its socket split) rather
    # than as the film-cut visible part
    joint_balls = []
    if split_joints and len(F):
        primary = [c.where for f in found.features for c in f.cutters if not c.film]
        joint_balls = [b for b in find_balls(V, F, found.states, colours, primary, found.sharp)
                       if b.exposed < 0.5]
    skip = {str(b.where) for b in joint_balls}
    for colour, parts in sorted(bodies.items()):
        part = manifold3d.Manifold.batch_boolean(parts, add) ^ obj
        everything.append(part)
        result.inlays[colour] = part
        result.bodies += len(parts)
    taken = []  # (color, part) cut off along edges
    for feature in found.features:
        painted = cKDTree(features.samples(V[F[found.states == feature.colour]], 0.1))
        for cutter in feature.cutters:
            if cancelled():
                raise Cancelled()
            if str(cutter.where) in skip:
                continue

            def attempt(cutter):
                kept, back = [], []
                for piece in (obj ^ cutter.solid).decompose():
                    if piece.volume() < MIN_PIECE_VOLUME:
                        if piece.volume() > 1e-9:
                            back.append(piece)  # a sliver the cut shaved off: keep it in the base
                    elif _painted_share(piece, painted) < 0.01:
                        back.append(piece)  # caught by the cutter but not painted: base material
                    else:
                        kept.append(piece)
                # the model's surface around this cut, sampled finely enough that any point on
                # it is within 0.04 mm of a sample
                box = cutter.solid.bounding_box()
                lo, hi = np.asarray(box[:3]), np.asarray(box[3:])
                tri_lo, tri_hi = V[F].min(axis=1), V[F].max(axis=1)
                near = np.all((tri_hi >= lo - 0.1) & (tri_lo <= hi + 0.1), axis=1)
                surface = cKDTree(features.samples(V[F[near]], 0.05))
                return kept, back, sum(_cut_through_base(p, surface, cutter) for p in kept)
            kept, back, sliced = attempt(cutter)
            crumbs = 0
            if sliced > CUT_SIDE_AREA and cutter.fallback is not None:
                film = cutter.fallback()
                if film is not None:
                    film_kept, film_back, film_sliced = attempt(film)
                    if film_sliced <= CUT_SIDE_AREA and film_kept:
                        film.mix = cutter.mix
                        cutter, kept, back, sliced = film, film_kept, film_back, film_sliced
            if cutter.film and sliced <= CUT_SIDE_AREA and kept:
                found.films += 1
            if sliced > CUT_SIDE_AREA:
                # the cut would slice through base geometry joined to the feature (a bridge
                # resting on it, a wall right beside it): leave the feature on the base
                found.notes.append(f"color {feature.colour} at {cutter.where}: the cut would "
                                   f"slice {sliced:.1f} mm^2 through base geometry")
                found.unsafe[feature.colour] = found.unsafe.get(feature.colour, 0) + 1
                found.failed.append(cutter.where)
                continue
            if not kept:
                continue
            result.crumbs += crumbs
            everything.append(cutter.solid)
            returned += back
            part = manifold3d.Manifold.batch_boolean(kept, add)
            taken.append((feature.colour, part))
            shares = {feature.colour: part}
            if cutter.mix:
                # several colors: each smaller one takes what its film claims, the main
                # color the rest, so the colors meet on one surface inside the feature
                for colour, mask in sorted(cutter.mix.items(), key=lambda kv: found.areas[kv[1]].sum()):
                    if colour == feature.colour:
                        continue
                    claim = features.colour_film(V, F, found.normals, found.areas, mask)
                    if claim is None:
                        found.notes.append(f"color {colour} at {cutter.where}: could not be "
                                           f"separated from color {feature.colour}")
                        continue
                    piece, _ = drop_crumbs(shares[feature.colour] ^ claim)
                    if piece.is_empty():
                        # too thin to be a part (a pupil on a head): it stays as paint on the
                        # part around it, which keeps its volume
                        result.part_paint.setdefault(feature.colour, set()).add(colour)
                        found.notes.append(f"color {colour} at {cutter.where}: too thin to be "
                                           f"a part; it stays painted on the color "
                                           f"{feature.colour} part")
                        continue
                    shares[feature.colour] = shares[feature.colour] - piece
                    shares[colour] = piece
            for colour, share in shares.items():
                share, _ = drop_crumbs(share)
                if share.is_empty():
                    continue
                old = result.inlays.get(colour)
                result.inlays[colour] = share if old is None else old + share
    _carried_paint(found, V, F, taken)
    # rescue: paint that could not be cut off along an edge but is part of a ball (a knee
    # joint) becomes that ball; a ball buried more than halfway also splits its socket
    cut = [c.where for f in found.features for c in f.cutters
           if c.where not in found.failed and str(c.where) not in skip]
    rest = find_balls(V, F, found.states, colours, cut, found.sharp) if len(F) else []
    balls = joint_balls + [b for b in rest if str(b.where) not in skip]
    for ball in balls:
        # the region is accounted for here, not as a failed cut
        counts = found.unsafe if ball.where in found.failed else found.surface
        counts[ball.colour] = counts.get(ball.colour, 0) - 1
        if counts[ball.colour] <= 0:
            del counts[ball.colour]
        found.notes = [n for n in found.notes if f" at {ball.where}:" not in n]
        if ball.exposed < 0.5 and not split_joints:
            # buried past its middle: it only comes out if its socket may be cut in two (an
            # option; a painted eye buried in a head looks just like a knee ball)
            result.held_balls += 1
            continue
        # many segments: the chord gap of the cut sphere stays far below the 0.02 mm margin
        sphere = manifold3d.Manifold.sphere(ball.cut_radius, BALL_SEGMENTS).translate(tuple(ball.centre))
        part = obj ^ sphere
        if part.is_empty():
            continue
        everything.append(sphere)
        old = result.inlays.get(ball.colour)
        result.inlays[ball.colour] = part if old is None else old + part
        result.balls += 1
    # surface paint, on request, like OrcaSlicer's text tool: a raised layer, or an inlay
    if surface_mode in ("raise", "inlay"):
        handled = {str(b.where) for b in balls}
        for colour, tris, where in found.surface_regions:
            if str(where) in handled:
                continue
            if cancelled():
                raise Cancelled()
            recess = found.recesses.get(str(where))
            if recess is not None:
                # paint lining a recess (engraved text): fill it flush, whatever the thickness
                fill = features.recess_fill(V, F, found.normals, found.areas, *recess)
                part, cutter = (None if fill is None else fill - obj), None
                if part is not None and part.is_empty():
                    part = None
                if part is not None:
                    result.recess_fills += 1
            elif surface_mode == "raise":
                part, cutter = surface.raised(obj, V, F[tris], surface_thickness), None
            else:
                part, cutter = surface.inlaid(obj, V, F[tris], surface_thickness)
            if part is None:
                found.notes.append(f"color {colour} at {where}: could not "
                                   f"{'raise' if surface_mode == 'raise' else 'inlay'} this paint")
                continue
            part, dropped = drop_crumbs(part)
            if part.is_empty():
                continue
            if cutter is not None:
                everything.append(cutter)
            old = result.inlays.get(colour)
            result.inlays[colour] = part if old is None else old + part
            result.surface_parts += 1
            left_paint[tris] = base_state
            found.surface[colour] = found.surface.get(colour, 0) - 1
            if found.surface[colour] <= 0:
                del found.surface[colour]
    if everything:
        base = obj - manifold3d.Manifold.batch_boolean(everything, add)
        for ball in balls:
            if ball.exposed < 0.5 and split_joints:  # it cannot go into one socket
                base, loose = _split_at(base, ball)
                if loose:
                    result.inlays[ball.colour] = manifold3d.Manifold.batch_boolean(
                        [result.inlays[ball.colour], *loose], add)
                result.split_joints += 1
        if returned:
            base = manifold3d.Manifold.batch_boolean([base, *returned], add)
            result.returned = sum(1 for p in returned if p.volume() >= MIN_PIECE_VOLUME)
        result.base, dropped = drop_crumbs(base)
        result.crumbs += dropped
    if len(F):
        result.painted = basepaint.Surface(V, F, left_paint, found.normals)
    return result


class ExportError(RuntimeError):
    """A part could not be turned into a watertight STL mesh; nothing should be written."""


_SEPARATION = 2e-4  # mm; far above float32 spacing (~1.5e-5 mm at 250 mm), far below print size


def stl_defects(V, F) -> int:
    """Edges not shared by exactly two triangles, plus collapsed triangles, after the mesh is
    stored as float32 and welded by exact coordinate the way STL readers (OrcaSlicer) do."""
    V32 = np.asarray(V, dtype=np.float32)
    _, vid = np.unique(V32, axis=0, return_inverse=True)
    G = vid.reshape(-1)[np.asarray(F, dtype=np.int64)]
    collapsed = int(((G[:, 0] == G[:, 1]) | (G[:, 1] == G[:, 2]) | (G[:, 0] == G[:, 2])).sum())
    edges = np.sort(np.concatenate([G[:, [0, 1]], G[:, [1, 2]], G[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts != 2).sum()) + collapsed


def export_mesh(solid: manifold3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    """Watertight mesh for STL, or ``ExportError``.

    manifold3d's topology is always closed, but where solids touch along an edge or point it
    keeps separate vertices at the same position, and booleans can leave vertices only
    ~1e-6 mm apart. STL stores no topology, so a reader welds those into non-manifold or open
    edges. Each vertex that shares its float32 position with another is moved ``_SEPARATION``
    mm inward along its own normal (rank-scaled within the group), which keeps the topology
    intact through the STL round trip. The result is verified the way a reader sees it.
    """
    mesh = solid.to_mesh64() if hasattr(solid, "to_mesh64") else solid.to_mesh()
    V = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3].copy()
    F = np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
    if len(F) == 0:
        return V, F
    V, F = clean_mesh(V, F)
    for attempt in range(10):
        _, group, counts = np.unique(V.astype(np.float32), axis=0, return_inverse=True,
                                     return_counts=True)
        group = group.reshape(-1)
        clash = counts[group] > 1
        if not clash.any():
            break
        c = V[F]
        fn = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
        vn = np.zeros_like(V)
        np.add.at(vn, F.reshape(-1), np.repeat(fn, 3, axis=0))
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)
        idx = np.flatnonzero(clash)
        order = idx[np.argsort(group[idx], kind="stable")]
        g = group[order]
        start = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
        rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
        V[order] -= vn[order] * (_SEPARATION * (rank + 1))[:, None]
        if attempt >= 3:
            # copies pushed along the same normal can meet again (two thin sheets touching
            # along an edge): also move them sideways, along the surface, by rank
            side = np.cross(vn[order], np.array([0.267, 0.535, 0.802]))
            side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
            V[order] += side * (_SEPARATION * (rank + 1) * (attempt - 2))[:, None]
    defects = stl_defects(V, F)
    if defects:
        raise ExportError(f"{defects} open or shared edges would remain in the STL")
    return V, F


def edge_components(F) -> np.ndarray:
    """Label triangles by connectivity over shared edges, the way OrcaSlicer's
    "split to objects" separates parts (a single shared vertex does not connect)."""
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    edges = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=2)
    _, edge_id = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
    edge_id = edge_id.reshape(-1)
    tri_of = np.repeat(np.arange(len(F)), 3)
    order = np.argsort(edge_id, kind="stable")
    same = edge_id[order][1:] == edge_id[order][:-1]
    a, b = tri_of[order][:-1][same], tri_of[order][1:][same]
    label = np.arange(len(F))
    while len(a):
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        label = label[label]
        if np.array_equal(before, label):
            break
    return label


def clean_mesh(V, F, min_volume: float = MIN_PIECE_VOLUME) -> tuple[np.ndarray, np.ndarray]:
    """Drop edge-connected components whose enclosed volume is below ``min_volume``.

    Final guard before writing: float32 output can collapse thin boolean slivers into
    zero-volume shells that OrcaSlicer's "split to objects" would list as separate objects.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    if len(F) == 0:
        return V, F
    tri_label = edge_components(F)
    c = V[F]
    signed = np.einsum("ij,ij->i", c[:, 0], np.cross(c[:, 1], c[:, 2])) / 6.0
    ids, inverse = np.unique(tri_label, return_inverse=True)
    volume = np.bincount(inverse, signed)
    keep_tri = np.abs(volume[inverse]) >= min_volume
    F = F[keep_tri]
    used = np.unique(F)
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def to_mesh(solid: manifold3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh()
    V = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    return V, np.asarray(mesh.tri_verts, dtype=np.int64).reshape(-1, 3)
