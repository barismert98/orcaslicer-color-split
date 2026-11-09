"""Painted features cut off along the sharp edge they stand on."""
import manifold3d
import numpy as np
import pytest

from color_split import features, solids
from color_split.solids import edge_components, export_mesh, stl_defects, to_mesh


# the cut sits LIFT above the block top, so the part is that much shorter
PART = 6 * 6 * (8 - features.LIFT)


def tower(bore=False, refine=1):
    """20x20x10 block with a 6x6x8 tower on top (optionally a blind bore 3 mm into the block)."""
    block = manifold3d.Manifold.cube((20, 20, 10))
    t = manifold3d.Manifold.cube((6, 6, 8)).translate((7, 7, 10))
    m = block + t
    if bore:
        m = m - manifold3d.Manifold.cylinder(11, 1.5, 1.5, 32).translate((10, 10, 7))
    return m.refine(refine) if refine > 1 else m


def leaves_of(m):
    V, F = to_mesh(m)
    return V[F]


def in_tower(c, margin=1e-6):
    xy = c[:, :2]
    return (np.all((xy > 7 - margin) & (xy < 13 + margin), axis=1)) & (c[:, 2] > 10 - 1e-6)


def run(m, painted, **options):
    leaves = leaves_of(m)
    states = np.where(painted(leaves.mean(axis=1)), 2, 0)
    return solids.split(m, leaves, states, base_state=1, **options), m


def watertight(solid):
    V, F = export_mesh(solid)
    return stl_defects(V, F) == 0 and len(np.unique(edge_components(F))) == 1


def test_painted_tower_is_cut_at_its_foot():
    result, m = run(tower(), lambda c: in_tower(c) & (c[:, 2] > 10 + 1e-6))
    part = result.inlays[2]
    assert part.volume() == pytest.approx(PART, rel=1e-4)
    assert result.base.volume() == pytest.approx(m.volume() - PART, rel=1e-4)
    assert (part ^ result.base).volume() < 1e-6
    assert watertight(part) and watertight(result.base)


def test_spilled_and_short_paint_snaps_to_the_edge():
    m = tower(refine=60)  # small triangles so the brush can miss or spill
    # spill 0.3 mm onto the block top, and stop 0.3 mm short of the foot on one side
    spill = lambda c: in_tower(c, 0.3) & ~((c[:, 0] < 7.5) & (c[:, 2] < 10.3))
    result, _ = run(m, spill)
    assert result.findings.snapped >= 1
    assert result.inlays[2].volume() == pytest.approx(PART, rel=1e-3)
    assert watertight(result.base)


def test_unpainted_bore_goes_with_the_feature_above_the_cut():
    result, m = run(tower(bore=True), lambda c: in_tower(c) & (np.linalg.norm(c[:, :2] - 10, axis=1) > 1.6)
                    & (c[:, 2] > 10 + 1e-6))
    expected = PART - np.pi * 1.5 ** 2 * (8 - features.LIFT)
    assert result.inlays[2].volume() == pytest.approx(expected, rel=2e-2)
    assert watertight(result.inlays[2]) and watertight(result.base)


def test_tilted_model_cuts_the_same():
    m = tower().rotate((20, 10, 0))
    V, F = to_mesh(m)
    back = np.linalg.inv(np.asarray(_rotation(20, 10)))
    leaves = V[F]
    c = leaves.mean(axis=1) @ back.T
    states = np.where(in_tower(c) & (c[:, 2] > 10 + 1e-6), 2, 0)
    result = solids.split(m, leaves, states, base_state=1)
    assert result.inlays[2].volume() == pytest.approx(PART, rel=1e-3)


def _rotation(x, y):
    rx, ry = np.radians(x), np.radians(y)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return Ry @ Rx  # manifold rotates about x, then y


def test_stripe_on_a_flat_face_stays_on_the_base():
    m = manifold3d.Manifold.cube((20, 20, 10)).refine(8)
    result, _ = run(m, lambda c: (c[:, 2] > 10 - 1e-6) & (np.abs(c[:, 0] - 10) < 2))
    assert not result.inlays
    assert result.findings.surface == {2: 1}
    assert result.base.volume() == pytest.approx(4000, rel=1e-6)


def test_lining_of_a_hole_is_surface_paint():
    plate = (manifold3d.Manifold.cube((40, 40, 6), True)
             - manifold3d.Manifold.cylinder(10, 5, 5, 48, True)).refine(4)
    result, _ = run(plate, lambda c: np.linalg.norm(c[:, :2], axis=1) < 5.2)
    assert not result.inlays and result.findings.surface == {2: 1}


def bridged(gap):
    """The tower under a bridge ``gap`` mm above its top."""
    m = tower() + manifold3d.Manifold.cube((20, 4, 2)).translate((0, 8, 18 + gap))
    return m + manifold3d.Manifold.cube((2, 4, 8 + gap)).translate((0, 8, 10))


def painted_tower(c):
    return in_tower(c) & (c[:, 2] > 10 + 1e-6) & (c[:, 2] < 18 + 1e-6)


def test_feature_under_a_bridge_is_cut():
    result, _ = run(bridged(0.5), painted_tower)
    assert result.inlays[2].volume() == pytest.approx(PART, rel=1e-3)
    assert watertight(result.base)


def test_neighbour_too_close_is_handled_by_the_film():
    # a wall 0.01 mm beside the tower: the cutter would slice into it, so the feature is cut
    # along a film across its foot instead, which never leaves the feature
    m = tower() + manifold3d.Manifold.cube((1, 6, 12)).translate((13.01, 7, 10))
    result, _ = run(m, painted_tower)
    assert result.findings.films == 1
    assert result.inlays[2].volume() == pytest.approx(6 * 6 * 8, rel=1e-3)
    assert watertight(result.inlays[2]) and watertight(result.base)


def test_ball_joint_between_two_parts_is_cut_from_both():
    # a ball joining two bars (a knee): the ball comes off at both bars, the bars stay base
    bar = manifold3d.Manifold.cube((8, 8, 10))
    ball = manifold3d.Manifold.sphere(6, 64).translate((4, 4, 15))
    m = bar + ball + bar.translate((0, 0, 20))
    result, _ = run(m, lambda c: (c[:, 2] > 10 + 1e-6) & (c[:, 2] < 20 - 1e-6))
    part = result.inlays[2]
    assert len(part.decompose()) == 1
    assert len(result.base.decompose()) == 2
    assert part.volume() + result.base.volume() == pytest.approx(m.volume(), rel=1e-6)
    assert (part ^ result.base).volume() < 1e-3
    assert watertight(part)


def knee():
    """Thigh and shin joined behind a ball buried past its middle; the paint wraps around
    the ball, so there is no single edge to cut along."""
    thigh = manifold3d.Manifold.cube((10, 10, 21))
    shin = manifold3d.Manifold.cube((10, 10, 21)).translate((0, 0, 25))
    back = manifold3d.Manifold.cube((10, 3, 46))
    ball = manifold3d.Manifold.sphere(6, 96).translate((5, 6, 23))
    return thigh + shin + back + ball


def on_knee_ball(c):
    return np.abs(np.linalg.norm(c - [5, 6, 23], axis=1) - 6) < 0.05


def test_paint_winding_around_a_buried_ball_is_cut_along_a_film():
    # by default the visible painted part comes off along a film across its border; the base
    # is not cut and nothing is trapped (the part touches the base only through the film)
    result, m = run(knee(), on_knee_ball)
    assert result.findings.films == 1 and result.split_joints == 0
    part = result.inlays[2]
    assert 0 < part.volume() < 4 / 3 * np.pi * 6 ** 3
    assert len(result.base.decompose()) == 1
    assert part.volume() + result.base.volume() == pytest.approx(m.volume(), rel=1e-3)
    assert watertight(part) and watertight(result.base)


def test_buried_ball_joint_comes_out_whole_and_splits_its_socket():
    m = knee()
    result, _ = run(m, on_knee_ball, split_joints=True)
    part = result.inlays[2]
    assert result.balls == 1 and result.split_joints == 1
    assert part.volume() == pytest.approx(4 / 3 * np.pi * 6 ** 3, rel=2e-2)
    assert len(result.base.decompose()) == 2  # the two halves of the socket come apart
    assert part.volume() + result.base.volume() == pytest.approx(m.volume(), rel=1e-4)


def test_paint_on_a_round_head_is_not_a_ball():
    # a patch painted on a sphere-shaped body: the body continues over the same sphere
    head = manifold3d.Manifold.sphere(10, 96) + manifold3d.Manifold.cube((6, 6, 10), True).translate((0, 0, -12))
    result, _ = run(head, lambda c: c[:, 0] > 6)
    assert result.balls == 0 and 2 in result.inlays  # cut along its border, not as a ball


def test_nearly_flat_paint_is_not_a_huge_ball():
    # a slightly bulged plate: its paint fits a sphere with a huge radius, but is no ball
    plate = manifold3d.Manifold.cube((30, 30, 4), True).refine(8).warp(
        lambda p: (p[0], p[1], p[2] + 0.002 * (p[0] ** 2 + p[1] ** 2) * (p[2] > 0)))
    plate = plate + manifold3d.Manifold.cube((6, 6, 6)).translate((-3, -3, 1))
    result, _ = run(plate, lambda c: (c[:, 2] > 2.5) & (np.abs(c[:, 0]) < 3) & (np.abs(c[:, 1]) < 3))
    assert result.balls == 0


def flat_stripe():
    m = manifold3d.Manifold.cube((20, 20, 10)).refine(10)
    return m, lambda c: (c[:, 2] > 10 - 1e-6) & (np.abs(c[:, 0] - 10) < 2)


@pytest.mark.parametrize("mode", ["raise", "inlay"])
def test_surface_paint_can_be_raised_or_inlaid(mode):
    m, stripe = flat_stripe()
    result, _ = run(m, stripe, surface_mode=mode, surface_thickness=0.6)
    part = result.inlays[2]
    assert result.surface_parts == 1 and not result.findings.surface
    assert part.volume() == pytest.approx(4 * 20 * 0.6, rel=0.02)
    assert watertight(part) and watertight(result.base)
    if mode == "raise":
        assert result.base.volume() == pytest.approx(4000, rel=1e-6)  # base unchanged
        assert part.bounding_box()[5] == pytest.approx(10.6, abs=0.01)
    else:
        assert part.volume() + result.base.volume() == pytest.approx(4000, rel=1e-6)
        assert part.bounding_box()[2] == pytest.approx(9.4, abs=0.01)
    assert (part ^ result.base).volume() < 1e-3


@pytest.mark.parametrize("mode", ["raise", "inlay"])
def test_stripe_around_a_rod_follows_the_curve(mode):
    rod = manifold3d.Manifold.cylinder(30, 5, 5, 96).refine_to_length(0.5)
    result, _ = run(rod, lambda c: (np.abs(c[:, 2] - 15) < 3) & (np.linalg.norm(c[:, :2], axis=1) > 4.9),
                    surface_mode=mode, surface_thickness=0.6)
    part = result.inlays[2]
    ring = np.pi * ((5.6 ** 2 - 25) if mode == "raise" else (25 - 4.4 ** 2)) * 6
    assert part.volume() == pytest.approx(ring, rel=0.03)
    assert len(part.decompose()) == 1 and watertight(part) and watertight(result.base)


def test_surface_paint_stays_painted_by_default():
    m, stripe = flat_stripe()
    result, _ = run(m, stripe)
    assert not result.inlays and result.findings.surface == {2: 1}


def test_painted_engraving_is_filled_flush_when_inlaid():
    block = manifold3d.Manifold.cube((30, 20, 6))
    slot = manifold3d.Manifold.cube((20, 4, 1)).translate((5, 8, 5))
    m = (block - slot).refine_to_length(0.5)
    inside = lambda c: (np.abs(c[:, 0] - 15) < 10 + 1e-4) & (np.abs(c[:, 1] - 10) < 2 + 1e-4) & \
        (c[:, 2] > 5 - 1e-4) & (c[:, 2] < 6 - 1e-4)
    kept, _ = run(m, inside)
    assert not kept.inlays  # by default it stays painted
    result, _ = run(m, inside, surface_mode="inlay")
    assert result.recess_fills == 1
    assert result.inlays[2].volume() == pytest.approx(20 * 4 * 1, rel=1e-3)
    assert result.base.volume() == pytest.approx(m.volume(), rel=1e-6)  # the base keeps its slot


def test_touching_colours_are_cut_as_one_feature_and_split_between_them():
    m = tower().refine_to_length(0.5)
    leaves = leaves_of(m)
    c = leaves.mean(axis=1)
    t = in_tower(c) & (c[:, 2] > 10 + 1e-6)
    states = np.where(t & (c[:, 0] < 10), 2, np.where(t, 3, 0))
    result = solids.split(m, leaves, states, base_state=1)
    red, blue = result.inlays[2], result.inlays[3]
    assert red.volume() + blue.volume() == pytest.approx(6 * 6 * 8, rel=1e-2)
    assert result.base.volume() == pytest.approx(m.volume() - 6 * 6 * 8, rel=1e-3)  # no core left
    assert (red ^ blue).volume() < 1e-3 and watertight(red) and watertight(blue)


def test_unpainted_patch_inside_a_feature_is_reported():
    m = tower().refine_to_length(0.5)
    dot = lambda c: (c[:, 1] < 7 + 1e-4) & (np.abs(c[:, 0] - 10) < 1.5) & (np.abs(c[:, 2] - 14) < 1.5)
    result, _ = run(m, lambda c: painted_tower(c) & ~dot(c))
    assert result.inlays[2].volume() == pytest.approx(6 * 6 * 8, rel=1e-2)
    assert len(result.findings.islands) == 1


def cabin(frame=True):
    """A hollow cabin on a deck, its door through the front wall (framed by a ring 0.8 mm proud)."""
    deck = manifold3d.Manifold.cube((40, 40, 4))
    shell = (manifold3d.Manifold.cube((16, 16, 13)).translate((12, 12, 4))
             - manifold3d.Manifold.cube((13, 13, 11.5)).translate((13.5, 13.5, 4)))
    door = (manifold3d.Manifold.cube((6, 4, 7)).translate((17, 11, 4))
            + manifold3d.Manifold.cylinder(4, 3, 3, 48).rotate((-90, 0, 0)).translate((20, 11, 11)))
    ring = (manifold3d.Manifold.cube((8, 0.8, 7)).translate((16, 11.2, 4))
            + manifold3d.Manifold.cylinder(0.8, 4, 4, 48).rotate((-90, 0, 0)).translate((20, 11.2, 11)))
    m = deck + shell + (ring if frame else manifold3d.Manifold()) - door
    return m.refine(3), deck


def painted_outside(c, frame=True):
    outer = (np.abs(c[:, 0] - 20) > 8 - 1e-4) | (np.abs(c[:, 1] - 20) > 8 - 1e-4) | (c[:, 2] > 17 - 1e-4)
    in_frame = (((np.abs(c[:, 0] - 20) < 4.01) & (c[:, 2] < 11))
                | (np.hypot(c[:, 0] - 20, c[:, 2] - 11) < 4.01)) & (c[:, 1] < 13.6) & frame
    return (c[:, 2] > 4 + 1e-6) & outer & ~in_frame


@pytest.mark.parametrize("frame", [False, pytest.param(True, marks=pytest.mark.xfail(
    reason="known: with every edge sharp, a bare frame standing 0.8 mm proud stays on the base"))])
def test_cabin_painted_outside_comes_off_whole_at_its_foot(frame):
    # paint only outside, the door left bare: the whole cabin, inside walls included, is cut
    # off at the deck
    m, deck = cabin(frame)
    result, _ = run(m, lambda c: painted_outside(c, frame))
    part = result.inlays[2]
    assert part.volume() == pytest.approx(m.volume() - deck.volume(), rel=5e-3)
    assert watertight(part) and watertight(result.base)
    assert not result.findings.surface
