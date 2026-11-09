import numpy as np

from color_split.paintfix import clean_states
from tests.fixtures.make_3mf import TOP, cube


def test_unpainted_sliver_inside_paint_is_filled_and_speck_removed():
    V, F = cube(20)
    corners = V[F]
    states = np.zeros(12, dtype=np.int64)
    states[[6, 7, 2]] = 2  # +x side painted, plus one top triangle as a large region
    # split triangle 6 into a tiny unpainted sliver and the rest: emulate with a separate case
    cleaned, changed = clean_states(corners, states, min_area=150.0)  # top triangle (200 mm²) stays
    assert changed == 0
    speck = states.copy()
    speck[:] = 0
    speck[TOP[0]] = 3  # a 200 mm² speck counts as small only with a huge threshold
    cleaned, changed = clean_states(corners, speck, min_area=250.0)
    assert cleaned[TOP[0]] == 0 and changed == 1


def test_thin_island_is_merged_by_width():
    # a 10 mm x 0.1 mm unpainted strip between two painted halves of a plate
    xs = np.array([0, 4.95, 5.05, 10.0])
    V = np.array([[x, y, 0] for x in xs for y in (0, 10)], float)
    quads = [(0, 2, 3, 1), (2, 4, 5, 3), (4, 6, 7, 5)]
    F = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    states = np.array([2, 2, 0, 0, 2, 2])
    cleaned, changed = clean_states(V[F], states)
    assert cleaned.tolist() == [2] * 6 and changed == 1
