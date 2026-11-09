import tracemalloc

import numpy as np
import pytest

from color_split import basepaint, features, paint


@pytest.mark.parametrize("state", [1, 2, 3, 4, 16, 17, 18, 20])
def test_leaf_code_round_trips(state):
    tri = np.eye(3)
    _, states = paint.leaf_triangles(tri, basepaint.leaf_code(state))
    assert states.tolist() == [state]


def square(z=0.0):
    V = np.array([[0, 0, z], [10, 0, z], [10, 10, z], [0, 10, z]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def surface(states):
    V, F = square()
    return basepaint.Surface(V, F, np.array(states), np.tile([0.0, 0, 1], (2, 1)))


def test_exact_triangle_keeps_its_code_and_corner_order():
    V, F = square()
    out_F = F[:, [1, 2, 0]]  # the same triangles, starting at another corner
    code = "00083"
    F2, codes = basepaint.transfer(V, out_F, V, F, [code, None], np.array([1, 1]), surface([2, 1]), 1)
    assert codes == [code, None]
    assert F2.tolist() == F.tolist()


def test_fragment_takes_the_paint_under_it():
    out_V = np.array([[1, 8, 0], [1, 2, 0], [9, 8, 0]], float)  # across the diagonal: no match
    out_V = out_V[[1, 2, 0]]
    V, F = square()
    _, codes = basepaint.transfer(out_V, np.array([[0, 1, 2]]), V, F, [None, None], np.array([1, 1]),
                                  surface([2, 2]), 1)
    _, states = paint.leaf_triangles(out_V[[0, 1, 2]], codes[0])
    assert set(states.tolist()) == {2}


def test_fragment_across_a_paint_border_is_subdivided():
    out_V = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)  # half of each source triangle
    V, F = square()
    _, codes = basepaint.transfer(out_V, np.array([[0, 1, 2]]), V, F, [None, None], np.array([1, 1]),
                                  surface([1, 2]), 1)
    leaves, states = paint.leaf_triangles(out_V, codes[0])
    area = 0.5 * np.linalg.norm(np.cross(leaves[:, 1] - leaves[:, 0], leaves[:, 2] - leaves[:, 0]), axis=1)
    assert area[states == 2].sum() == pytest.approx(25, rel=0.15)
    assert area[states == 0].sum() == pytest.approx(25, rel=0.15)


def test_new_faces_off_the_surface_are_not_painted():
    out_V, out_F = square(0.02)  # a cap, lifted off the painted surface
    V, F = square()
    _, codes = basepaint.transfer(out_V, out_F, V, F, [None, None], np.array([1, 1]), surface([2, 2]), 1)
    assert codes == [None, None]


def grid_leaves(n):
    """A triangle grid in which every other triangle was split in four: T-junctions everywhere."""
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    p = np.stack([x, y, 0 * x], axis=2)
    a, b, c, d = p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]
    whole = np.stack([a, b, c], axis=2).reshape(-1, 3, 3)
    t = np.stack([a, c, d], axis=2).reshape(-1, 3, 3)
    ab, bc, ca = 0.5 * (t[:, 0] + t[:, 1]), 0.5 * (t[:, 1] + t[:, 2]), 0.5 * (t[:, 2] + t[:, 0])
    parts = [np.stack(c, axis=1) for c in ((t[:, 0], ab, ca), (ab, t[:, 1], bc), (bc, t[:, 2], ca), (ab, bc, ca))]
    return np.concatenate([whole, *parts])


def test_conform_fans_t_junctions_and_keeps_area():
    big = np.array([[[0, 0, 0], [2, 0, 0], [0, 2, 0]]], float)
    split = np.array(paint._children(np.array([[2, 0, 0], [0, 2, 0], [2, 2, 0]], float), 3, 0))
    leaves = np.concatenate([big, split])
    V, F, parent = features.conform(leaves)
    area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    assert area.sum() == pytest.approx(4)
    assert area[parent == 0].sum() == pytest.approx(2)
    assert (parent == 0).sum() == 4  # the big triangle was fanned around the T-junction
    a, b, _ = features.edge_pairs(F)
    assert len(a) >= 5  # interior edges all shared


def test_conform_memory_stays_flat():
    leaves = grid_leaves(150)  # 111 thousand leaves, a third of them fanned
    tracemalloc.start()
    features.conform(leaves)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / len(leaves) < 1300, f"{peak / len(leaves):.0f} bytes per leaf"
