import numpy as np
import pytest

from color_split.paint import PaintFormatError, leaf_triangles, split_mesh

TRI = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], float)


def tri_area(c):
    return 0.5 * np.linalg.norm(np.cross(c[1] - c[0], c[2] - c[0]))


# --- reference encoder (inverse of the verified OrcaSlicer format) ---------------------------
# tree: int state for a leaf, or (split_sides, special_side, [child_0, ..., child_split]).

def _leaf_nibbles(state):
    if state < 3:
        return [state << 2]
    if state <= 17:
        return [0b1100, state - 3]
    return [0b1100, 0b1111, state - 18]


def _nibbles(tree):
    if isinstance(tree, int):
        return _leaf_nibbles(tree)
    split, special, children = tree
    out = [split | ((0 if split == 3 else special) << 2)]
    for child in reversed(children):  # serialized child index split .. 0
        out += _nibbles(child)
    return out


def encode(tree):
    return "".join("0123456789ABCDEF"[n] for n in reversed(_nibbles(tree)))


def mid(a, b):
    return 0.5 * (a + b)


# --- tests ------------------------------------------------------------------------------------

def test_leaf_state_simple():
    corners, states = leaf_triangles(TRI, "4")
    assert states.tolist() == [1] and np.allclose(corners[0], TRI)


def test_extended_states():
    assert leaf_triangles(TRI, "0C")[1].tolist() == [3]
    assert leaf_triangles(TRI, encode(17))[1].tolist() == [17]
    assert leaf_triangles(TRI, encode(20))[1].tolist() == [20]


def test_split1_geometry():
    v0, v1, v2 = TRI
    corners, states = leaf_triangles(TRI, encode((1, 0, [1, 2])))
    assert states.tolist() == [1, 2]
    assert np.allclose(corners[0], [v0, v1, mid(v1, v2)])
    assert np.allclose(corners[1], [mid(v1, v2), v2, v0])


def test_split1_special_side_rotates():
    v0, v1, v2 = TRI
    corners, _ = leaf_triangles(TRI, encode((1, 1, [1, 2])))
    # rotated start: (v1, v2, v0) -> midpoint of (v2, v0)
    assert np.allclose(corners[0], [v1, v2, mid(v2, v0)])


def test_split2_and_split3_geometry():
    v0, v1, v2 = TRI
    corners, _ = leaf_triangles(TRI, encode((2, 0, [1, 2, 1])))
    a, b = mid(v0, v1), mid(v0, v2)
    assert np.allclose(corners, [[v0, a, b], [a, v1, b], [v1, v2, b]])
    corners, states = leaf_triangles(TRI, encode((3, 0, [1, 2, 1, 2])))
    a, b, c = mid(v0, v1), mid(v1, v2), mid(v2, v0)
    assert np.allclose(corners, [[v0, a, c], [a, v1, b], [b, v2, c], [a, b, c]])
    assert states.tolist() == [1, 2, 1, 2]


def _random_tree(rng, depth):
    if depth == 0 or rng.random() < 0.35:
        return int(rng.integers(0, 20))
    split = int(rng.integers(1, 4))
    special = 0 if split == 3 else int(rng.integers(0, 3))
    return (split, special, [_random_tree(rng, depth - 1) for _ in range(split + 1)])


def _leaves(tree):
    return [tree] if isinstance(tree, int) else [s for c in tree[2] for s in _leaves(c)]


def test_random_round_trip_conserves_area():
    rng = np.random.default_rng(0)
    for _ in range(200):
        tree = _random_tree(rng, 4)
        corners, states = leaf_triangles(TRI, encode(tree))
        assert states.tolist() == _leaves(tree)
        assert np.isclose(sum(tri_area(c) for c in corners), tri_area(TRI))
        bary = corners[..., :2] / 2.0  # TRI spans x, y in [0, 2] with x + y <= 2
        assert (bary >= -1e-12).all() and (bary.sum(-1) <= 1 + 1e-12).all()


def test_orca_example_string_decodes():
    corners, states = leaf_triangles(TRI, "1C0C2C0C1C13")
    assert len(corners) == len(states) >= 4
    assert np.isclose(sum(tri_area(c) for c in corners), tri_area(TRI))


@pytest.mark.parametrize("bad", ["3", "G", "", "7"])
def test_malformed_raises(bad):
    with pytest.raises(PaintFormatError):
        leaf_triangles(TRI, bad)


def test_split_mesh_keeps_unpainted_whole():
    V = np.vstack([TRI, TRI + [5, 0, 0]])
    F = np.array([[0, 1, 2], [3, 4, 5]])
    corners, states = split_mesh(V, F, [None, encode((1, 0, [2, 0]))])
    assert states.tolist() == [0, 2, 0]
    assert np.allclose(corners[0], TRI)
