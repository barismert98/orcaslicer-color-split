import numpy as np

from color_split.stl import read_stl, write_stl


def test_round_trip(tmp_path):
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    write_stl(tmp_path / "t.stl", V, F)
    V2, F2 = read_stl(tmp_path / "t.stl")
    assert len(V2) == 4 and len(F2) == 4
    assert np.allclose(np.sort(V2, axis=0), np.sort(V, axis=0))


def test_solids_touching_along_an_edge_export_watertight():
    # two blocks meeting only along one edge (thin inlay sheets did this on a Benchy cabin):
    # an STL reader welds the shared edge into four faces unless the copies are pulled apart
    import manifold3d
    import numpy as np

    from color_split.solids import edge_components, export_mesh, stl_defects
    a = manifold3d.Manifold.cube((10, 10, 0.6))
    b = manifold3d.Manifold.cube((10, 10, 0.6)).translate((10, 10, 0))
    V, F = export_mesh(a + b)
    assert stl_defects(V, F) == 0
    assert len(np.unique(edge_components(F))) == 2
