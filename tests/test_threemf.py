from pathlib import Path

import numpy as np
import pytest

from color_split.threemf import ThreeMFError, list_objects, read_object
from tests.fixtures.make_3mf import TOP, Obj, cube, write_3mf

DATA = Path(__file__).parent / "data"


def rot_z90():
    M = np.eye(4)
    M[:2, :2] = [[0, -1], [1, 0]]
    M[:3, 3] = [100, 0, 0]
    return M


def test_reads_by_index_with_duplicate_names_and_transform(tmp_path):
    V, F = cube()
    codes = [None] * 12
    codes[TOP[0]] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, extruder=1),
                                   Obj("cube", V, F, codes, extruder=3, transform=rot_z90())])
    assert list_objects(tmp_path / "p.3mf") == [(0, "cube"), (1, "cube")]
    obj = read_object(tmp_path / "p.3mf", 1)
    assert obj.index == 1 and obj.extruder == 3
    assert obj.codes[TOP[0]] == "8" and obj.codes[0] is None
    expected = V @ rot_z90()[:3, :3].T + rot_z90()[:3, 3]
    assert np.allclose(obj.V, expected)
    assert np.array_equal(obj.F, F)


def test_missing_object_and_file(tmp_path):
    V, F = cube()
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F)])
    with pytest.raises(ThreeMFError):
        read_object(tmp_path / "p.3mf", 5)
    with pytest.raises(ThreeMFError):
        read_object(tmp_path / "nope.3mf", 0)


@pytest.mark.skipif(not (DATA / "painted_cube.3mf").exists(), reason="owner fixture not provided")
def test_real_orca_fixture():
    objects = list_objects(DATA / "painted_cube.3mf")
    assert objects
    obj = read_object(DATA / "painted_cube.3mf", 0)
    assert any(obj.codes) and len(obj.codes) == len(obj.F)


def test_mirrored_transform_keeps_outward_winding(tmp_path):
    from color_split.solids import object_manifold
    V, F = cube()
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, transform=np.diag([-1.0, 1, 1, 1]))])
    obj = read_object(tmp_path / "p.3mf", 0)
    assert object_manifold(obj.V, obj.F).volume() == pytest.approx(8000)


def test_index_counts_objects_not_instances(tmp_path):
    V, F = cube()
    codes = [None] * 12
    codes[TOP[0]] = "8"
    write_3mf(tmp_path / "p.3mf", [Obj("A", V, F, instances=2), Obj("B", V, F, codes)])
    assert list_objects(tmp_path / "p.3mf") == [(0, "A"), (1, "B")]
    assert read_object(tmp_path / "p.3mf", 1).name == "B"


def test_only_normal_parts_are_read(tmp_path):
    V, F = cube()
    write_3mf(tmp_path / "p.3mf", [Obj("cube", V, F, modifier=cube(5))])
    obj = read_object(tmp_path / "p.3mf", 0)
    assert len(obj.F) == 12
